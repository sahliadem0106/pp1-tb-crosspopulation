# %% [markdown]
# We replicate the PP1 training protocol with a second architecture, EfficientNet-B4, to check whether the cross-population finding, in particular the collapse on the Qatari MENA cohort, depends on the network architecture. EfficientNet-B4 is larger than DenseNet-121 and uses compound scaling, so it is a meaningfully different inductive bias. We train it on the same NIH split, then evaluate it on the held-out NIH test set and on the Qatar dataset for a direct comparison with the DenseNet result.

# %%
import os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
from torchvision import transforms

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)

# %% [markdown]
# The data. We read the NIH metadata file and keep two classes only, images that show Infiltration and images with no finding.

# %%
import kagglehub
NIH = kagglehub.dataset_download("khanfashee/nih-chest-x-ray-14-224x224-resized")
import glob
csv_candidates = glob.glob(os.path.join(NIH, "**", "Data_Entry_2017.csv"), recursive=True)
assert csv_candidates, "Data_Entry_2017.csv not found"
df = pd.read_csv(csv_candidates[0])
mask = df["Finding Labels"].str.contains("Infiltration") | (df["Finding Labels"] == "No Finding")
df = df[mask].reset_index(drop=True)
print("rows after filtering:", len(df))

img_candidates = glob.glob(os.path.join(NIH, "**", "images-224", "**", "*.png"), recursive=True)
print("image files found:", len(img_candidates))
IMG_DIR = os.path.dirname(os.path.dirname(img_candidates[0]))
print("IMG_DIR:", IMG_DIR)

# %% [markdown]
# Patient split. We split by patient ID so the same person never lands in more than one set, keeping the evaluation clean.

# %%
patients = df["Patient ID"].unique()
g = torch.Generator().manual_seed(42)
patients = patients[torch.randperm(len(patients), generator=g).numpy()]
n_train = int(0.7 * len(patients)); n_dev = int(0.1 * len(patients))
train_pat, dev_pat, test_pat = patients[:n_train], patients[n_train:n_train+n_dev], patients[n_train+n_dev:]
train_df = df[df["Patient ID"].isin(train_pat)].reset_index(drop=True)
dev_df   = df[df["Patient ID"].isin(dev_pat)].reset_index(drop=True)
test_df  = df[df["Patient ID"].isin(test_pat)].reset_index(drop=True)
print("train:", len(train_df), "dev:", len(dev_df), "test:", len(test_df))

# %%
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = torchvision.io.read_image(os.path.join(self.img_dir, row["Image Index"])).float() / 255.0
        label = 1.0 if "Infiltration" in row["Finding Labels"] else 0.0
        return self.transform(img), torch.tensor(label, dtype=torch.float32)

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_ds = ChestXRayDataset(train_df, IMG_DIR, train_tf)
dev_ds   = ChestXRayDataset(dev_df,   IMG_DIR, eval_tf)
test_ds  = ChestXRayDataset(test_df,  IMG_DIR, eval_tf)
train_dloader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
dev_dloader   = DataLoader(dev_ds,   batch_size=32, shuffle=False, num_workers=2)
test_dloader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)
print("loaders ready")

# %% [markdown]
# Model setup. We load a pretrained EfficientNet-B4 and swap the classifier head for a single-output logistic head. EfficientNet-B4's final feature dimension is 1792.

# %%
model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
model.classifier = nn.Linear(1792, 1)
model = model.to(device)
print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# %% [markdown]
# This is the training loop. We use a weighted loss because the classes are imbalanced, add weight decay and a cosine decay schedule on the learning rate, log one row per epoch, and save the best model by development AUC with early stopping.

# %%
num_epochs = 25
patience   = 6
best_dev_auc = 0.0
no_improve = 0
def manual_auc(labels, probs):
    y = torch.tensor(labels, dtype=torch.float32)
    p = torch.tensor(probs, dtype=torch.float32)
    p_pos = p[y == 1]; p_neg = p[y == 0]
    higher = (p_pos.unsqueeze(1) > p_neg.unsqueeze(0)).sum().float()
    tied   = (p_pos.unsqueeze(1) == p_neg.unsqueeze(0)).sum().float()
    return ((higher + 0.5 * tied) / (len(p_pos) * len(p_neg))).item()

n_pos = int(train_df["Finding Labels"].str.contains("Infiltration").sum())
n_neg = len(train_df) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
with open("/kaggle/working/training_log.csv", "w") as f:
    f.write("epoch,train_loss,dev_auc,lr\n")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    model.eval()
    dev_probs, dev_labels = [], []
    with torch.no_grad():
        for img, lbl in dev_dloader:
            img = img.to(device)
            p = torch.sigmoid(model(img)).cpu()
            dev_probs.extend(p.numpy().ravel())
            dev_labels.extend(lbl.numpy())
    dev_auc = manual_auc(dev_labels, dev_probs)
    print(f"Epoch {epoch+1} | train loss: {running_loss/len(train_dloader):.4f} | dev AUC: {dev_auc:.4f}")
    with open("/kaggle/working/training_log.csv", "a") as f:
        f.write(f"{epoch+1},{running_loss/len(train_dloader):.4f},{dev_auc:.4f},{scheduler.get_last_lr()[0]:.6f}\n")
    if dev_auc > best_dev_auc:
        best_dev_auc = dev_auc
        torch.save(model.state_dict(), "/kaggle/working/best_model.pt")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break
    scheduler.step()

# %% [markdown]
# Evaluation. We reload the saved best model and run it over the held-out NIH test split to get the home AUC.

# %%
model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
model.classifier = nn.Linear(1792, 1)
model.load_state_dict(torch.load("/kaggle/working/best_model.pt"))
model = model.to(device)
model.eval()

def predict(dloader):
    probs, labels = [], []
    with torch.no_grad():
        for img, lbl in dloader:
            img = img.to(device)
            p = torch.sigmoid(model(img)).cpu()
            probs.extend(p.numpy().ravel())
            labels.extend(lbl.numpy())
    return np.array(probs), np.array(labels)

test_probs, test_labels = predict(test_dloader)
from sklearn.metrics import roc_auc_score
print("n_test:", len(test_labels), "| positives:", int(test_labels.sum()), "| negatives:", int((test_labels == 0).sum()))
print("NIH Test AUC:", roc_auc_score(test_labels, test_probs))

# %% [markdown]
# Cross-population check. We evaluate the same trained model on the Qatari MENA dataset to compare directly against the DenseNet result (AUC 0.571). This is the single most important number for the architecture question.

# %%
QATAR = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
q_images = glob.glob(os.path.join(QATAR, "**", "*.[jJ][pP][gG]"), recursive=True)
q_pos = [p for p in q_images if "Tuberculosis" in p]
q_neg = [p for p in q_images if "Normal" in p]
print("Qatar TB:", len(q_pos), "Normal:", len(q_neg))

q_probs, q_labels = [], []
with torch.no_grad():
    for path in q_pos:
        img = torchvision.io.read_image(path).float() / 255.0
        if img.shape[0] != 3:
            img = img[:1].repeat(3, 1, 1)
        img = eval_tf(img).unsqueeze(0).to(device)
        q_probs.append(torch.sigmoid(model(img)).item()); q_labels.append(1)
    for path in q_neg:
        img = torchvision.io.read_image(path).float() / 255.0
        if img.shape[0] != 3:
            img = img[:1].repeat(3, 1, 1)
        img = eval_tf(img).unsqueeze(0).to(device)
        q_probs.append(torch.sigmoid(model(img)).item()); q_labels.append(0)
q_probs, q_labels = np.array(q_probs), np.array(q_labels)
print("Qatar AUC:", roc_auc_score(q_labels, q_probs))

# %%
torch.save(model.state_dict(), "/kaggle/working/efficientnet_b4_best.pt")
pd.DataFrame({"prob": q_probs, "label": q_labels}).to_csv("/kaggle/working/qatar_effnet_predictions.csv", index=False)
print("saved efficientnet_b4_best.pt + qatar_effnet_predictions.csv")
