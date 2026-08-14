# =====================================================================
# PP1 — Phase 5 (external evaluation) + Phase 6 (UMAP + BiomedCLIP)
# Google Colab notebook — the COMPLETE working pipeline
# (evaluation on Montgomery/Shenzhen/Qatar + foundation-model comparison)
#
# exposed. For public datasets kagglehub needs no token; drop the line
# or use a fresh token.
# =====================================================================

!pip install -q kaggle

import os


!pip install -q kagglehub

# =========================
# COLAB SETUP
# =========================

!pip install -q kagglehub

import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from PIL import Image
from collections import Counter

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
else:
    print("WARNING: No GPU detected.")

# =========================
# DOWNLOAD DATASETS
# =========================

import kagglehub

M_root = kagglehub.dataset_download(
    "raddar/tuberculosis-chest-xrays-montgomery"
)

S_root = kagglehub.dataset_download(
    "raddar/tuberculosis-chest-xrays-shenzhen"
)

Q_root = kagglehub.dataset_download(
    "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
)

print("Montgomery:", M_root)
print("Shenzhen:", S_root)
print("Qatar:", Q_root)

# =========================
# DATASET PATHS
# =========================

M_Dir = os.path.join(M_root, "images", "images")
S_Dir = os.path.join(S_root, "images", "images")
Q_Dir = os.path.join(Q_root, "TB_Chest_Radiography_Database")

print("M:", M_Dir)
print("S:", S_Dir)
print("Q:", Q_Dir)

print("\nMontgomery:", os.listdir(M_Dir)[:3])
print("Shenzhen:", os.listdir(S_Dir)[:3])
print("Qatar:", os.listdir(Q_Dir)[:5])

# =========================
# UPLOAD MODEL CHECKPOINT
# =========================

from google.colab import files

uploaded = files.upload()

CHECKPOINT_PATH = "/content/best_model.pt"

print("Checkpoint exists:", os.path.exists(CHECKPOINT_PATH))
print("Checkpoint:", CHECKPOINT_PATH)

def make_label_csv(img_dir, csv_path):
    # scan a folder..read the label from each filename's last digit, save as CSV.
    rows = []

    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith(".png"):
            label = int(
                filename.rsplit("_", 1)[1].split(".")[0]
            )

            rows.append({
                "Image Index": filename,
                "label": label
            })

    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False)

    print(df["label"].value_counts().to_dict())

    return df


mont_df = make_label_csv(
    M_Dir,
    "/content/montgomery_labels.csv"
)

shen_df = make_label_csv(
    S_Dir,
    "/content/shenzhen_labels.csv"
)

def make_folder_csv(base_dir, csv_path):
    rows = []

    for label, folder in [
        (1, "Tuberculosis"),
        (0, "Normal")
    ]:
        for filename in sorted(
            os.listdir(os.path.join(base_dir, folder))
        ):
            if filename.endswith(".png"):

                rows.append({
                    "Image Index": f"{folder}/{filename}",
                    "label": label
                })

    if len(rows) == 0:
        raise ValueError(
            f"No .png files found in {base_dir} — check the path!"
        )

    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False)

    print(df["label"].value_counts().to_dict())

    return df


qatar_df = make_folder_csv(
    Q_Dir,
    "/content/qatar_labels.csv"
)

class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = f"{self.img_dir}/{row['Image Index']}"

        img = Image.open(path).convert("L")

        img = self.transform(img)

        return img, torch.tensor(
            row["label"],
            dtype=torch.float32
        )

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

mont_ds = ChestXRayDataset(
    mont_df,
    M_Dir,
    eval_transform
)

shen_ds = ChestXRayDataset(
    shen_df,
    S_Dir,
    eval_transform
)

qatar_ds = ChestXRayDataset(
    qatar_df,
    Q_Dir,
    eval_transform
)

mont_dloader = DataLoader(
    mont_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

shen_dloader = DataLoader(
    shen_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

qatar_dloader = DataLoader(
    qatar_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

model = models.densenet121(weights=None)

model.classifier = nn.Linear(1024, 1)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("Checkpoint loaded ✅")
print("Model device:", next(model.parameters()).device)

def run_inference(model, dloader):
    probs, labels = [], []

    model.eval()

    with torch.no_grad():

        for img, lbl in dloader:

            img = img.to(device)

            p = torch.sigmoid(
                model(img)
            ).cpu()

            probs.extend(
                p.numpy().ravel()
            )

            labels.extend(
                lbl.numpy()
            )

    return probs, labels

mont_probs, mont_labels = run_inference(
    model,
    mont_dloader
)

shen_probs, shen_labels = run_inference(
    model,
    shen_dloader
)

qatar_probs, qatar_labels = run_inference(
    model,
    qatar_dloader
)

print(
    "Sizes:",
    len(mont_probs),
    len(shen_probs),
    len(qatar_probs)
)

# ---- metric helpers (both functions in one cell) ----
def manual_auc(labels, probs):
    y = torch.tensor(labels, dtype=torch.float32)
    p = torch.tensor(probs, dtype=torch.float32)
    p_pos = p[y == 1]
    p_neg = p[y == 0]
    higher = (p_pos.unsqueeze(1) > p_neg.unsqueeze(0)).sum().float()
    tied   = (p_pos.unsqueeze(1) == p_neg.unsqueeze(0)).sum().float()
    return ((higher + 0.5 * tied) / (len(p_pos) * len(p_neg))).item()

def who_operating_point(labels, probs, target=0.90):
    y = torch.tensor(labels, dtype=torch.float32)
    p = torch.tensor(probs, dtype=torch.float32)
    for t in torch.linspace(1.0, 0.0, 1000):
        preds = (p >= t).float()
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tn = ((preds == 0) & (y == 0)).sum()
        rec = tp / (tp + fn)
        if rec >= target:
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            return t.item(), rec.item(), spec.item(), prec.item()
    return None
def bootstrap_auc_ci(labels, probs, n_boot=200, seed=42):
    y = torch.tensor(labels, dtype=torch.float32)
    p = torch.tensor(probs, dtype=torch.float32)
    n = len(y)
    g = torch.Generator().manual_seed(seed)
    aucs = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)          # resample with replacement
        aucs.append(manual_auc(y[idx].numpy(), p[idx].numpy()))
    aucs = torch.tensor(aucs)
    return aucs.quantile(0.025).item(), aucs.quantile(0.975).item()

# CHUNK 1 — the WHO operating point finder (the new function of this phase)
def who_operating_point(labels, probs, target=0.90):
    """Highest threshold where recall >= target. Returns (t, recall, specificity, precision)."""
    y = torch.tensor(labels, dtype=torch.float32)
    p = torch.tensor(probs, dtype=torch.float32)
    for t in torch.linspace(1.0, 0.0, 1000):          # high → low (most specific first)
        preds = (p >= t).float()
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tn = ((preds == 0) & (y == 0)).sum()
        rec = tp / (tp + fn)
        if rec >= target:                              # WHO met — this is the operating point
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            return t.item(), rec.item(), spec.item(), prec.item()
    return None                                        # unreachable (t=0 → recall 1.0)

# CHUNK 2 — the full report per dataset (the raw material of Table 1)
for name, probs, labels in [("Montgomery", mont_probs, mont_labels),
                            ("Shenzhen",   shen_probs, shen_labels),
                            ("Qatar",      qatar_probs, qatar_labels)]:
    auc = manual_auc(labels, probs)
    t, rec, spec, prec = who_operating_point(labels, probs)
    print(f"{name:10s} | AUC {auc:.3f} | WHO point: t={t:.3f} sens={rec:.3f} spec={spec:.3f} prec={prec:.3f}")



import os, pandas as pd, matplotlib.pyplot as plt

DATA = "/content" # Define DATA path
OUT = "/content"   # Define OUT path

# NIH test curve — ONLY if you uploaded test_predictions.csv to Drive (from Documents/PP1/results/)
datasets = []
nih = pd.read_csv("/content/test_predictions.csv")
if os.path.exists(f"{DATA}/test_predictions.csv"):
    nih = pd.read_csv(f"{DATA}/test_predictions.csv")
    datasets.append(("NIH test", nih["prob"].tolist(), nih["true_label"].tolist()))
datasets = [("NIH test",    nih["prob"].tolist(), nih["true_label"].tolist()),
            ("Montgomery",  mont_probs, mont_labels),
            ("Shenzhen",    shen_probs, shen_labels),
            ("Qatar",       qatar_probs, qatar_labels)]

def roc_points(labels, probs, n_thr=300):
    y = torch.tensor(labels, dtype=torch.float32)
    p = torch.tensor(probs, dtype=torch.float32)
    tprs, fprs = [], []
    for t in torch.linspace(1.0, 0.0, n_thr):
        preds = (p >= t).float()
        tp = ((preds == 1) & (y == 1)).sum(); fn = ((preds == 0) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum(); tn = ((preds == 0) & (y == 0)).sum()
        tprs.append((tp / (tp + fn)).item() if (tp + fn) > 0 else 1.0)
        fprs.append((fp / (fp + tn)).item() if (fp + tn) > 0 else 0.0)
    return fprs, tprs

plt.figure(figsize=(6, 6))
for name, probs, labels in datasets:
    fprs, tprs = roc_points(labels, probs)
    plt.plot(fprs, tprs, label=f"{name} (AUC {manual_auc(labels, probs):.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random (0.5)")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.title("ROC — NIH-trained model on external populations")
plt.savefig(f"{OUT}/roc_overlay.png", dpi=150, bbox_inches="tight")   # ⭐ Drive, not the VM
plt.show()
print("Saved to Drive:", f"{OUT}/roc_overlay.png")

rows = []
for name, probs, labels in datasets:
    auc = manual_auc(labels, probs)
    lo, hi = bootstrap_auc_ci(labels, probs)
    y = torch.tensor(labels, dtype=torch.float32)
    p = torch.tensor(probs, dtype=torch.float32)
    preds = (p >= 0.5).float()
    tp = ((preds == 1) & (y == 1)).sum().item(); fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item(); tn = ((preds == 0) & (y == 0)).sum().item()
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    t, sens, spec, wprec = who_operating_point(labels, probs)
    rows.append({
        "Dataset": name, "n": len(labels),
        "AUC": round(auc, 3), "CI95": f"[{lo:.3f}, {hi:.3f}]",
        "P@0.5": round(prec, 3), "R@0.5": round(rec, 3), "F1@0.5": round(f1, 3),
        "WHO_thr": round(t, 3), "WHO_sens": round(sens, 3), "WHO_spec": round(spec, 3),
    })

table1 = pd.DataFrame(rows)
table1.to_csv(f"{OUT}/table1_external_eval.csv", index=False)     # ⭐ Drive
print(table1.to_string(index=False))
print("Saved to Drive:", f"{OUT}/table1_external_eval.csv")

# ---- Significance: bootstrap 95% CI of (AUC_NIH - AUC_external) ----
nih_probs  = nih["prob"].tolist()
nih_labels = nih["true_label"].tolist()

def bootstrap_auc_diff(labels1, probs1, labels2, probs2, n_boot=200, seed=42):
    """95% CI for (AUC1 - AUC2). Significant if the CI excludes 0."""
    y1, p1 = torch.tensor(labels1, dtype=torch.float32), torch.tensor(probs1, dtype=torch.float32)
    y2, p2 = torch.tensor(labels2, dtype=torch.float32), torch.tensor(probs2, dtype=torch.float32)
    n1, n2 = len(y1), len(y2)
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_boot):
        i1 = torch.randint(0, n1, (n1,), generator=g)     # resample dataset 1
        i2 = torch.randint(0, n2, (n2,), generator=g)     # resample dataset 2
        a1 = manual_auc(y1[i1].numpy(), p1[i1].numpy())
        a2 = manual_auc(y2[i2].numpy(), p2[i2].numpy())
        diffs.append(a1 - a2)
    diffs = torch.tensor(diffs)
    return diffs.quantile(0.025).item(), diffs.quantile(0.975).item()

for name, p2, l2 in [("Shenzhen",  shen_probs, shen_labels),
                     ("Qatar",     qatar_probs, qatar_labels),
                     ("Montgomery", mont_probs, mont_labels)]:
    lo, hi = bootstrap_auc_diff(nih_labels, nih_probs, l2, p2)
    sig = "SIGNIFICANT ✅" if not (lo <= 0 <= hi) else "not significant ❌"
    print(f"NIH vs {name:10s}: diff CI [{lo:+.3f}, {hi:+.3f}] → {sig}")

import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()              # capture [1,1024,7,7]
        if output.requires_grad:                        # ⭐ the inplace-relu fix:
            output.register_hook(self._save_gradient)   # hook the TENSOR, not the module

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(self, img_tensor):
        """img_tensor: [1,3,224,224] on device → heatmap [224,224] (0-1)."""
        self.model.zero_grad()
        out = self.model(img_tensor)                    # [1,1] logit
        out.backward()                                  # gradients to the feature layer
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)     # [1,1024,1,1]
        cam = (weights * self.activations).sum(dim=1)               # [1,7,7]
        cam = torch.relu(cam)                           # positive evidence only
        cam = cam.unsqueeze(1)                           # ⭐ [1,1,7,7] — add channel dim
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze()                              # back to [224,224]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)    # 0-1
        return cam.detach().cpu().numpy()

cam = GradCAM(model, model.features)

def collect_tf(df, probs, labels, thresh=0.5, n=10):
    """Return (tp_image_names, fn_image_names) — TB cases only, up to n each."""
    tp_names, fn_names = [], []
    for (_, row), p, l in zip(df.iterrows(), probs, labels):
        if l == 1:                                  # TB cases only
            if p >= thresh and len(tp_names) < n:
                tp_names.append(row["Image Index"])
            elif p < thresh and len(fn_names) < n:
                fn_names.append(row["Image Index"])
        if len(tp_names) >= n and len(fn_names) >= n:
            break
    return tp_names, fn_names

mont_tp, mont_fn = collect_tf(mont_df, mont_probs, mont_labels)
shen_tp, shen_fn = collect_tf(shen_df, shen_probs, shen_labels)
print("Montgomery TPs:", len(mont_tp), "| FNs:", len(mont_fn))
print("Shenzhen  TPs:", len(shen_tp), "| FNs:", len(shen_fn))

def gradcam_row(img_dir, image_names, title, n=10):
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    for i, name in enumerate(image_names):
        img = Image.open(os.path.join(img_dir, name)).convert("L")
        tensor = eval_transform(img).unsqueeze(0).to(device)
        heatmap = cam.generate(tensor)          # [224,224] 0-1

        orig = np.array(img.resize((224, 224))) / 255.0
        overlay = np.stack([orig]*3, axis=-1)           # gray → 3-channel
        heat_rgb = plt.cm.jet(heatmap)[:, :, :3]        # heatmap → colors
        blended = 0.5 * overlay + 0.5 * heat_rgb        # ⭐ blend

        axes[i].imshow(blended)
        axes[i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    return fig

gradcam_row(M_Dir, mont_tp, "Montgomery — True Positives (correctly caught TB)")
gradcam_row(M_Dir, mont_fn, "Montgomery — False Negatives (MISSED TB)")
gradcam_row(S_Dir, shen_tp, "Shenzhen — True Positives")
gradcam_row(S_Dir, shen_fn, "Shenzhen — False Negatives")
plt.savefig("/content/gradcam_montgomery_shenzhen.png", dpi=150, bbox_inches="tight")

def raw_cam_grid(img_tensor):
    """Raw 7x7 activation sum (no ReLU, no norm) — the truth before rescaling."""
    cam.model.zero_grad()
    out = cam.model(img_tensor)
    out.backward()
    w = cam.gradients.mean(dim=(2, 3), keepdim=True)
    return (w * cam.activations).sum(dim=1).squeeze().detach().cpu()   # [7,7]

for name in mont_fn[:3]:
    img = Image.open(os.path.join(M_Dir, name)).convert("L")
    t = eval_transform(img).unsqueeze(0).to(device)
    grid = raw_cam_grid(t)
    print(name)
    print("  raw max:",    round(grid.max().item(), 4))
    print("  bottom row:", round(grid[-1].mean().item(), 4), "| lungs:", round(grid[1:-2].max().item(), 4))

!pip install -q umap-learn open_clip_torch

# ---- NIH test dataframe + image dir (from the training run's predictions) ----
import kagglehub
NIH_root = kagglehub.dataset_download("khanfashee/nih-chest-x-ray-14-224x224-resized")   # ~2.5GB, few min
NIH_DIR = os.path.join(NIH_root, "images-224", "images-224")
print("NIH images exist:", os.path.isdir(NIH_DIR))

nih_df = pd.read_csv("/content/test_predictions.csv")          # Image Index, true_label, prob
print("columns:", nih_df.columns.tolist())                     # ⭐ SEE the real names first
nih_df = nih_df.rename(columns={"true_label": "label"})        # match the universal format
print(nih_df["label"].value_counts().to_dict())                # expect {0: 14244, 1: 3016}

def embed_densenet(df, img_dir, max_n=None, seed=42):
    """Global-pooled, relu'd DenseNet features (1024-d) per image."""
    if max_n is not None and max_n < len(df):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(df), generator=g)[:max_n].tolist()
        df = df.iloc[idx].reset_index(drop=True)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            img = Image.open(os.path.join(img_dir, row["Image Index"])).convert("L")
            t = eval_transform(img).unsqueeze(0).to(device)
            f = torch.relu(model.features(t)).mean(dim=(2, 3))   # ⭐ relu, then pool
            feats.append(f.squeeze().cpu().numpy())
            labels.append(row["label"])
    return np.array(feats), np.array(labels)

nih_emb, nih_lbl = embed_densenet(nih_df, NIH_DIR, max_n=2000)
mont_emb,  mont_lbl  = embed_densenet(mont_df, M_Dir)           # all 138
shen_emb,  shen_lbl  = embed_densenet(shen_df, S_Dir)           # all 662
qatar_emb, qatar_lbl = embed_densenet(qatar_df, Q_Dir, max_n=2000)

import umap

all_emb = np.vstack([nih_emb, mont_emb, shen_emb, qatar_emb])
all_lbl = ["NIH"]*len(nih_emb) + ["Montgomery"]*len(mont_emb) + ["Shenzhen"]*len(shen_emb) + ["Qatar"]*len(qatar_emb)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
xy = reducer.fit_transform(all_emb)

plt.figure(figsize=(8, 6))
for ds, color in [("NIH", "tab:blue"), ("Montgomery", "tab:green"), ("Shenzhen", "tab:orange"), ("Qatar", "tab:red")]:
    mask = np.array(all_lbl) == ds
    plt.scatter(xy[mask, 0], xy[mask, 1], s=6, alpha=0.5, label=ds, color=color)
plt.legend(); plt.title("UMAP of DenseNet features — colored by dataset")
plt.savefig(f"{OUT}/umap_densenet.png", dpi=150, bbox_inches="tight"); plt.show()

import open_clip
model_bc, preprocess_bc, _ = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
model_bc = model_bc.to(device).eval()

def embed_biomedclip(df, img_dir, max_n=None, seed=42):
    if max_n is not None and max_n < len(df):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(df), generator=g)[:max_n].tolist()
        df = df.iloc[idx].reset_index(drop=True)
    feats, labels = [], []
    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            img = Image.open(os.path.join(img_dir, row["Image Index"])).convert("RGB")  # ⭐ RGB
            t = preprocess_bc(img).unsqueeze(0).to(device)
            f = model_bc.encode_image(t)
            feats.append(f.squeeze().cpu().numpy())
            labels.append(row["label"])
    return np.array(feats), np.array(labels)

def zero_shot_auc(df, img_dir, max_n=None, seed=42):
    if max_n is not None and max_n < len(df):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(df), generator=g)[:max_n].tolist()
        df = df.iloc[idx].reset_index(drop=True)
    prompts = ["a chest x-ray showing infiltration", "a chest x-ray showing no abnormality"]
    t_emb = model_bc.encode_text(tokenizer(prompts).to(device))
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    probs, labels = [], []
    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            img = Image.open(os.path.join(img_dir, row["Image Index"])).convert("RGB")
            t = preprocess_bc(img).unsqueeze(0).to(device)
            i_emb = model_bc.encode_image(t)
            i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
            logits = (i_emb @ t_emb.T) * model_bc.logit_scale.exp()
            probs.append(logits.softmax(dim=-1)[0, 0].item())     # P(infiltration)
            labels.append(row["label"])
    return manual_auc(labels, probs)

print("BiomedCLIP zero-shot AUCs (vs DenseNet):")
for name, df, d in [("NIH", nih_df, NIH_DIR), ("Montgomery", mont_df, M_Dir),
                    ("Shenzhen", shen_df, S_Dir), ("Qatar", qatar_df, Q_Dir)]:
    zs = zero_shot_auc(df, d, max_n=2000)
    print(f"{name:10s} | BiomedCLIP zero-shot: {zs:.3f}")

# ---- PHASE 6 PACKAGING ----
# 1) ⭐ FILL THESE with the 4 numbers the running loop prints:
zs_aucs = {"NIH": 0.633, "Montgomery": 0.850, "Shenzhen": 0.650, "Qatar": 0.849}

# 2) DenseNet AUCs recomputed from memory (no hardcoding)
dn = {"NIH":        manual_auc(nih_labels, nih_probs),
      "Montgomery": manual_auc(mont_labels, mont_probs),
      "Shenzhen":   manual_auc(shen_labels, shen_probs),
      "Qatar":      manual_auc(qatar_labels, qatar_probs)}

rows = [{"Dataset": k, "DenseNet AUC": round(dn[k], 3), "BiomedCLIP zero-shot AUC": v}
        for k, v in zs_aucs.items()]
pd.DataFrame(rows).to_csv("/content/table2_zero_shot.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

# 3) Manifest — everything in /content ready to download
import glob
print("\n📥 Files ready to download:")
for f in sorted(glob.glob("/content/*.png") + glob.glob("/content/*.csv")):
    print("  ", f)

# ---- COLOR-NEUTRALIZED Qatar zero-shot check ----
def zero_shot_auc_gray(df, img_dir, max_n=None, seed=42):
    """Same as zero_shot_auc, but grayscale-first → kills the color artifact."""
    if max_n is not None and max_n < len(df):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(df), generator=g)[:max_n].tolist()
        df = df.iloc[idx].reset_index(drop=True)
    prompts = ["a chest x-ray showing infiltration", "a chest x-ray showing no abnormality"]
    t_emb = model_bc.encode_text(tokenizer(prompts).to(device))
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    probs, labels = [], []
    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            img = Image.open(os.path.join(img_dir, row["Image Index"])).convert("L").convert("RGB")  # ⭐ gray first
            t = preprocess_bc(img).unsqueeze(0).to(device)
            i_emb = model_bc.encode_image(t)
            i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
            logits = (i_emb @ t_emb.T) * model_bc.logit_scale.exp()
            probs.append(logits.softmax(dim=-1)[0, 0].item())
            labels.append(row["label"])
    return manual_auc(labels, probs)

print("Qatar zero-shot, color-neutralized:", zero_shot_auc_gray(qatar_df, Q_Dir, max_n=2000))



# ---- BiomedCLIP UMAP — the comparison figure ----
# 1) embed all datasets (seed 42 → same images as the DenseNet panel)
bc_nih,  _ = embed_biomedclip(nih_df,  NIH_DIR, max_n=2000)
bc_mont, _ = embed_biomedclip(mont_df, M_Dir)
bc_shen, _ = embed_biomedclip(shen_df, S_Dir)
bc_qat,  _ = embed_biomedclip(qatar_df, Q_Dir, max_n=2000)

bc_all = np.vstack([bc_nih, bc_mont, bc_shen, bc_qat])
bc_lbl = (["NIH"]*len(bc_nih) + ["Montgomery"]*len(bc_mont) +
          ["Shenzhen"]*len(bc_shen) + ["Qatar"]*len(bc_qat))

# 2) UMAP fit
bc_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
bc_xy = bc_reducer.fit_transform(bc_all)

# 3) TWO panels: DenseNet space vs BiomedCLIP space
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {"NIH": "tab:blue", "Montgomery": "tab:green", "Shenzhen": "tab:orange", "Qatar": "tab:red"}

for ax, xy2, emb_lbl, title in [(axes[0], xy,     all_lbl, "DenseNet features"),
                                (axes[1], bc_xy,  bc_lbl,  "BiomedCLIP features")]:
    for ds, c in colors.items():
        m = np.array(emb_lbl) == ds
        ax.scatter(xy2[m, 0], xy2[m, 1], s=6, alpha=0.5, label=ds, color=c)
    ax.set_title(title)
    ax.legend()

plt.suptitle("Representation space — colored by dataset")
plt.savefig(f"{OUT}/umap_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", f"{OUT}/umap_comparison.png")
