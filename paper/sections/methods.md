# Methods

> Drafted from `research_log.md` (source of truth). Every number here is a logged,
> verified fact. Follows the Sida-Peng `method.md` guide. Edit freely — this is a scaffold.

## 2.1 Datasets

**Training set — NIH ChestX-ray14.** We trained on the NIH ChestX-ray14 dataset
(112,120 frontal chest radiographs) using the *Infiltration* finding as a proxy
label for tuberculosis, following the CheXNet protocol. Because NIH provides no
TB-specific labels, we defined the positive class as images whose multi-label
finding vector contained *Infiltration* and the negative class as images labeled
*No Finding*. Images with other finding combinations were excluded, yielding
19,870 positive and 60,412 negative images (ratio ≈ 1:3.0). Images were resized
to 224×224 pixels (the 224-resolution release).

**External test sets.** Three publicly available TB datasets were used for
zero-shot external evaluation, none of which was used for training:

- **Montgomery** (USA; n = 138; 80 normal / 58 TB) — from the Montgomery County
  Department of Health and Human Services.
- **Shenzhen** (China; n = 662; 326 normal / 336 TB) — from the Shenzhen No. 3
  People's Hospital.
- **Qatar TB** (MENA; n = 4,200; 3,500 normal / 700 TB) — assembled at Hamad
  Medical Corporation (Doha, Qatar), used here as a MENA-population proxy.
  This dataset carries a known acquisition artifact: a portion of the
  tuberculosis subset is colorized while the normal subset is uniformly
  grayscale. As detailed below, we evaluate on this set with color cues removed.

External dataset labels were encoded either in the filename suffix
(Montgomery/Shenzhen: `_0` = normal, `_1` = TB) or in the directory structure
(Qatar: `Normal/` vs `Tuberculosis/`). We generated per-dataset label manifests
(CSV) from these encodings and verified the resulting class counts against the
published dataset sizes before evaluation.

## 2.2 Preprocessing

All images were converted to grayscale and resized to 224×224. For the
DenseNet input we replicated the grayscale channel to three channels and applied
the ImageNet normalization (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224,
0.225]). During training only, a random horizontal flip was applied; the
external-evaluation transform contained no augmentation and was identical to the
training-set evaluation transform, ensuring a fair comparison across populations.

## 2.3 Model and training

**Architecture.** We fine-tuned a DenseNet-121 initialized with ImageNet weights;
the classification head was replaced with a single linear unit producing a logit
for the Infiltration proxy (BCE). All ~7.0M parameters were trainable
(full fine-tune).

**Training protocol.** NIH images were split **patient-wise** (70/15/15) by
stratified assignment of unique patient identifiers, preventing patient leakage
between train, development, and test. Training minimized
binary-cross-entropy-with-logits with class re-weighting
(`pos_weight = n_neg/n_pos ≈ 3.04`) to counter the ~1:3 imbalance, using AdamW
(lr = 1e-4, weight decay = 1e-4), a cosine annealing schedule (T_max = 25),
gradient clipping (max norm = 1.0), and early stopping (patience = 6 epochs) by
development-set AUC. The best checkpoint by development AUC was retained. Seeds
were fixed across Python, NumPy, and torch RNGs for reproducibility.

## 2.4 Evaluation

**Frozen inference.** The trained model was evaluated in `eval()` mode with
gradients disabled on the held-out NIH test split and on each external dataset.
Per image we recorded the sigmoid probability of the positive class.

**Metrics.** For each dataset we computed the area under the ROC curve (AUC)
using a pure-PyTorch rank computation, and precision, recall, and F1 at a 0.5
threshold. 95% confidence intervals (CIs) for AUC were obtained by bootstrap
resampling (n = 200 replicates; the pair-count method made larger replicates
expensive on CPU).

**WHO operating-point analysis.** Because tuberculosis screening tools are
required to achieve high sensitivity, we determined, for each dataset, the
highest decision threshold at which sensitivity ≥ 0.90 and report the
corresponding specificity and precision. This mirrors the WHO-recommended
screening sensitivity target and is the clinically meaningful operating point.

**Statistical significance.** Differences between dataset AUCs were assessed via
bootstrap confidence intervals on the AUC difference (AUC_NIH − AUC_external);
a difference was deemed significant when its 95% CI excluded zero.

## 2.5 Interpretability — Grad-CAM

To examine which image regions drove the model's decisions, we computed
gradient-weighted class activation maps (Grad-CAM; Selvaraju et al., 2017) from
the final convolutional feature map of the DenseNet (7×7). For each of
Montgomery and Shenzhen we selected 10 true positives and 10 false negatives
(at threshold 0.5). Heatmaps were upsampled to 224×224, rectified, min-max
normalized, and blended at 50% opacity over the input. For a subset of false
negatives we additionally inspected raw (un-normalized) activation magnitudes to
rule out rescaling artifacts. The backward hook was registered on the output
tensor to avoid the in-place-ReLU autograd conflict in DenseNet.

## 2.6 Foundation-model comparison

We compared the fine-tuned specialist against a large biomedical foundation
model, BiomedCLIP (400M parameters; pretrained on ~15M biomedical image-text
pairs), used **zero-shot**: images were scored by cosine similarity between the
image embedding and the text prompts "a chest x-ray showing infiltration" and
"a chest x-ray showing no abnormality", followed by a softmax over the two
prompts. AUC was computed from the infiltration-prompt probability. Because the
Qatar dataset contains a color acquisition artifact and BiomedCLIP accepts RGB
input, we additionally re-evaluated Qatar with color cues removed (grayscale
input), so the foundation-model comparison controls for the same artifact
neutralized in the DenseNet pipeline.

## 2.7 Implementation

Experiments were run in PyTorch on NVIDIA T4 GPU. External datasets were
obtained from their public repositories; the model, data-processing, and
evaluation code is available in the project repository. All AUC, threshold, and
significance computations were implemented in pure PyTorch without reliance on
scikit-learn.
