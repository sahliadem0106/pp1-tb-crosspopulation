# Results

> Drafted from logged, verified numbers (Table 1, Phase 7 Grad-CAM, Phase 6
> BiomedCLIP). Follows `experiments.md`. Edit freely — this is a scaffold.

## 3.1 Home held-out performance

The fine-tuned DenseNet achieved a test AUC of **0.715** (95% CI [0.707, 0.725])
on the held-out NIH test split (n = 17,260; 3,016 positives / 14,244 negatives),
with precision 0.313, recall 0.574, and F1 0.405 at a 0.5 threshold. This is
consistent with prior reports for the Infiltration-vs-No-Finding proxy
(Wang et al. ~0.66; CheXNet 0.7345).

## 3.2 Cross-population external evaluation (Table 1)

Evaluated frozen on the three external datasets, the model transferred **unevenly**:

| Dataset | n | AUC | 95% CI | P@0.5 | R@0.5 | F1@0.5 | WHO thr | WHO sens | WHO spec |
|---|---|---|---|---|---|---|---|---|---|
| NIH (home) | 17,260 | 0.715 | [0.707, 0.725] | 0.313 | 0.574 | 0.405 | 0.203 | 0.900 | 0.320 |
| Montgomery | 138 | 0.739 | [0.650, 0.813] | 0.917 | 0.190 | 0.314 | 0.065 | 0.914 | 0.112 |
| Shenzhen | 662 | 0.769 | [0.734, 0.802] | 0.882 | 0.199 | 0.325 | 0.073 | 0.902 | 0.383 |
| Qatar (MENA) | 4,200 | 0.571 | [0.546, 0.594] | 0.230 | 0.319 | 0.267 | 0.088 | 0.903 | 0.139 |

**Two striking patterns emerge.**

First, on two of the three external populations the model performed **at or above
its home score**: Shenzhen (AUC 0.769) was significantly higher than NIH
(bootstrap difference CI [−0.093, −0.020], i.e. NIH − Shenzhen < 0), and
Montgomery (0.739) was comparable but noisier given its small size (CI crosses
zero). That a model trained on NLP-derived proxy labels generalizes to *real*
TB cases on two populations — despite different scanners, populations, and
countries — is a positive-transfer result.

Second, the model **failed on the MENA dataset**: Qatar AUC was 0.571, near
chance and statistically well below every other dataset (vs NIH, difference CI
[+0.116, +0.170]). The Qatar confidence interval [0.546, 0.594] is fully
disjoint from all others.

**WHO-operability failure.** At the WHO-recommended screening sensitivity
(≥ 0.90), specificity collapsed on **every** dataset, including the home set:
0.320 (NIH), 0.112 (Montgomery), 0.383 (Shenzhen), and 0.139 (Qatar). The
thresholds required to reach 90% sensitivity were very low (0.065–0.203),
meaning the model effectively labels any image with measurable signal as
positive. No dataset could be screened at WHO sensitivity without referring
roughly 62–89% of healthy patients.

## 3.3 What the model attends to (Grad-CAM)

Grad-CAM on true positives from both external datasets showed attention
concentrated **within the lung parenchyma**, with no reliance on corner markers,
laterality text, or image borders — ruling out the most blatant
burned-in-annotation shortcut. However, the highlighted region was **stereotyped**:
near-identical size, shape, and right-upper/mid-lung position across patients.
Right-upper-lobe involvement is the most common presentation of post-primary TB,
so this pattern is consistent with a coarse, epidemiologically plausible spatial
prior — but not with lesion-specific localization, since the fixed geometry
across patients suggests the model weights a lung region rather than detecting
discrete pathology.

False negatives exhibited **two distinct failure mechanisms**: (i) attention
collapsing *outside* the lungs (most often a horizontal bottom-border band, and
in some cases a shoulder hotspot), and (ii) lung-region activation that was
present and anatomically plausible but too weak to cross the 0.5 threshold
(borderline-confidence failure). Raw (un-normalized) activation magnitudes
confirmed that the border response was genuine rather than a min-max rescaling
artifact; a modest border response (≈ 8–11% of peak activation) was present in
all false negatives examined and became dominant when lung evidence was weak.

## 3.4 A zero-shot foundation model succeeds where the specialist fails (Table 2)

Using BiomedCLIP zero-shot (no training, text-prompt scoring), we compared the
foundation model against the fine-tuned specialist:

| Dataset | DenseNet (fine-tuned) | BiomedCLIP (zero-shot) |
|---|---|---|
| NIH | 0.715 | 0.633 |
| Montgomery | 0.739 | **0.850** |
| Shenzhen | **0.769** | 0.650 |
| Qatar (MENA) | 0.571 | **0.838** (color-neutralized) |

The result is striking and **artifact-controlled**: on the target MENA population,
a foundation model with zero training achieved AUC **0.838** — a +0.267 gain over
the fine-tuned specialist — and this held after color cues were removed from the
Qatar input (0.838 vs 0.849 with color, Δ = 0.011). No single model wins
everywhere (the specialist retains NIH and Shenzhen), but **on the population of
clinical interest, the generalist wins decisively.**

## 3.5 Representation geometry (UMAP)

UMAP projections of the DenseNet features colored by dataset formed a single
continuous manifold with no hard boundaries; **Qatar was not an island** but
overlapped extensively with NIH. The same overlap held in BiomedCLIP feature
space, where Montgomery and Shenzhen formed distinct sub-clusters. This indicates
the Qatar failure is **not** attributable to gross out-of-distribution
separation of the data — consistent with a decision-process explanation rather
than a representation-space one.
