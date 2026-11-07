# PP2 / PP5 — Beyond the Heatmap: quantifying what the model actually looks at

Working title of the paper: *Beyond the Heatmap: quantifying shortcut reliance in chest X-ray models against radiologist bounding boxes*.

Status as of 2026-11-07: **design locked, recon done, no results.** One recon script committed (`notebooks/pp5_recon_bbox.py`), one replication notebook built and never run (`notebooks/pp5_efficientnet_b4.py`).

Naming note: this project has carried two identifiers. It was PP5 "Beyond the Heatmap" in the original stack and is PP2 "Where CNNs look" in the current chain. Same recon, same design, same target.

## The question

Does shortcut reliance in a fully supervised chest X-ray CNN predict where, and whether, the model fails to localize disease, measured against radiologist bounding boxes and stratified by whether the disease is actually in the suspected region?

PP1 produced a qualitative version of this claim: Grad-CAM sat on a stereotyped right upper lobe blob, and attention collapsed to background on errors. That claim is hedged in PP1 because it rests on 20 true positives and 20 false negatives per dataset, read by eye. This project turns it into a number with a control.

## Data

123 Infiltration bounding boxes from `BBox_List_2017.csv`, the radiologist-annotated subset of NIH ChestX-ray14, matched to the 224 px image set.

Recon, verified: 123 boxes, all matched to an image, none out of bounds. Spatial split **22 boxes true in the right upper lobe (~18%), 101 elsewhere**. That 18% is the anchor for the naive control.

## Design, locked

- **Primary metric:** pointing game accuracy, threshold free. Does the peak of the Grad-CAM map land inside the radiologist box. Robust, one number, defensible.
- **Secondary:** IoU sweep, binarization at 10%, 20% and 30% of max activation, plus Otsu.
- **Centerpiece:** per-image correlation between the Grad-CAM peak and the box centroid. A lesion-driven model's attention tracks the lesion. A shortcut model's does not.
- **Cleanest test:** stratify by right-upper-lobe-true (n = 22) versus not (n = 101). High accuracy on the first group with collapse on the second is close to proof of guessing the lobe rather than reading the image.
- **Controls:** an always-guess-right-upper-lobe heuristic, ignoring the image entirely, expected to score about 18% from the base rate; and a box-area-weighted chance baseline.
- **Causal check:** paste the lesion into a different location on a clean background with a Poisson blend, re-run, and see whether the peak follows the transplanted lesion. Sham paste as the control.
- **Robustness:** bootstrap confidence intervals over the 123 images, a training-split leak check on the box images, and a paired bootstrap or McNemar comparison against EfficientNet-B4 to test whether the shortcut belongs to the architecture or to the training paradigm.

## Honesty rules for this project

- n = 123. Intervals are wide. Everything is reported as a range, never as a single number stated as a conclusion.
- If attention does follow the lesion, that is a real result, it is publishable, and it strengthens PP1's "consistent with" hedge rather than weakening the portfolio. Do not hunt for the answer the story wants.
- Scope is one question: does attention follow the lesion. Nothing else gets added.

## What exists

| Piece | State |
|---|---|
| Box recon, spatial split, 18% base rate | done, verified |
| DenseNet Grad-CAM pipeline | exists, inherited from PP1 |
| EfficientNet-B4 replication notebook | built, 12/12 self-checks pass, never run |
| Sweep over the 123 images | not run |
| Light `pp5-bbox-123` Kaggle dataset | not confirmed built, blocks the committed sweep |
| Controls, statistics, figures | not started |
| Write-up | not started |

Blocking gotcha, learned the hard way in PP1: every external dataset must be attached as a notebook input before a committed run. kagglehub cannot first-time-download inside a commit, and the run dies with a backend error that looks like a data problem.

## Done when

Pointing game, stratification, correlation and both controls are reported with bootstrap intervals, for DenseNet-121 and EfficientNet-B4, and the numbers are in a preprint with a repo that reproduces them.
