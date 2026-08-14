# Cross-Population TB Screening — Proxy-Trained Model Fails on the Target MENA Population

A controlled cross-population evaluation of a proxy-trained chest X-ray model for
tuberculosis. A DenseNet-121 trained on NIH ChestX-ray14 (Infiltration proxy) is
evaluated **frozen** on three external datasets — Montgomery (USA), Shenzhen
(China), and Qatar (MENA) — and compared against a zero-shot foundation model
(BiomedCLIP).

## Key findings

- **Uneven transfer**: the model matches/beats its home score on Shenzhen
  (AUC 0.769 vs 0.715) but collapses to near chance on Qatar (AUC 0.571).
- **Clinical**: at WHO-recommended sensitivity (≥ 0.90), specificity collapses on
  every dataset (0.11–0.38) — not usable as a standalone screener.
- **Mechanism**: Grad-CAM shows a stereotyped right-upper-lung spatial prior whose
  attention collapses to background on errors; UMAP shows Qatar is *not*
  out-of-distribution in feature space → the failure is a **decision-process
  shortcut, not distribution shift**.
- **Foundation model**: zero-shot BiomedCLIP succeeds on the same Qatar images
  (AUC 0.838, color-neutralized) — +0.267 over the fine-tuned specialist.

## Project layout

```
PP1/
├── paper/            ← manuscript (md + LaTeX/LNCS), figures, notes, writing plan
├── notebooks/        ← NIH training + external-eval/UMAP/BiomedCLIP notebooks
├── results/          ← metrics CSVs + figures (checkpoints excluded via .gitignore)
├── research_log.md   ← dated decisions (the Methods source of truth)
├── data/             ← dataset notes (medical images never committed)
└── src/              ← shared modules
```

## Reproduce

1. `notebooks/pp1_nih_training.ipynb` — train the DenseNet-121 on NIH.
2. `notebooks/phase5_phase6_external_eval_umap_biomedclip.ipynb` — external
   evaluation (Table 1), Grad-CAM, UMAP, and the BiomedCLIP zero-shot comparison
   (Table 2). Run on Google Colab (CPU/GPU); datasets via `kagglehub`.

## Paper

Full manuscript: `paper/manuscript_claude.md` (markdown) and
`paper/pp1_dali.tex` (Springer LNCS, for MICCAI DALI). Prepared for a medRxiv
preprint, targeted at MICCAI DALI 2026.

## Author

Adem Sahli — Higher Institute of Computer Science (ISI), Tunisia.
ORCID: 0009-0003-0749-0575
