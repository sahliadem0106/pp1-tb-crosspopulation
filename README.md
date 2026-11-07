# Cross-population evaluation of a proxy-trained tuberculosis model

A controlled cross-population evaluation of a chest X-ray model for tuberculosis screening. A DenseNet-121 is trained on NIH ChestX-ray14 using the *Infiltration* finding as a proxy label for tuberculosis, then evaluated **frozen** on three external collections and compared against a zero-shot foundation model (BiomedCLIP).

This repository is the anchor artifact of a larger measurement program: whether medical vision models actually look at the disease, and whether that survives a change of population. The program is described in [`docs/research_line.md`](docs/research_line.md).

## Status

Experiment complete, manuscript written, **no DOI yet**.

- medRxiv submission was **rejected at the scope screen** (student project), not on scientific content. Manuscript ID MS MEDRXIV/2026/360550. It is not published there and must not be cited as if it were.
- Next venue: preprints.org or ResearchSquare for an immediate DOI, then arXiv (needs an endorser), then the MICCAI DALI workshop.
- Known open item: the provenance of the third external collection does not support the population claim the manuscript currently makes. See the note below and [`paper/provenance_check.md`](paper/provenance_check.md).

## Findings

- **Uneven transfer.** The model matches or beats its home score on Shenzhen (AUC 0.769 vs 0.715) but collapses to near chance on the third collection (AUC 0.571, confidence interval disjoint from every other set).
- **Not deployable anywhere.** At the WHO operating point of 90% sensitivity, specificity falls to 0.11 to 0.38 on all four sets, home included.
- **Mechanism.** Grad-CAM shows a stereotyped right upper lobe spatial prior whose attention collapses to background on errors. UMAP shows the failing collection is not out of distribution in feature space. The failure is consistent with a decision-process shortcut rather than distribution shift.
- **Foundation model.** Zero-shot BiomedCLIP succeeds on the same images where the fine-tuned specialist fails (AUC 0.838, colour-neutralised). No single model wins everywhere, which is the point.

## Read the provenance note before quoting the third column

The third external collection is **not a single-population cohort**. Its own data record (IEEE DataPort, DOI 10.21227/mps8-kb56) lists its sources as the NLM Montgomery and Shenzhen sets, a Belarus NIAID drug-resistance collection, the NIAID TB portal, and RSNA pneumonia. Only the curators are Qatari; the patients are not.

Two consequences, both recorded in [`paper/provenance_check.md`](paper/provenance_check.md):

- the population framing in the current manuscript is being withdrawn, and
- RSNA pneumonia derives from NIH ChestX-ray14, which is our own training corpus, so part of the "external" normals may be in-distribution until a perceptual-hash dedup is run.

The headline result does not depend on the population claim. The name of the third collection does.

## Layout

```
PP1/
├── paper/            manuscript (markdown and LaTeX/LNCS), figures, provenance check
├── docs/             the research line, the protocol note, the PP2 design
├── notebooks/        NIH training, external eval + UMAP + BiomedCLIP, the third-set audit, PP2 recon
├── results/          metrics CSVs, prediction files, figures
├── research_log.md   dated decisions, the Methods source of truth
└── requirements.txt
```

## Reproduce

1. `notebooks/pp1_nih_training.ipynb` trains the DenseNet-121 on NIH ChestX-ray14 (Infiltration vs No Finding, patient-level 70/15/15 split).
2. `notebooks/external_eval_umap_biomedclip.ipynb` produces Table 1 (frozen external evaluation plus the WHO sweep), Table 2 (BiomedCLIP zero-shot), the UMAP comparison and the Grad-CAM panels.
3. `notebooks/qatar_investigation.ipynb` is the closed audit of the third collection's colour artifact and the retracted 0.98 claim.

Run on Kaggle or Colab. Datasets come from Kaggle mirrors via `kagglehub`; images are never committed. Every external dataset must be attached as a notebook input before a committed run, because kagglehub cannot first-time-download inside a commit.

The model checkpoint and the packed artifacts are attached to the GitHub release rather than committed, since both are 28 MB.

## Known gaps

- The clean-negative retrain notebook, which produced the ablation behind `results/paired_clean_vs_mixed_test.csv`, is not in this repository. Only the joined per-image CSV survived. The statistics reproduce from it; the training does not.
- `table2_zero_shot.csv` was never written to disk. The Table 2 numbers exist in the notebook output and in the manuscript.
- The perceptual-hash deduplication of the third collection has not been run. Until it is, the paper states that image-level overlap with the training corpus and the other external sets cannot be excluded.

## Author

Adem Sahli — Higher Institute of Computer Science (ISI), Tunisia.
ORCID: [0009-0003-0749-0575](https://orcid.org/0009-0003-0749-0575)

## License

Code is under the MIT License ([`LICENSE`](LICENSE)). The manuscript, documentation and figures are under CC BY 4.0 ([`LICENSE-docs`](LICENSE-docs)).
