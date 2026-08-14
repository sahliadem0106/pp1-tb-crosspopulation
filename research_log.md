# PP1 Research Log — every decision, dated

> Rule: log every decision with a date. This becomes the Methods section and protects against forgetting.

## 2026-08-13 — PHASE 5 EXTERNAL RESULTS (THE HEADLINE) — NIH-trained DenseNet121 (test AUC 0.7153), frozen, evaluated on 3 external populations:
- **FINAL TABLE 1:** NIH test 17260: AUC 0.715 [0.707-0.725], P 0.313/R 0.574/F1 0.405, WHO t=0.203 sens 0.900 spec 0.320. Montgomery 138: AUC 0.739 [0.650-0.813], P 0.917/R 0.190/F1 0.314, WHO t=0.065 sens 0.914 spec 0.112. Shenzhen 662: AUC 0.769 [0.734-0.802], P 0.882/R 0.199/F1 0.325, WHO t=0.073 sens 0.902 spec 0.383. Qatar 4200: AUC 0.571 [0.546-0.594], P 0.230/R 0.319/F1 0.267, WHO t=0.088 sens 0.903 spec 0.139.
- **THREE FINDINGS (CI-based):** (1) Shenzhen significantly BEATS NIH (CI fully above) — proxy transfers better than home score; confirmed labels + severity likely dominate. (2) Montgomery ≈ NIH, point estimate higher, too noisy (n=138). (3) Qatar near chance, CI fully disjoint from all others — THE cross-population failure, dataset-specific, real. (4) Bonus: even NIH fails WHO-operability (spec 0.32 at sens 0.90) — not deployable anywhere.
- **SIGNIFICANCE (bootstrap diff CIs, n_boot=200):** NIH vs Shenzhen diff CI [-0.093, -0.020] → SIGNIFICANT (Shenzhen beats NIH). NIH vs Qatar [+0.116, +0.170] → SIGNIFICANT (Qatar fails). NIH vs Montgomery [-0.106, +0.073] → not significant (n=138). **Results section statistically complete.**
- **Paper narrative:** transfers rank-wise to real-TB populations but fails on MENA (Qatar) and cannot meet WHO operating requirements anywhere. Discussion: label quality, severity bias, dataset-specific shift, Qatar artifact caveat.
- **Environment note:** moved to Google Colab (Kaggle phone verification blocked). Data via kagglehub (public datasets, no token needed); checkpoint uploaded via files.upload(); CPU inference fine. Outputs in /content, download via Colab Files panel → Documents/PP1/results/.

## 2026-08-13 — PHASE 6 UMAP + BIOMEDCLIP ZERO-SHOT (the foundation-model comparison):
- **BiomedCLIP zero-shot AUCs (no training, hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, prompts "infiltration"/"no abnormality"):** NIH 0.633 (vs DenseNet 0.715), Montgomery 0.850 (vs 0.739), Shenzhen 0.650 (vs 0.769), Qatar RGB 0.849 / **color-neutralized 0.838** (vs DenseNet 0.571).
- **HEADLINE: zero-shot foundation model beats the fine-tuned specialist on the target MENA population by +0.267 (0.838 vs 0.571) — artifact-controlled (color-neutralized ≈ RGB, Δ0.011). The Qatar grayscale signal EXISTS; the specialist's shortcut features don't transfer. Pretraining diversity beats proxy-label fine-tuning for cross-population generalization.**
- **Nuance for paper:** no single model wins everywhere — specialist wins NIH (0.715>0.633) + Shenzhen (0.769>0.650); generalist wins Montgomery (0.850) + Qatar (0.838). Qatar reported color-neutralized per golden rule (0.838), RGB 0.849 as note.
- **UMAP (DenseNet features + BiomedCLIP comparison, colored by dataset):** **HYPOTHESIS KILLED — Qatar is NOT an island in either space.** One continuous manifold in DenseNet space; Qatar diffuse + overlapping NIH everywhere. BiomedCLIP: NIH/Qatar intermixed; M/S separate MORE (tight green crescent / orange arc) but separation does NOT track AUC (Montgomery isolated + gain, Shenzhen isolated + drop). Minority Qatar subset forms an island in BiomedCLIP space (subgroup footnote idea: colored subset?).
- **MECHANISTIC SYNTHESIS (ties Phases 5+7+6):** Qatar data is NOT foreign (UMAP overlap) → the specialist's failure is DECISION-side: its stereotyped RUL spatial prior (Grad-CAM) can't exploit the overlapping space where zero-shot BiomedCLIP succeeds (0.838). "Cross-population failure = shortcut artifact of the decision process, NOT distribution shift." → natural setup for PP5.
- **Deliverables:** umap_comparison.png + table2_zero_shot.csv → PP1/results/. Notes: umap_notes_claude.md (PP1/paper/).
## 2026-08-13 — PHASE 7 GRAD-CAM (NIH model on Montgomery + Shenzhen) — 10 TPs + 10 FNs per dataset, model.features (7×7) layer, tensor-hook pattern (inplace-relu fix), one-line unsqueeze fix for interpolate:
- **TPs (both datasets):** heat in lung parenchyma, NO corner markers/text/borders → blatant marker shortcut ruled out. BUT stereotyped right-upper/mid-lung blob, near-identical shape/size across patients → coarse spatial prior, NOT lesion-specific localization. RUL is the textbook most-common TB site → prior is epidemiologically plausible (ambiguity must be in paper).
- **FNs (both datasets):** TWO failure modes — (a) attention collapses OUTSIDE lungs (bottom-border band / shoulder hotspot), (b) lung evidence present but under-confident (below 0.5). Both modes exist in BOTH datasets.
- **Raw-magnitude verification (Montgomery FNs 0104/0108/0113):** 0104 border 0.0114 vs lungs -0.0002 → border attention RAW-CONFIRMED (min-max artifact caveat defeated). 0108/0113: lungs = raw max → under-confident mode. Persistent modest border response 8-11% of peak in all three → takes over when lung evidence weak.
- **Figure 2 regenerated as full 4-row montage (was single Shenzhen-FN panel) + independently verified by Claude:** completeness PASS (4×10). Two blank frames (Montgomery FN #9 MCUCXR_0182_1, Shenzhen FN #2 CHNCXR_0328_1) raw-checked → ReLU'd sum = 0.00000, all activations negative → genuine NO-EVIDENCE cases (strongest FNs), not a bug → strengthens attention-collapse story. Figure 2 VERIFIED complete + correct; caption updated in manuscript.
- **Paper claim (honest):** localizes to parenchyma (no marker shortcut); stereotyped RUL prior consistent with TB epidemiology but not lesion-specific; FN attention collapse → candidate explanation for Qatar failure (spatial/laterality shortcut that fails to transfer). NOT "learned TB pathology."
- **Deliverables:** gradcam_section.md (publication draft) + gradcam_notes_claude.md (raw analysis + verification) in PP1/paper/.

## 2026-08-11
- **First NIH held-out test results (DenseNet121, frozen early layers + last block, pos_weight ≈ 3, lr 1e-4, Adam, cosine, best-checkpoint by dev AUC):**
  - **Test AUC 0.6951** | TP 1926 / FP 5303 / FN 1090 / TN 8941 (n_test 17,260)
  - Precision 0.266 | Recall 0.639 | F1 0.376 @ threshold 0.5
  - Dev curve: peaked 0.676 @ epoch 4, then overfit (train loss ↓, dev AUC ↓) → early stopping worked as designed.
  - Literature context: Infiltration vs No Finding AUC ≈ 0.66 (Wang 2017) – 0.73 (CheXNet). Our 0.695 is in-zone.
  - **Training notebook COMPLETE** (data → model → train → eval → ROC). Recipe proven; artifacts regenerable (checkpoint was lost to a Kaggle session wipe — lesson: download/commit immediately after every run).
- **FULL FINE-TUNE RUN (7M params, AdamW + weight_decay 1e-4, RandomHorizontalFlip, T_max 25, patience 6):**
  - **Test AUC 0.7153** | TP 1732 / FP 3795 / FN 1284 / TN 10449
  - Precision 0.313 | Recall 0.574 | F1 0.405 @ threshold 0.5
  - Dev curve: climbed 0.686 → peak **0.7138** (epoch 10/13) → declined; early stop at 19. Overfitting wall moved epoch 4 → 13 (regularization worked); plateau ~0.71 = single-task ceiling at 224.
  - Δ vs partial run: AUC +0.020, F1 +0.029, model less aggressive (R↓ P↑). Now ≈ CheXNet-level for this task. **Adopted as the paper model.**
- **Decision: stop chasing NIH AUC (ceiling reached ~0.71-0.72 single-task). Proceed to Phase 5 (external eval) with this checkpoint.**
- **Notes:** patient-level split → row counts vary per split (test got 21.5% of rows: 23.6% of negatives vs 15.2% of positives, ratio 1:4.7). Not a bug; patient exclusivity is the guarantee. Possible upgrade later: stratified patient split.
- **Next:** commit training notebook (Save & Run All = artifact bridge) → Phase 5 external eval notebook (Montgomery/Shenzhen/Qatar frozen inference + WHO sweep → Table 1).

## 2026-08-10
- **Project skeleton created** — this folder layout (README, research_log, data/, notebooks/, results/, paper/, src/).
- **Montgomery + Shenzhen sourced** via Kaggle raddar mirrors (`raddar/tuberculosis-chest-xrays-montgomery` 613MB, `raddar/tuberculosis-chest-xrays-shenzhen` 3.77GB). No OpenI download needed. Labels encoded in filename suffix (`_0` = normal, `_1` = TB) — verify counts on first load.
- **NIH sources confirmed**: iteration = `khanfashee/nih-chest-x-ray-14-224x224-resized` (2.47GB, includes `Data_Entry_2017.csv` with Patient ID); paper run = `biditdas06/nih-chestxray14` (45GB, GCloud $300).
- **NIH proxy-label rule decided**: positive = "Infiltration" in Finding Labels; negative = "No Finding" only (multi-label CSV; drop multi-finding images from negatives). CheXNet-style.
- **Split decided**: patient-level 70/15/15 stratified by Patient ID (prevents patient leakage — upgrade over the Qatar notebook's random image split).

## 2026-08-09
- **Qatar TB investigation CLOSED.** Full chain of reasoning: `la future si on vivre/tb-investigation-full-log.md`. Key outcomes:
  - Color artifact confirmed at file level (TB ~half colored vs Normal all gray) — latent for our pipeline (grayscale transform verified R==G==B).
  - Grayscale signal weak (entropy AUC 0.84).
  - **"0.98 grayscale → multi-source leak" RETRACTED** — stale-kernel run; do not cite.
  - Grad-CAM on the stabilized run: TPs on lung fields, FNs diffuse/uncertain → real anatomy, not shortcut.
  - Stabilized run: 89% acc / TB precision 1.00 / recall 0.36 (seed 42, lr 1e-4, grad-clipped; effectively unweighted — weights computed but never passed to the loss).
- **Decision: Qatar TB = external test 3 ONLY, never training.** Reported numbers carry the caveat (Limitations paragraph).
- **Fix list established** for the NIH pipeline: held-out eval, weighted loss wired in, freeze early layers + lr 1e-4 + cosine decay, best-checkpoint by dev AUC, per-class P/R/F1, seeds + grad clipping.
- **Training instability lesson**: same code, unseeded → recall swung 0.43/0.60/~1.00/0.19/0.36 across runs; one total collapse (19% acc). Seed everything.
- Annotated teaching notebook produced: `Downloads/qatar-tb-annotated.ipynb` (code identical to the run).

## 2026-08-08
- **PadChest DROPPED** as external test 3 (agreement wait not worth it; not a TB set anyway). Replaced by **Qatar TB** (Kaggle tawsifurrahman, 700 TB / 3500 Normal, Hamad Medical — MENA proxy).
- Chose DenseNet121 + EfficientNet-B4 per project spec; NIH Infiltration vs No Finding as TB proxy.
