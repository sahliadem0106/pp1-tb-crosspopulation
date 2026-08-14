# PP1 — Paper Writing Plan (master roadmap)

> Status: PLAN · Owner: ademsahlii (first author) · Division: me=structure/draft, him=edit/judge, Claude=voice
> Skills used: `research-paper-drafting` (workflow/order) + `sida-peng-paper-writing` (section method) + `research-paper-writing` (Orchestra pipeline, 103KB)
> Deliverables live in `Documents/PP1/paper/`.

---

## 1. The paper's identity

**Working title:**
"Cross-population generalization of proxy-trained chest X-ray models for tuberculosis:
a specialist fails on the target MENA population where a zero-shot foundation model succeeds"

**Target venues (in order):**
1. **medRxiv preprint** (no review barrier, fast, citable, PubMed-indexed) — the version we polish first
2. **MICCAI DALI workshop** (LaTeX template on Overleaf) — the peer-reviewed attempt after

**Core narrative (the one-sentence thesis):**
> Cross-population TB screening failure is a **shortcut artifact of the decision process**,
> not distribution shift — and a zero-shot foundation model, free of that shortcut,
> succeeds on the population where the trained specialist fails.

**The four lines of evidence:**
1. Table 1: uneven transfer (Shenzhen 0.769 beats home 0.715; Qatar collapses to 0.571, CI fully disjoint); WHO-operability fails everywhere (spec 0.11–0.38)
2. Grad-CAM: stereotyped RUL spatial prior + attention collapse to background on errors (raw-verified)
3. UMAP: Qatar is NOT an island — overlap in both spaces → failure is decision-side, not data-side
4. Table 2: zero-shot BiomedCLIP succeeds on the same Qatar images (0.838 color-neutralized, vs 0.571)

## 2. Inputs (everything we already own)
- `research_log.md` — dated decisions (the Methods source of truth)
- Table 1 (external AUCs + CIs + WHO points + significance), Table 2 (BiomedCLIP zero-shot)
- Figures: ROC overlay (F1), Grad-CAM rows (F2), UMAP two-panel (F3)
- `paper/gradcam_section.md`, `gradcam_notes_claude.md`, `umap_notes_claude.md`
- Artifacts + predictions CSVs in `PP1/results/`

## 3. Locked honest-reporting decisions (do NOT waver)
- Qatar reported at **0.838 (color-neutralized)**, with RGB 0.849 as a Methods note
- Shenzhen **beats** NIH home score — report as a finding, not a bug
- WHO-operability failure is the clinical headline
- The mechanism claim is carefully worded: "consistent with decision-process shortcut, supported by Grad-CAM + UMAP + zero-shot" — never "we proved the model cheats"
- Limitations MUST own: proxy labels (Infiltration ≠ confirmed TB), Montgomery n=138 (wide CI), Qatar acquisition artifact, no clinical validation, single-specialist vs foundation-model confound

## 4. Section-by-section plan — WRITING ORDER (not paper order)

| Order | Section | Content | Skill guide | Status |
|---|---|---|---|---|
| 1 | **Methods** | Datasets (NIH train; M/S/Q external, Qatar artifact caveat), preprocessing (224, gray, ImageNet norm), model+training (DenseNet121 full fine-tune, AdamW, weight decay, cosine, pos_weight, patient split, early stop), evaluation (frozen AUC, bootstrap CIs, WHO sweep, DeLong/boot significance), Grad-CAM protocol, BiomedCLIP zero-shot | `method.md` | ⬜ next |
| 2 | **Results** | Table 1 + significance; Grad-CAM findings; Table 2 (foundation model) — transcribe from log/tables, largely done | `experiments.md` | ⬜ |
| 3 | **Related Work** | TB screening AI; cross-population generalization / distribution shift; shortcut learning; foundation models for medical imaging | `related-work.md` | ⬜ |
| 4 | **Introduction** | TB burden (WHO), MENA gap, AI promise, unproven generalization → the question | `introduction.md` | ⬜ |
| 5 | **Discussion** | The mechanism (shortcut not shift); foundation-model implication; tie Grad-CAM/UMAP to numbers (don't restate); future work = PP5 | — | ⬜ |
| 6 | **Conclusion** | Honest 1-paragraph summary + limitations + future (shortcut detection) | `conclusion.md` | ⬜ |
| 7 | **Abstract** | LITERALLY LAST. Version 2 structure (Challenge → Insight → Contribution → results). 200–250 words | `abstract.md` | ⬜ |

## 5. Methodology (from the installed skills)
- Mini-outline before prose; one paragraph = one message, stated in first sentence
- **Claim → Evidence → Status** map for every major claim (hard honesty gate)
- Reverse outlining after each section
- Load ONLY the section guide needed for the current edit (Sida-Peng execution rule)
- Five-dimension self-review before submission (contribution / clarity / experimental strength / evaluation completeness / method soundness)
- Adversarial review as a skeptical reviewer (`paper-review.md`)

## 6. Division of labor (hard rule)
- **Me:** draft every section from the log, structure, LaTeX, literature extraction (2-3 venue exemplar skeletons), applying the skills
- **Him:** edit every draft (each edit = his thinking), decide all claims, final voice
- **Claude:** final humanizer/voice pass + image-based figure reads (as already used for Grad-CAM/UMAP)

## 7. The step-by-step workflow
1. **Recon:** pull 2-3 real medRxiv/MICCAI cross-population medical-imaging papers → extract their section skeletons + lengths; grab the DALI LaTeX template
2. **Draft Methods** (`methods.md`) from the log, using `method.md` guide → him edit → Claude pass
3. **Results** (`results.md`) — transcribe Table 1 + Grad-CAM + Table 2 → him edit
4. **Related Work** — extract 2-3 exemplar skeletons → draft → him edit
5. **Introduction** → him edit
6. **Discussion + Conclusion** → him edit
7. **Abstract** (last, needs final numbers) → him edit
8. **References** via Zotero (from Related Work onward)
9. **Figure polish** (clean teaser + pipeline figure + minimal-ink tables)
10. **Self-review gate** (`paper-review.md`, 5-dimension) → fix
11. **Claude voice pass** → **convert to `.tex`** (DALI template)
12. **Submit** medRxiv → later DALI

## 8. Targets (refine after recon)
- Abstract: 200–250 words
- Methods/Results: the bulk (~60% of a MICCAI 8-page budget; medRxiv more flexible)
- Intro ~1 page, Discussion ~1 page, Related Work ~0.5–1 page
- Figures: 3 (ROC, Grad-CAM, UMAP) + 1 pipeline/teaser; Tables: 2

## 9. Decisions to lock with the author (before/early drafting)
- [ ] Final title
- [ ] Authorship list (solo preprint? advisor/ISI co-author? — honesty: only list contributors)
- [ ] Confirm "Qatar = color-neutralized 0.838" is the reported number (already decided, reconfirm)
- [ ] Whether to include the Qatar color-artifact audit as a Methods/limitation subsection

## 10. Anti-goals (what the paper will NOT claim)
- NOT "the model learned TB pathology"
- NOT "foundation models are always better" (specialist wins NIH/Shenzhen)
- NOT quantitative UMAP-cluster claims
- NOT a mechanistic proof — a well-supported, honest hypothesis
