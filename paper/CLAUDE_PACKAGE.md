# Claude Package — PP1 Paper (give this stack to Claude)

> This file tells you WHAT to give Claude, and WHAT Claude should do with it.
> The task: reshape the assembled draft into a final, publication-ready manuscript.

---

## 1. The stack (hand all of these to Claude)

**Primary:**
- `main_draft.md` — the full assembled paper (all sections, paper order)

**Figures (attach):**
- `../results/roc_overlay.png` — Figure 1, ROC overlay (NIH + 3 external)
- `../results/gradcam_montgomery_shenzhen.png` — Figure 2, Grad-CAM rows (TPs/FNs)
- `../results/umap_comparison.png` — Figure 3, two-panel UMAP
- *(also in `../results/`: roc_nih.png, table1_external_eval.csv, table2_zero_shot.csv)*

**Grounding notes (Claude can read for depth):**
- `gradcam_notes_claude.md` — the Grad-CAM analysis + raw-magnitude verification
- `umap_notes_claude.md` — the UMAP analysis + the mechanistic synthesis

**Methodology to follow:**
- Sida-Peng section method (`references/` of the `sida-peng-paper-writing` skill):
  one paragraph = one message; claim→evidence mapping; reverse outlining; five-dimension self-review.
- Venue: medRxiv preprint first → MICCAI DALI workshop (LaTeX template).

---

## 2. The task Claude should do

1. **Preserve the science exactly** — do NOT change any number, CI, or claim.
2. **Reshape structure & flow** — tighten paragraph logic, ensure one message per
   paragraph with a clear first sentence, improve cross-section coherence.
3. **Elevate the voice** — publication-grade academic prose; consistent terminology
   (e.g. pick one term for the mechanism and use it everywhere).
4. **Fill flagged gaps** — replace `[CITE]` markers with real citations where the
   draft already names the works; flag any WHO statistic in the Introduction that
   must be verified before submission.
5. **Compress** — toward the DALI page budget if over-long (the Methods/Results are
   the bulk; tighten Intro/Related Work).
6. **Produce** — a full polished manuscript (and, if asked, a section-by-section diff).

## 3. HONESTY RULES (hard constraints — do NOT break these)

- **Qatar is reported at 0.838 (color-neutralized)**, with the RGB 0.849 as a
  Methods note. Never report 0.849 as the headline Qatar number.
- The mechanism is framed as **"consistent with a decision-process shortcut"** —
  NEVER "we proved the model cheats" or "learned TB pathology."
- Keep "**no single model wins everywhere**" — the specialist retains NIH/Shenzhen.
- Keep all limitations: proxy labels, Montgomery n=138 (wide CI), Qatar artifact,
  no clinical validation, single-specialist-vs-single-generalist.
- No quantitative UMAP-cluster claims (UMAP is a visualization).
- Every number must match `research_log.md`. If a number looks wrong, FLAG it —
  do not silently "fix" it.

## 4. Output format from Claude

- A polished **full manuscript** (Abstract 200–250 words, all sections, figure
  captions, table captions).
- A short **summary of what changed** (structure, voice, any flagged risks).
- A **claim→evidence checklist** confirming every claim is supported.

## 5. After Claude's pass

1. **You (author) edit** the reshaped draft — every edit is your thinking.
2. Bring it back to Hermes → we reconcile against `research_log.md`, fill real
   citations (Zotero), fix the WHO stats, and convert to `.tex` (DALI template).
