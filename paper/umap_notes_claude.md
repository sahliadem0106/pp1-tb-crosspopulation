# UMAP analysis (Claude's read, 2026-08-13) — DenseNet vs BiomedCLIP representation space

> Raw material for Figure 3 caption + Discussion. Verdict: the figure KILLED the
> "Qatar island" hypothesis — and that's the right outcome (honesty first).

## Per-panel (Claude)
- **DenseNet features:** one continuous connected manifold, no hard boundaries. NIH has a
  blue-dominant tendril (lower-center) where few others appear; elsewhere heavy overlap.
  Qatar = most diffuse of the four, present at high density across the entire blob. Shenzhen
  modest right-arm concentration. Montgomery (n=138) = a couple of tight sub-clusters, too
  sparse to call "separated" vs under-sampled.
- **BiomedCLIP features:** more structured, but NOT in the hypothesis's direction. NIH and
  Qatar occupy the same two large regions, extensively intermixed — no clean boundary. What
  separates cleanly: Montgomery (tight isolated green crescent, center) and Shenzhen
  (elongated orange arc, right/bottom). Small isolated red Qatar island far right — but a
  MINORITY of Qatar points; bulk stays mixed into the NIH mass.

## Cross-panel (Claude)
- NO clean "DenseNet separates Qatar / BiomedCLIP unifies Qatar" story. Qatar/NIH overlap
  heavily in BOTH panels → the 0.571 failure is NOT gross representational separation →
  decision-boundary/feature-weighting problem, not OOD.
- What changes most: M/S go from loosely-overlapping (DenseNet) to well-separated
  (BiomedCLIP) — and separation does NOT track performance (Montgomery isolated + AUC gain
  0.739→0.850; Shenzhen isolated + AUC drop 0.769→0.650).
- Within-population spread: Montgomery tightens into a compact cluster (BiomedCLIP);
  Shenzhen goes diffuse → coherent arc; Qatar stays broad + overlapping with NIH in both.

## Can claim / cannot claim (Claude)
CAN: no well-separated clusters in DenseNet space (one manifold, regional density diffs);
Qatar shows no visible separation from NIH; M/S form distinct sub-regions in BiomedCLIP
space; a minority Qatar subset forms an island in BiomedCLIP space (subgroup follow-up,
not "Qatar" generally); representation separation does not track consistently with AUC.
CANNOT: "specialist's space separates Qatar explaining failure" (figure shows the
opposite); "foundation model unifies populations" (only true for Qatar/NIH, and that
overlap already existed); quantitative cluster-distance claims; statistical significance
from a picture; causality (figure is descriptive — motivates feature-level analysis, i.e. PP5).

## Suggested caption (Claude, adopted)
"UMAP projections of 1024-dimensional image embeddings from the trained DenseNet121 (left)
and zero-shot BiomedCLIP (right), colored by source dataset. Qatar overlaps extensively with
NIH in both representation spaces, while Montgomery and Shenzhen form more distinct
sub-clusters under BiomedCLIP; cluster overlap and separation are descriptive of local
embedding structure only and are not evaluated quantitatively."

## Hermes synthesis — the paper's mechanistic story (ties all phases together)
1. Data-side: Qatar images are NOT foreign in feature space (overlap NIH, both models).
2. Decision-side: the specialist fails on them anyway (0.571) — Grad-CAM shows a
   stereotyped RUL spatial prior that collapses to background on errors.
3. Proof it's the process, not the data: zero-shot BiomedCLIP succeeds on the SAME
   overlapping images (0.838 color-neutralized) — the signal exists; the specialist's
   shortcut weights can't use it.
4. M/S separation-doesn't-track-performance → representation geometry alone doesn't
   explain transfer; decision-process quality does.
=> "Cross-population failure is a shortcut artifact of the decision process, not
   distribution shift." Sets up PP5 (shortcut detection, causal) as the natural follow-up.

## Open footnote idea
The minority-Qatar island in BiomedCLIP space — check whether it corresponds to the
colored subset or a specific acquisition subgroup (cheap to test: color-neutralized
embeddings vs RGB).
