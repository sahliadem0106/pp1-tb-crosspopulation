# Grad-CAM analysis (Claude's read, 2026-08-13) — NIH model on Montgomery + Shenzhen

> Raw material for the paper's Grad-CAM section. Cross-checked by Hermes; see
> the RUL-epidemiology counterpoint and the raw-magnitude verification below.

## Montgomery — True Positives
Heat consolidates into a tight, high-intensity, roughly square red/yellow blob on the
right lung, mid-to-upper zone, in 8/10 images. Strikingly uniform: near-identical size,
shape, position across patients with presumably different lesions. Real TB findings vary
in size/shape/laterality per patient — a fixed "stamp" is more consistent with a coarse
right-lung spatial prior than pixel-level localization. Heat stays within lung parenchyma
(no corner markers / text / borders — crudest shortcuts absent). Two images (#1, #9)
deviate: heat pulled toward lower chest/diaphragm.

## Montgomery — False Negatives
CLEAREST finding: in ~7–8/10 missed cases, essentially no heat in the lungs. Hottest
region = horizontal red/orange band at the very bottom edge of the frame (background), and
in two cases (#2, #10) an isolated hotspot above the lung apex near shoulder/clavicle.
Only #3 shows focal heat inside the chest (off to the mediastinum side). Genuine
absence-of-lung-evidence, not misplaced-but-plausible attention: when the model doesn't
find its target it defaults to non-anatomical background/border pixels.
CAVEAT (Claude): per-image min-max normalization guarantees a "hot" pixel everywhere;
border hotspots may be rescaling artifacts. Verify against raw (un-normalized)
activation magnitudes before claiming border attention. → Hermes: verification code
provided; run it.

## Shenzhen — True Positives
Similar in kind, less rigid: heat predominantly right lung upper-to-mid in ~8/10, shape
more variable (compact squares, elongated strips; #3, #8 extend centrally into
mediastinal/perihilar region — less TB-specific, could reflect response to general
density/silhouette). No corner markers/text anywhere. Consistent right-lung bias across
~20 images total strengthens the "right lung gets weighted regardless" concern.

## Shenzhen — False Negatives
GENUINELY DIFFERENT from Montgomery FNs — two failure modes:
(a) ~half (#2–5): substantial focal right-lung heat resembling the TP signature, yet
below the 0.5 threshold → calibration / borderline-confidence failure, not "found
nothing".
(b) other half: weak/absent lung heat; #6 shows the same bottom-border artifact as
Montgomery.

## Verdict (Claude)
MIXED. CAN claim: Grad-CAM localizes to lung parenchyma (not markers/text) in most
correct cases in both datasets → rules out the most blatant burned-in-marker shortcut.
CANNOT claim: "learned TB pathology," that good external AUC is validated by anatomically
faithful attention, or lesion-specific localization. Stereotyped size/shape/position is
more consistent with a right-lung spatial prior. FN border-collapse is exactly the
behavior expected from a shortcut-dependent classifier, not genuine differential search.
Border hotspots must carry the min-max caveat.

## Hermes additions (domain knowledge Claude lacked)
1. **RUL counterpoint:** right upper lobe is the TEXTBOOK most-common location of
   post-primary TB. The stereotyped right-upper/mid heat may be a legitimate
   epidemiological prior (model learned "TB = RUL density"), not (only) a shortcut.
   Paper must present the ambiguity: consistent-with-TB-epidemiology BUT stereotyped +
   FN border-collapse keeps the shortcut hypothesis alive. Evidence cannot fully separate.
2. **Verification DONE (raw magnitudes, Montgomery FNs):** 0104: bottom row 0.0114 vs lungs -0.0002 → border attention REAL, raw-confirmed (Claude's artifact caveat defeated for this image). 0108: lungs 0.0368 (max), border 0.0085 → lung evidence present, under-confident. 0113: lungs 0.0138 (max), border 0.009 → same. ALL THREE show a persistent modest border response (8-11% of peak) — background behavior that takes over when lung evidence is weak. BOTH failure modes exist in Montgomery too (border-collapse AND under-confident-lung), not Shenzhen-only.
3. **Paper framing:** Grad-CAM section = "where the model looks" (parenchyma, no blatant
   marker shortcut) + "stereotyped right-lung prior consistent with RUL TB epidemiology
   but not lesion-specific" + "FN attention collapse to background" → candidate
   explanation for the Qatar failure (a laterality/spatial shortcut that fails to
   transfer), NOT evidence against it.
