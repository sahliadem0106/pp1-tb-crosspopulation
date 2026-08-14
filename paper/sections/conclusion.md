# Conclusion

> Drafted. Edit freely.

We evaluated a proxy-trained tuberculosis model across three external
populations and found that its failure on the target MENA cohort is a
**decision-process artifact, not a distribution-shift property of the data**.
Grad-CAM revealed a stereotyped, shortcut-like spatial prior whose attention
collapses to background on errors; UMAP showed the target data overlaps the
training distribution; and a zero-shot foundation model, free of that prior,
succeeded on the same images (+0.267 AUC). At WHO-required sensitivity, no
dataset — including the home set — was clinically usable as a standalone
screener.

These findings carry a clear implication for AI-based TB screening: reported
in-distribution accuracy does not guarantee cross-population safety, and
foundation-model representations that are less entangled with proxy-label
shortcuts offer a promising path toward population-invariant deployment. We
release our evaluation protocol and code to support reproducible
cross-population assessment of TB screening models.
