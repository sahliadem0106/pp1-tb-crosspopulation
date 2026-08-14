# Abstract

> VERSION 2 STRUCTURE (Sida-Peng): Challenge → Insight → Contribution → results.
> LITERALLY LAST to finalize — numbers below are from the final runs and are
> correct as logged. Length target: 200–250 words. Marked [X] for items to
> finalize (venue/word count).

**Title:** Cross-population generalization of proxy-trained chest X-ray models for tuberculosis: a specialist fails on the target MENA population where a zero-shot foundation model succeeds

**Abstract**

Tuberculosis remains a leading cause of death, and automated chest radiography
screening is a promising tool for high-burden and underserved regions, including
the Middle East and North Africa (MENA). Whether such models generalize across
populations is largely untested. We trained a DenseNet-121 on NIH ChestX-ray14
using the Infiltration finding as a proxy for tuberculosis and evaluated the
frozen model on three external datasets: Montgomery (USA), Shenzhen (China), and
a MENA dataset from Qatar. The model transferred unevenly — matching or beating
its home score on Shenzhen (AUC 0.769 vs 0.715) but collapsing to near chance on
Qatar (AUC 0.571, confidence interval disjoint from all others) — and at the
WHO-recommended sensitivity of 0.90, specificity was unusably low on every
dataset (0.11–0.38).

We probed the mechanism of this failure. Grad-CAM revealed the specialist's
decisions rest on a stereotyped right-upper-lung spatial prior whose attention
collapses to image background on errors, and UMAP showed the target population is
*not* out-of-distribution in feature space — indicating a decision-process
artifact rather than distribution shift. Consistent with this, a zero-shot
foundation model (BiomedCLIP) succeeded on the same Qatar images (AUC 0.838,
artifact-controlled) with +0.267 over the fine-tuned specialist. We conclude that
cross-population failure in proxy-trained TB models is driven by shortcut-like
decision processes, and that shortcut-robust foundation-model representations
offer a promising path toward population-invariant TB screening.
