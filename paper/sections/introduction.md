# Introduction

> Drafted — WHO numbers marked [CITE] and MUST be verified against WHO TB report
> before submission (honesty rule: never fabricate statistics).

Tuberculosis remains one of the world's deadliest infectious diseases. In [YEAR]
the WHO estimated approximately [10.6M] new cases and [1.25M] deaths, with the
largest burden in low- and middle-income regions [CITE: WHO Global TB Report].
The Middle East and North Africa (MENA) is increasingly recognized as a region
of concern, where case detection gaps and rising drug resistance make reliable
screening a priority [CITE]. Chest radiography is a first-line screening tool,
and deep learning has raised the prospect of automated, low-cost TB screening in
exactly these resource-constrained settings.

Yet a fundamental question remains largely unanswered: **do chest X-ray models
generalize across populations?** Most deep-learning TB systems are trained and
evaluated on a single population, frequently the one that produced the training
data. When such a model is deployed on a different population — different
scanner, different acquisition protocol, different disease presentation, and
different demographic makeup — its performance is not guaranteed to transfer.
For a screening tool aimed at populations unlike its training cohort, this
uncertainty is not a technical detail; it is a safety and deployment concern.

We investigate this question through a **controlled cross-population
evaluation** of a tuberculosis model. We train a single DenseNet-121 on the NIH
ChestX-ray14 dataset using the Infiltration finding as a proxy label for
tuberculosis, then evaluate the *frozen* model on three external datasets:
Montgomery (USA), Shenzhen (China), and a MENA dataset from Qatar. We further ask
*why* the model succeeds or fails, using Grad-CAM interpretability and
representation-space analysis, and we compare the specialist against a large
zero-shot foundation model (BiomedCLIP).

Our central finding is that the model's failure on the target MENA population is
**not a simple distribution-shift failure**: its decision process relies on a
coarse, shortcut-like spatial prior, and a zero-shot foundation model that lacks
that prior succeeds on the same images. These results suggest that
cross-population generalization for TB screening is governed less by whether the
data is "in distribution" and more by *how the model decides* — and that
foundation-model representations, free of proxy-label shortcuts, offer a
promising path toward deployable, population-invariant TB screening.

## Contributions
- A controlled cross-population evaluation of a proxy-trained TB model on three
  external populations, including a MENA cohort (Table 1), with statistical
  significance and WHO-operating-point analysis.
- An interpretability and representation analysis showing the failure is
  decision-process-mediated (stereotyped spatial prior; attention collapse on
  errors; no representation-space separation).
- A head-to-head zero-shot foundation-model comparison in which BiomedCLIP
  beats the fine-tuned specialist on the target population by +0.267 AUC,
  artifact-controlled.
