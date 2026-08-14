# Related Work

> Skeleton drafted from genre knowledge. Citations are PLACEHOLDERS — fill via
> Zotero + the 2-3 exemplar skeletons during the recon step. Do not submit with
> [CITE] markers.

## AI for tuberculosis screening from chest radiographs
Deep learning has repeatedly been shown to detect chest X-ray abnormalities at
radiologist-competitive levels [CITE: CheXNet 2017; CheXpert; ChestX-ray14,
Wang 2017]. For tuberculosis specifically, several systems report high accuracy
on single-population cohorts and are promoted for screening in high-burden
settings [CITE: CAD4TB; multiple]. A recurrent methodological caveat is that
many of these systems are trained and evaluated on the *same* population, leaving
cross-population generalization — the property that matters for deployment —
largely untested [CITE].

## Cross-population generalization and distribution shift in medical imaging
Medical imaging models are known to degrade under distribution shift — differences
in scanner, acquisition protocol, population demographics, and disease
presentation between training and deployment cohorts [CITE: distribution shift
medical imaging]. A common finding is that models trained on one institution's
data underperform on others', with performance drops that are difficult to
predict a priori [CITE: multi-institution generalization]. We add a MENA-targeted
cross-population evaluation and probe the *mechanism* of failure rather than
reporting accuracy degradation alone.

## Shortcut learning and dataset bias
Deep classifiers frequently rely on spurious correlations — "shortcuts" — rather
than the medically meaningful signal: patient-position laterality markers, scan
overlays, or acquisition-specific color/texture biases [CITE: shortcut learning;
Zech-style "confounded" attention]. Such shortcuts inflate within-distribution
accuracy and silently break under shift [CITE]. Our Grad-CAM and representation
analyses ask whether the observed cross-population failure is a consequence of
shortcut-dependent decision processes rather than data being out-of-distribution.

## Foundation models and zero-shot transfer for medical imaging
Contrastive image-text foundation models pretrained on large biomedical corpora
(BiomedCLIP, ~15M image-text pairs) transfer to downstream tasks with little or
no task-specific training [CITE: BiomedCLIP 2023]. A key open question is whether
such generalist models generalize across populations better than task-specialists
fine-tuned on proxy-labeled data [CITE]. We provide a head-to-head zero-shot
comparison on the same external populations.
