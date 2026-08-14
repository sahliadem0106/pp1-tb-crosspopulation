# Discussion

> Drafted from the mechanistic synthesis (Phase 5 + 7 + 6). Edit freely.

## 4.1 The failure is a decision-process artifact, not distribution shift

The central question we set out to answer was whether a proxy-trained TB model
generalizes across populations. The answer is more subtle than a simple
"yes" or "no", and the evidence points to a specific mechanism.

Our UMAP analysis shows that **Qatar is not out-of-distribution in feature
space**: its embeddings overlap extensively with NIH in both the DenseNet and
BiomedCLIP representation spaces. If the Qatar failure were driven by the data
being foreign, we would expect separation; we observe overlap. Yet the specialist
still fails (AUC 0.571), while a zero-shot foundation model succeeds on the same
images (AUC 0.838). The signal is present in the data; the specialist's decision
process cannot exploit it.

Grad-CAM explains why. The specialist's true-positive decisions rest on a
**stereotyped right-upper-lung spatial prior** — a near-identical activation
blob across patients, plausibly echoing the epidemiological predominance of
right-upper-lobe TB, but not lesion-specific. When that prior is not satisfied,
attention **collapses to non-anatomical regions** (image borders), rather than
searching elsewhere in the chest. This is the signature of a shortcut-dependent
classifier: it has learned *where* to look, not *what* to look for. A model whose
decisions depend on such a prior is fragile to exactly the acquisition and
presentation differences that characterize a new population.

Taken together, these results support the interpretation that **the observed
cross-population failure is a decision-process artifact** — a consequence of
shortcut-like spatial priors learned from proxy-labeled training data — rather
than a distribution-shift property of the target data. We are deliberately
careful to frame this as a well-supported hypothesis, not a mechanistic proof:
Grad-CAM shows where the model looks, and the representation and zero-shot
results are consistent with a decision-side explanation.

## 4.2 Why the foundation model succeeds

BiomedCLIP, a 400M-parameter model pretrained on ~15M biomedical image-text
pairs, achieved a higher AUC on Qatar (0.838, color-neutralized) with zero
training than the fine-tuned specialist (0.571) — a +0.267 gain that held after
controlling for the known color artifact. This is consistent with pretraining
*representational diversity*: features learned across many imaging contexts and
modalities are more population-invariant and less entangled with task-specific
shortcuts than features fine-tuned on a single proxy-labeled dataset.

We note that no single model wins everywhere: the specialist retained the edge on
its home distribution (NIH) and on Shenzhen, while the generalist dominated
Montgomery and Qatar. This "no single model wins" pattern argues against a naive
"foundation models are always better" reading and in favor of a more useful
conclusion: **for cross-population screening, representation diversity and
shortcut-robustness matter more than proxy-label fine-tuning.**

## 4.3 Clinical implications

At the WHO-recommended sensitivity of 0.90, specificity collapsed on every
dataset (0.11–0.38), including the home set. Operationally, this means the
specialist — and, by extension, models of this class — cannot serve as a
standalone TB screener without overwhelming referral volumes, even on the
population it was trained on. Screening deployment therefore requires either
substantially more discriminative models, explicit operating-point selection with
human-in-the-loop triage, or a recalibration strategy on target-population data.

## 4.4 Limitations

We are explicit about the limitations of this study:
- **Proxy labels.** NIH provides no TB labels; the Infiltration proxy is
  correlated with, but not identical to, true TB. Positive-transfer to real TB
  datasets (Montgomery/Shenzhen) partially validates the proxy, but the training
  signal is not a confirmed diagnosis.
- **Small external sets.** Montgomery (n = 138) yields wide confidence intervals;
  its AUC is consistent with NIH but not distinguishable from it.
- **Qatar acquisition artifact.** The Qatar dataset contains a colorized subset
  in its TB class. We evaluate it with color cues removed; the color-neutralized
  number (0.838) is reported for the foundation model and the specialist pipeline
  is grayscale by construction. The artifact limits the strength of Qatar-only
  conclusions and motivates the follow-up subgroup analysis.
- **Single specialist vs single generalist.** We compare one fine-tuned DenseNet
  with one foundation model; both are single instances. The mechanism hypothesis
  (shortcut-mediated failure) should be tested across architectures.
- **No clinical validation.** These are offline AUC/operating-point analyses, not
  prospective screening evaluations.

## 4.5 Future work

Our findings motivate a direct causal test of the shortcut hypothesis: measuring
whether shortcut reliance predicts adaptation difficulty, and whether
shortcut-robust (or foundation-model) representations eliminate the
cross-population gap. We also plan to examine the minority-Qatar subgroup that
forms an isolated cluster in BiomedCLIP space, and to validate on additional
MENA populations.
