# The protocol discrepancy, and what has to change in the manuscript

Status: **open**. Drafted 2026-09, not applied to either manuscript file as of 2026-11-07.

## What happened

The Methods section describes the filtered CheXNet protocol: positives are images whose finding vector contains Infiltration, negatives are images labelled No Finding, everything else excluded. That gives 19,870 positives and 60,412 negatives, ratio about 1:3.0.

The run that produced the paper's headline numbers used a wider negative pool: other-finding images were still in the negatives. That pool is about 19,894 positives and 92,226 negatives, ratio 1:4.6.

So the paper's Methods describe a protocol the paper's model was not trained under. It is a text-versus-run mismatch, not a fabricated result. The AUCs are what they are; the description of how the data was assembled is wrong.

## What was done about it

Rather than fix the prose and leave it, both protocols were trained and compared on the images they share, so the size of the effect is a measured number instead of a sentence in Limitations.

| Model (evaluated on) | n | AUC | 95% CI | Specificity @ 90% sens |
|---|---|---|---|---|
| clean model, clean test | 12,249 | 0.7710 | [0.7618, 0.7805] | 0.393 |
| mixed model, clean test | 12,249 | 0.7528 | [0.7448, 0.7625] | 0.374 |
| mixed model, full test (the paper's number) | 17,260 | 0.7153 | [0.7069, 0.7253] | 0.320 |
| clean model, full test | 17,260 | 0.7025 | [0.6937, 0.7128] | 0.311 |

Paired statistics on the 12,249 identical images:

- DeLong paired: delta +0.0182, Z = 5.20, p < 0.0001, CI [+0.0113, +0.0250]
- Paired bootstrap, 2000 resamples: delta +0.0183, CI [+0.0114, +0.0253], P(delta > 0) = 1.000
- Specificity at fixed 90% sensitivity: 0.394 vs 0.375, +1.9 pp, CI [+0.003, +0.043]
- McNemar at 0.5: 1120 vs 1051 discordant, p = 0.144, not significant
- In clinical terms: 174 fewer false positives on 9,233 negatives

Pipeline identity was verified before any of this was believed: the retrained pipeline reproduced the paper's home AUC to four decimals (0.7153 expected, 0.7153 obtained).

Evidence file: `results/paired_clean_vs_mixed_test.csv`, one row per shared image, with the true label and both models' scores.

## The four corrections still owed to the paper

Drafted for Methods 2.1, Methods 2.3, Results 3.1 and Limitations 4.4. Each must state the run it describes, not the protocol that was intended.

1. **Methods 2.1, dataset assembly.** Name the negative pool as used: positives 19,894, negatives 92,226, ratio 1:4.6, other-finding images retained in the negatives. State that this differs from the CheXNet filtered protocol and cite the clean-pool count alongside it.
2. **Methods 2.3, training and imbalance handling.** The positive weight that was actually applied corresponds to the pool that was actually built. State the value and the pool it came from, so a reader can reproduce the loss.
3. **Results 3.1, home evaluation.** Report the paper's headline number on the full test set, then the ablation as a separate paragraph: same images, both protocols, paired test, p < 0.0001. Do not present the two as an improvement claim.
4. **Limitations 4.4.** Say plainly that the original run's negative pool was wider than the described protocol, that the effect was measured rather than assumed, and that the external numbers in Table 1 come from the mixed-pool model.

## Wording rules, non-negotiable

- This is a **protocol effect**, never an improvement. Each model wins under its own protocol: clean beats mixed on clean data (0.771 vs 0.753), mixed beats clean on mixed data (0.715 vs 0.703).
- The two 95% confidence intervals overlap by 0.0007. An overlap read says "no difference". The paired test on the same images says p < 0.0001. Same-image comparisons use paired tests, and this trap gets a sentence in the paper because a reviewer will fall into it.
- No deployability claim anywhere.

## Still missing

The ablation notebook is not on disk. Only the joined CSV survived, which is enough to reproduce the statistics but not the training. If the numbers are ever questioned, retraining is the only answer, and that means either finding the Kaggle version or rewriting the cell. This is the single largest reproducibility hole in the project.
