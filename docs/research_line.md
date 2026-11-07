# The research line: Grounding Under Shift

Written 2026-11-07. This is the program the individual projects belong to, stated as a research angle rather than an application plan.

## The thesis, in one sentence

Medical vision models fail when the population changes not because the new images are unfamiliar but because their decision process was never grounded in the disease, and this can be measured, compared across model generations, and repaired deliberately.

Every number in the portfolio is evidence for or against that sentence.

## Why this is a program and not five papers

Three things hold it together, and they have to stay fixed or the series falls apart into unrelated studies.

1. **One instrument.** A box-referenced pointing game, run with an always-guess-the-prior control. The control is what makes it a measurement: an 18% score from a model that never sees the image is the floor, and a model that scores near the floor has demonstrated nothing about the disease. Call it the prior-controlled pointing game. Fix the protocol in one document, version it, and reuse it unchanged in every paper. A named, reused instrument is what makes a body of work citable as a body of work.
2. **One axis of variation at a time.** Population (PP1, D), model generation (PP1 to A), and supervision regime (proxy label, zero-shot text, few-shot) are varied separately, never all at once. A result that varies two things at once is not a result.
3. **One standard of evidence.** Paired tests on shared images, bootstrap intervals on small n, hedged mechanism claims, dataset provenance checked before any population claim. This is the part that survives peer review, and it is the part most student work skips.

## The four claims the series makes, in order

| # | Claim | Project | Status |
|---|---|---|---|
| 1 | A proxy-trained CNN can hold up on two external populations and collapse on a third, and the collapse is decision-side, not distribution shift | PP1 | **executed**, no DOI |
| 2 | The decision-side failure is measurable against radiologist boxes, and it survives an always-prior control | PP2 / PP5 | design locked, not run |
| 3 | A medical vision-language model that generates text has the same grounding problem, and its stated boxes can be checked against radiologist boxes | A | not started |
| 4 | Grounding, not accuracy, is what adaptation should be selected on: few-shot and continual approaches can raise accuracy while eroding localization | C, D | not started |

Claim 1 is the strongest thing in the portfolio because it is measured and it contains a negative result plus a positive control (BiomedCLIP). Claim 4 is the one that would make the program matter outside its own papers, because it turns the measurement into a selection rule.

## The series

Ordered by dependency, not by ambition.

| Step | Project | Question | Artifact it produces | Status |
|---|---|---|---|---|
| 0 | PP1 fixes and DOI | is the population claim defensible; does the protocol description match the run | preprint with a DOI, provenance table, ablation | provenance check done, wording not applied, no DOI |
| 1 | PP2 / PP5 | does attention follow the lesion, measured against boxes with a prior control | the instrument, two architectures, tables | recon only |
| 2 | A | does a text-generating medical model ground what it asserts | box-vs-box agreement, perturbation stability, seen/unseen leakage split | not started |
| 3 | C | can a new disease be learned from text alone, and what does continual zero-shot adaptation cost the old classes | pilot curves, a predictability rule from text similarity | not started |
| 4 | D | when does a conformal guarantee silently break on a new population, and can it be repaired without labels | breakage detector, weighted-conformal fix, on PP1's stored predictions | not started |
| banked | B | does token pruning preserve localization or only diagnosis | diagnosis vs localization under pruning, by lesion size | not started |
| banked | E | is there a time-blindness failure in X-ray models | controlled synthetic change probe | not started, partly scooped |

The chain's PP2 is the same project as the old PP5. B and E are banked deliberately: each is interesting, neither is on the critical path, and starting one now costs the series its only scarce resource, which is a finished paper with a DOI.

## What makes this an angle and not a niche

- It is a **measurement** program. Contributions are protocols, controls and honest negative results, not state of the art. That is a defensible position in a field saturated with benchmark chasing, and it is the position the strongest reviewers respect most.
- It crosses **model generations** on the same instrument, which almost nobody does: a proxy-supervised CNN, a zero-shot vision-language model, and an adapted few-shot model, measured with the same ruler on the same images.
- It has a **failure-first** structure. The paper that opens the program reports where a model breaks, and the program's value is what that break implies for everything downstream. Programs that start from a success claim have to defend the claim; this one has to explain a failure, which is easier to do honestly.
- It has a **geographic stake that is not a marketing claim**. Gulf and MENA health systems are deploying screening tools built elsewhere. After the provenance check, that is a motivation in the introduction and not a claim about a dataset.

## Boundaries

Stated so the work does not drift.

- Not an architecture program. No new backbones, no pretraining runs.
- Not a deployment program. No clinical claims, no prospective data, no regulatory framing.
- Not a benchmark. The instrument measures one thing, on data that already exists.
- Not a thesis about MENA. The MENA claim died in the provenance check and stays dead until there is per-image source metadata.

## The sentence a professor should be able to repeat

> He measures whether medical vision models actually look at the disease, with a controlled instrument, across model generations and across populations, and he uses that measurement to decide what adaptation is safe.

## Near-term order, given the current state

1. Close PP1: apply the provenance wording and the four protocol corrections, run the pHash dedup, get the DOI. Nothing else in the series has value until the anchor paper is citable and true.
2. Build the instrument in PP2 and freeze its protocol document. This is the reusable asset.
3. Borrow fifteen hundred pounds of compute for A, which is the most expensive step, only after 1 and 2.
4. C and D overlap with writing, not with each other.

## Open questions the program leaves unanswered, on purpose

- Does the prior control need to be lesion-specific rather than lobe-level. Deciding that is a PP2 result, not a design choice.
- Can the instrument run without radiologist boxes, on data where only report text exists. That is the version that matters for the Gulf, and it is a plausible third or fourth paper.
- Does adaptation that preserves grounding also preserve calibration. That bridges to D and to the uncertainty work.
