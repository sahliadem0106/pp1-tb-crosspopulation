# PP1 dataset provenance check (step 0)

Checked 2026-10-08 against primary sources. This is the check the VSRP-chain plan (STEP 0) requires before PP1 gets a DOI.

## Verdict

**The "Qatar = MENA population" claim does not survive contact with the source.** The dataset we call Qatar is a **multi-source collection curated by Qatar University**, assembled from four public collections on three continents. Only the curators are Qatari; the patients are not. The MENA framing in the title, subtitle, abstract, intro, contributions and related work has to go, and the Qatar numbers have to be re-examined for overlap with the other two external sets and with the NIH training set.

## What the authors say (primary evidence)

Source: the dataset's own IEEE DataPort record, authored by the same team as the paper — Tawsifur Rahman, Amith Khandakar, Muhammad E. H. Chowdhury, "Tuberculosis (TB) Chest X-ray Database", DOI 10.21227/mps8-kb56, created 2020-10-20, last updated 2022-05-18. From its *Contribution* section, verbatim:

> "This dataset contains CXR images of Normal (3500) and patients with TB (3500). The TB database is collected from the source:
> **NLM dataset**: National Library of Medicine (NLM) in the U.S. has made two lung X-ray datasets publicly available: the Montgomery and Shenzhen datasets.
> **Belarus dataset**: Belarus Set was collected for a drug resistance study initiated by the National Institute of Allergy and Infectious Diseases, Ministry of Health, Republic of Belarus.
> **NIAID TB dataset**: NIAID TB portal program dataset, which contains about 3000 TB positive CXR images from about 3087 cases.
> **RSNA CXR dataset**: RSNA pneumonia detection challenge dataset, which is comprised of about 30,000 chest X-ray images, where 10,000 images are normal and others are abnormal and lung opacity images."

Supporting: Rahman et al., "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization", IEEE Access 8:191586-191601, 2020 (preprint arXiv 2007.14895), which is the paper the dataset record credits.

Composition of the version we actually used (the Kaggle mirror `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`, 4,200 images): 700 public TB images + 3,500 normal images; the record itself says a further ~2,800 TB images are available only under an NIAID data agreement. Confirmed against our own `results/qatar_labels.csv`: 700 `Tuberculosis/*`, 3,500 `Normal/*`.

## Why this matters for PP1

1. **Population claim falls.** Montgomery is USA, Shenzhen is China, Belarus is Belarus, the NIAID portal is multi-country, RSNA is USA. "Target MENA population" is not defensible.
2. **The external test can overlap the training set.** RSNA Pneumonia is a ~30,000-image subset of NIH ChestX-ray14, and NIH ChestX-ray14 is what we trained on. Any RSNA-derived images inside the 3,500 "Normal" images are images from the training distribution, not an external population.
3. **The external test sets can overlap each other.** Montgomery and Shenzhen images are themselves listed as sources of this collection. If any of them are inside the 4,200, our three "independent" external sets are not independent.
4. **The two classes are not one population either.** TB images and normal images come from different source mixes, so the near-chance AUC on this set may be riding on source composition as much as on decision-process shortcut.

None of this invalidates the headline finding (specialist collapses, zero-shot BiomedCLIP succeeds on the same images, mechanism is decision-side). It changes *what the set is called* and forces a dedup before the numbers are reported.

## Wording decision (pick one, do it everywhere)

Option A (recommended): rename the third external set throughout to **"a multi-source TB collection curated by Qatar University (Rahman et al., 2020)"**, drop every MENA/population claim, keep the finding as "collapse on a curated multi-source collection". Then dedup and report how many images were removed.

Option B: keep a regional angle only if the per-image source metadata shows a coherent regional subset large enough to stand alone. Not currently supported by anything we have; do not use it without the metadata.

Either way: title and subtitle change. Current manuscript title (`paper/manuscript_claude.md` L1-3) is "Cross-Population Generalization of Proxy-Trained Chest X-Ray Models for Tuberculosis" with the subtitle "A specialist fails on the target MENA population where a zero-shot foundation model succeeds". The subtitle is the part that has to move; the title itself survives if the set is relabelled.

## Remaining work before the DOI

1. Download the 4,200 images and the metadata file that ships with them (`metadata.xlsx` / `metadata.csv`), then check whether it carries a per-image source label. If it does, the provenance table gets real per-source counts and possibly a clean regional subset.
2. Perceptual hash every image (pHash, 256 px grayscale) across NIH train, Montgomery, Shenzhen and the collection. Count near-duplicates per pair of sets at a fixed threshold, and report the threshold.
3. Remove duplicates from the external test and re-run the Qatar cell of Table 1. Report how many were removed and what the AUC becomes.
4. Rewrite the Limitations paragraph (draft below) and the abstract/intro/contributions claims.
5. Only then publish the DOI.

## Draft Limitations paragraph

> The third external collection is not a single-population cohort. It is a curated composite of four public sources (Montgomery and Shenzhen via the US National Library of Medicine, a Belarus drug-resistance collection, the NIAID TB portal, and the RSNA pneumonia challenge), re-released by Rahman et al. (2020) as 3,500 TB and 3,500 normal images, of which the 700 TB and 3,500 normal images used here are the publicly available subset. We therefore do not describe it as a MENA population cohort and do not make regional claims from it. Because two of its sources are also used as external test sets here, and its RSNA component derives from the same underlying images as our training corpus, we deduplicated the images by perceptual hash and report all three external results after removing overlaps (n removed: X). The per-source composition of the public subset is not published by the dataset authors and we could not verify it.

## Gaps (unverified, do not assert)

- Per-source counts inside the 4,200-image public subset.
- Whether the shipped metadata file labels each image's source.
- Whether the 3,500 normals contain RSNA images at all, and how many (implied by the source list, not stated per class).
- Whether our Montgomery/Shenzhen copies are byte-identical or merely near-identical to images inside the collection; only the hash pass settles it.
