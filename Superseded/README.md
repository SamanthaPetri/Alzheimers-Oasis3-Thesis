> # SUPERSEDED
>
> **The results in this document were produced on data that has since been
> found to contain preprocessing defects. Do not cite these numbers.**
>
> Current results: see the main `README.md`.
>
> Four issues were identified and corrected after this version:
>
> **1. PET temporal averaging.** The full PIB acquisition was averaged into a
> single volume, including the early perfusion-weighted frames that track
> blood flow rather than amyloid binding. Corrected to the final 9 frames —
> the standard binding window, matching Vo et al.'s preprocessing.
>
> **2. Whole-brain masking.** The whole-brain MRI cache was never brain-masked:
> 66% of each volume was non-zero, meaning skull, scalp and neck were included
> and the patch-embedding occupancy mask marked nearly every patch as valid.
> The whole-brain PET cache was unmasked in one version and over-masked with
> the DKT segmentation (discarding ~19% of brain volume, mostly cortical
> ribbon) in another. Both now use FastSurfer's `mask.mgz`, giving a matched
> ~7% brain fraction across modalities. **All whole-brain results in this
> document are therefore invalid.**
>
> **3. Augmentation interpolation.** `scipy.ndimage.rotate` was called at its
> default `order=3` (cubic spline). Splines overshoot at sharp edges, and this
> data has background at exactly 0 against tissue at ±3 after z-scoring. The
> ringing raised the non-zero fraction from 0.13 to 0.84 for ROIs and from
> 0.07 to 0.39 for whole-brain volumes, so most patches passed the occupancy
> threshold on interpolation artefacts rather than anatomy.
>
> **4. Cohort quality control.** Nine subjects had failed hippocampal
> segmentation in FastSurfer's DKT parcellation — contributing blank tokens at
> ROI indices 0 and 1 — and five of those also had PET regions where z-scoring
> was skipped, leaving raw intensities two orders of magnitude above the rest
> of the cohort. A tenth subject (OAS30065) has a corrupt single-volume source
> PET. All ten are excluded from the current cohort, which is 200 subjects
> (101 converters / 99 stable).
