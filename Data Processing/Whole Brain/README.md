# Whole-Brain Extraction

Preprocessing for the whole-brain baseline — used to test whether ROI-targeted input superior to whole-brain input, with token count held equal.

## What it does

Full MRI/PET volume with no cropping/ROI selection, resized to match the ROI pipeline's token count (3,072), normalised, and cached as `.npy`. Same rotation/flip augmentation and seeds (`[1, 101, 42]`) as ROI models, for consistency.

## Outputs

- `preprocessed_cache_wholebrain_mri(_aug)` — whole-brain MRI
- `preprocessed_cache_wholebrain_pet(_aug)` — whole-brain PET
