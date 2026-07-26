# Whole-Brain Extraction

Preprocessing for the whole-brain baseline — used to test whether ROI-targeted input is superior to whole-brain input, in two variants: token-matched (compute held equal) and native resolution (no downsampling at all).

## Token-Matched Variant

Full MRI/PET volume, no cropping/ROI selection, resized to match the ROI pipeline's token count (3,072), normalised, and cached as `.npy`. Same rotation/flip augmentation and seeds (`[1, 101, 42]`) as ROI models, for consistency.

**Outputs**
- `preprocessed_cache_wholebrain_mri(_aug)` — whole-brain MRI
- `preprocessed_cache_wholebrain_pet(_aug)` — whole-brain PET

## Native Resolution Variant

Same MRI/PET volumes, no cropping/ROI selection, **no resize applied at all** — used at their true native resolution (256³ for MRI, FastSurfer-conformed; PET at its registered resolution). Uses the identical augmentation function (rotation ±15°, random flips) and the same seeds (`[1, 101, 42]`).

Token count at native resolution: 32,768 (vs. 3,072 token-matched)

**Outputs**
- `preprocessed_cache_wholebrain_native_mri(_aug)` — whole-brain MRI, native resolution
- `preprocessed_cache_wholebrain_native_pet(_aug)` — whole-brain PET, native resolution
