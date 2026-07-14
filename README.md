# Alzheimers-Oasis3-Thesis

OASIS-3 Preprocessing Pipeline — ROI-Based Multimodal Mamba (GENG4411/4412)

Preprocessing pipeline for a 210-subject OASIS-3 cohort, extracting anatomically-defined ROIs from MRI and PET for a ROI-based multimodal Mamba model predicting CN→MCI/AD conversion. Run scripts/notebooks in the order below.

Pipeline

## 1. thesis_cohort_final.csv

Cohort definition for all 210 subjects: subject_id, baseline_day, conversion_day, pet_day, mri_day, mri_session, outcome_label (1 = converter to MCI/AD within 10 years, 0 = stable CN). Every other script in this pipeline reads from this file.

## 2. check_pet_days.py

Verifies each subject has a PIB PET scan within 30 days of their target pet_day, matching against the raw PET scan directory. Reports exact matches, close matches (≤30 days), and subjects with no usable PET scan. Verification only — produces no output files.

## 3. convert_all_mgz.py

Converts FastSurfer's orig.mgz output to orig.nii.gz for each subject. Required before PET registration, since FLIRT (step 4) needs a NIfTI reference volume, not MGZ.

## 4. register_all_pet_v2.sh

Averages each subject's PET frames (fslmaths -Tmean) then rigidly registers the averaged PET volume onto the subject's MRI space (flirt, 6 degrees of freedom) using FastSurfer's orig.nii.gz as the reference. Outputs {subject_id}_PIB_in_MRI.nii.gz plus the transform matrix.

## 5. MRI_Extraction.ipynb

Extracts 6 anatomically-defined ROIs (bilateral hippocampus, cerebellum WM, cerebral WM) from FastSurfer-segmented MRI. Pipeline: mask ROI from segmentation → tight crop (+3 voxel padding) → z-score normalise (ROI voxels only) → resize to 64×64×64 → cache as .npy. Also generates 3 augmented copies (random rotation ±15°, random flips) per training-subject only, using seeds [1, 101, 42].

## 6. extract_pet_rois.py

Extracts the same 6 ROIs from registered PET volumes, reusing the MRI-derived segmentation masks (PET has no reliable anatomical detail of its own to segment on). Same crop → normalise → resize-to-64³ pipeline as MRI, so PET and MRI caches stay dimensionally consistent for later fusion.

## 7. PET_Augmentation.ipynb

Generates matching augmented copies for PET, mirroring the MRI augmentation exactly — same rotation range, same seeds, same training-only subject split — so both modalities have equal 4× training-set sizes for multimodal fusion. MRI and PET are augmented independently (not with a shared per-subject transform), which is fine since the multimodal model uses two separate encoder branches (late fusion).

## 8. PET_ROI_Images.ipynb (QA/visualisation)

Visual sanity checks: plots individual PET ROIs to confirm anatomically sensible extraction, checks cache completeness/shape consistency across all 210 subjects, and includes a minimal end-to-end multimodal dataloader test.
