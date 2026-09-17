# OASIS-3 Data Preprocessing Pipeline

Preprocessing for a 200-subject OASIS-3 cohort, extracting anatomically-defined
ROIs and whole-brain volumes from MRI and PET for a multimodal Mamba model
predicting CN to MCI/AD conversion.

Subject selection, dataset conventions and evaluation setup follow Vo et
al. [1], the closest prior work predicting CN to MCI/AD conversion on OASIS-3
with MRI and PET, and the baseline this project benchmarks against.

The choice of anatomically-defined ROIs over whole-brain volumes follows
Khan et al. [2].

### Prerequisites

**FastSurfer** [3] — MRI segmentation and brain masking.
https://github.com/Deep-MI/FastSurfer

**FSL** [4] — provides `flirt`, `fslmaths`, `fslroi` and `bet`. Requires a
Linux/WSL environment. https://fsl.fmrib.ox.ac.uk/fsl/docs/

**SynthStrip** [5] via Docker — PET skull-stripping, MNA-net comparison branch
only. `docker pull freesurfer/synthstrip`

---

## Main pipeline — native space

Run in the order below.

### 1. `thesis_cohort_final.csv`

Created using same selection criteria as Vo et. al [1]. Scripts for this can be found [here.](https://github.com/JamieVo890/Multimodal-Attention-based-Neural-Networks-for-the-Prediction-of-Cognitive-Decline.git)

Cohort definition for all 210 candidate subjects: `subject_id`,
`baseline_day`, `conversion_day`, `pet_day`, `mri_day`, `mri_session`,
`outcome_label` (1 = converts to MCI/AD within 10 years, 0 = stable CN).

(Not included in this repository: it contains OASIS-3 subject IDs linked to
clinical outcomes and visit dates, and the OASIS-3 Data Use Agreement does not
permit redistribution of subject-level data. It can be regenerated from the
OASIS-3 clinical and imaging records by approved users.)

### 2. `check_pet_days.py`

Verifies each subject has a PIB PET scan within 30 days of their target
`pet_day`, matching against the raw PET directory. Reports exact matches,
close matches, and subjects with no usable scan.

### 3. FastSurfer

Run per subject via FastSurfer's own pipeline [3] (not included here).
Produces `FastSurfer_output/{mri_session}/mri/` containing `orig.mgz`,
`mask.mgz` and `aparc.DKTatlas+aseg.deep.mgz`. Every later step depends on
this output existing first.

### 4. `convert_all_mgz.py`

Converts `orig.mgz` to `orig.nii.gz`, since FLIRT needs a NIfTI reference
volume rather than MGZ.

### 5. `register_all_pet_v3_late.sh`

Requires FSL [4]. Averages the **final 9 frames** of each dynamic PIB
acquisition (`fslroi` then `fslmaths -Tmean`), then rigidly registers the
result onto the subject's MRI space (`flirt`, 6 DOF) using `orig.nii.gz` as
reference.

The late frames are the 5-minute frames constituting the amyloid binding
window; follows Vo et al. [1]

Outputs `PET_registered_v3/{subject_id}_PIB_in_MRI_v3.nii.gz` plus transform
matrices, and logs the frames used per subject to
`logs_v3/frame_selection_v3.csv`.


### 6. `MRI_Extraction.ipynb`

Extracts 6 anatomically-defined ROIs (bilateral hippocampus, cerebellar white
matter, cerebral white matter) from the FastSurfer segmentation.

**Pipeline**: mask ROI from segmentation → tight crop with 3-voxel padding →
z-score normalise over ROI voxels only → resize to 64×64×64 → cache as `.npy`.

### 7. `extract_pet_rois_v3.py`

Extracts the same 6 ROIs from `PET_registered_v3`, reusing the MRI-derived
segmentation masks — PET has no reliable anatomical detail of its own to
segment on. Identical crop → normalise → resize pipeline, so PET and MRI
caches stay dimensionally consistent for later fusion.

### 8. `extract_wholebrain_mri_v2.py` and `extract_wholebrain_pet_v4.py`

Whole-brain volumes at native 256³, masked with FastSurfer's `mask.mgz` and
z-scored over non-zero voxels.

### 9. `qc_audit.py`

Audits all four caches per subject for empty regions, failed z-scoring
(|mean| far from zero, indicating the region standard deviation was zero),
non-finite values, and implausible brain fractions. Writes `qc_report.csv`.

Identified nine subjects with failed hippocampal segmentation in the DKT
parcellation, five of which additionally had PET regions where z-scoring was
skipped, leaving raw intensities two orders of magnitude above the rest of the
cohort.

### 10. `make_clean_cohort.py`

Writes `thesis_cohort_clean.csv` (200 subjects, 101 converters / 99 stable)
and `excluded_subjects.csv`. Exclusion improves class balance from 107/103.

### 11. `augment_all_clean_tio.py`

Builds augmented caches for all four datasets in one pass, using torchio [6]:

```python
tio.Compose([
    tio.RandomAffine(scales=(0.95, 1.05), degrees=7, translation=2,
                     default_pad_value=0),
    tio.RandomFlip(axes=(0,)),
])
```

Three copies per training subject at seeds `[1, 101, 42]`, expanding
120 → 480 training samples. Validation and test sets are not augmented.

Three constraints relative to torchio's defaults, each addressing a measured
problem:

- **`default_pad_value=0`** — without it, space rotated in from outside the
  volume is filled with a non-zero value. On z-scored data (background exactly
  0, tissue ±3) this fabricates voxels above the magnitude of real tissue.
- **No elastic deformation** — the ROI crops are bounding boxes with 3 voxels
  of padding, so local warping displaces anatomy outside the crop.
- **Single-axis flips** — the six ROIs form three bilateral pairs with fixed
  left/right indices. A left–right flip
  would place a right-hemisphere structure at the index the model encodes as
  left.

Rotation is reduced from torchio's ±10° default to ±7°, because at ±10 a
corner voxel of a 64³ crop moves further than the available padding.

### 12. `PET_ROI_Images.ipynb`

Visual sanity checks: plots individual PET ROIs to confirm anatomically
sensible extraction, checks cache completeness and shape consistency across
all subjects, and includes a minimal end-to-end multimodal dataloader test.

---

## MNA-net comparison branch — MNI space

A parallel pipeline reproducing Vo et al.'s preprocessing [1], used only for
the baseline reproduction. **The main models do not use these caches.**

### Resolution: 1 mm native vs 2 mm MNI

The two pipelines operate at different resolutions, and this is not
interchangeable.

| | Main pipeline | MNA-net comparison |
|---|---|---|
| Space | Native subject | MNI152 |
| Resolution | 1 mm | 2 mm |
| Volume shape | 256³ | 91 × 109 × 91 |
| Reference | FastSurfer conformed | `MNI152_T1_2mm_brain` |

The main pipeline stays in native 1 mm space because the FastSurfer
segmentation is defined there.

The comparison branch must use 2 mm MNI because Vo's frozen patch encoders
were trained on volumes of exactly that shape.

### Steps

**MRI** — `orig.nii.gz` masked with `mask.mgz`, min-max scaled to [0,1], then
`flirt` to `MNI152_T1_2mm_brain` with `-cost corratio`.

**PET** — from `PET_averaged_late_v3`:

1. Smooth at 8 mm FWHM (`fslmaths -s 3.397`)
2. `bet -f 0.5 -g 0.1`
3. SynthStrip [5] via Docker, `-b 4`
4. Min-max scale to [0,1]
5. `flirt` to `MNI152_T1_2mm_brain`

The BET-then-SynthStrip order follows Vo's own script [1]. 

**Patching** — crop to 88 × 108 × 88, then a 3×3×3 grid of 44 × 54 × 44
patches with 50% overlap, giving 27 patches per modality.

---

## Output cache structure

| Cache | Contents | Shape |
|---|---|---|
| `preprocessed_cache_roi64/` | MRI ROIs, originals only | `(6, 64, 64, 64)` |
| `preprocessed_cache_pet_v3/` | PET ROIs, originals only, last-9-frame | `(6, 64, 64, 64)` |
| `preprocessed_cache_wholebrain_mri_v2/` | MRI whole brain, brain-masked | `(256, 256, 256)` |
| `preprocessed_cache_wholebrain_pet_v4/` | PET whole brain, brain-masked | `(256, 256, 256)` |
| `aug_clean_tio/roi_mri/` | MRI ROIs, originals plus augmented | `(6, 64, 64, 64)` |
| `aug_clean_tio/roi_pet/` | PET ROIs, originals plus augmented | `(6, 64, 64, 64)` |
| `aug_clean_tio/wb_mri/` | MRI whole brain, originals plus augmented | `(256, 256, 256)` |
| `aug_clean_tio/wb_pet/` | PET whole brain, originals plus augmented | `(256, 256, 256)` |

Augmented caches hold 560 files each: 200 originals plus 120 training subjects
× 3 augmentation seeds. MRI files are keyed by `mri_session`, PET files by
`subject_id`.

---

## References

[1] J. Vo, N. Sharif, and G. M. Hassan, "MNA-net: Multimodal neuroimaging attention-based architecture for cognitive decline prediction," in *Predictive Intelligence in Medicine (PRIME 2024)*, LNCS 15155, Springer, 2025, pp. 86–98.

[2] I. J. Khan et al., "Enhanced ROI guided deep learning model for Alzheimer's detection using 3D MRI images," *Informatics in Medicine Unlocked*, vol. 56, Art. no. 101650, 2025.

[3] L. Henschel, S. Conjeti, S. Estrada, K. Diers, B. Fischl, and M. Reuter, "FastSurfer — A fast and accurate deep learning based neuroimaging pipeline," *NeuroImage*, vol. 219, Art. no. 117012, 2020.

[4] M. Jenkinson, C. F. Beckmann, T. E. J. Behrens, M. W. Woolrich, and S. M. Smith, "FSL," *NeuroImage*, vol. 62, no. 2, pp. 782–790, 2012.

[5] A. Hoopes, J. S. Mora, A. V. Dalca, B. Fischl, and M. Hoffmann, "SynthStrip: Skull-stripping for any brain image," *NeuroImage*, vol. 260, Art. no. 119474, 2022.

[6] F. Pérez-García, R. Sparks, and S. Ourselin, "TorchIO: A Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning," *Computer Methods and Programs in Biomedicine*, vol. 208, Art. no. 106236, 2021.
