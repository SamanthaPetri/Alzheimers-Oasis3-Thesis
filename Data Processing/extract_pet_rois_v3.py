"""
extract_pet_rois_v3.py

Extracts the 6 anatomical ROIs from the v3 PET volumes (late-frame averaging).

"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom

# ── Paths ─────────────────────────────────────────────────────
FS_OUTPUT = 'E:/Oasis3/FastSurfer_output'
PET_REG   = 'E:/Oasis3/PET_registered_v3'                    # v3
CACHE_DIR = 'D:/mamba_model/preprocessed_cache_pet_v3'       # v3
COHORT    = 'D:/mamba_model/thesis_cohort_final.csv'
os.makedirs(CACHE_DIR, exist_ok=True)

PET_SUFFIX = '_PIB_in_MRI_v3.nii.gz'                         # v3 filename pattern
EXCLUDE    = {'OAS30065'}                                    # corrupt source PET

# ── ROI labels ────────────────────────────────────────────────
ROI_INFO = {
    'Left-Hippocampus':    17,
    'Right-Hippocampus':   53,
    'Left-Cerebellum-WM':   7,
    'Right-Cerebellum-WM': 46,
    'Left-Cerebral-WM':     2,
    'Right-Cerebral-WM':   41,
}
ROI_SIZE = (64, 64, 64)


# ── PET ROI extraction ────────────────────────────────────────
def extract_pet_rois(subject_id, mri_session, roi_size=ROI_SIZE):
    seg_path = f'{FS_OUTPUT}/{mri_session}/mri/aparc.DKTatlas+aseg.deep.mgz'
    pet_path = f'{PET_REG}/{subject_id}{PET_SUFFIX}'

    seg_data = nib.load(seg_path).get_fdata()
    pet_data = nib.load(pet_path).get_fdata()

    # v3 volumes are already 3D, but keep the guard
    if pet_data.ndim == 4:
        pet_data = pet_data.mean(axis=-1)

    if seg_data.shape != pet_data.shape:
        raise ValueError(f'shape mismatch: seg {seg_data.shape} vs pet {pet_data.shape}')

    rois = {}
    for roi_name, label_id in ROI_INFO.items():
        mask    = (seg_data == label_id)
        roi_vol = pet_data * mask

        coords = np.where(mask)
        if len(coords[0]) == 0:
            rois[roi_name] = np.zeros(roi_size, dtype=np.float32)
            continue

        pad = 3
        z1 = max(0, coords[0].min() - pad)
        z2 = min(seg_data.shape[0] - 1, coords[0].max() + pad)
        y1 = max(0, coords[1].min() - pad)
        y2 = min(seg_data.shape[1] - 1, coords[1].max() + pad)
        x1 = max(0, coords[2].min() - pad)
        x2 = min(seg_data.shape[2] - 1, coords[2].max() + pad)

        cropped = roi_vol[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]

        # Z-score normalise (ROI voxels only)
        roi_voxels = cropped[cropped != 0]
        if len(roi_voxels) > 0 and roi_voxels.std() > 0:
            cropped = np.where(
                cropped != 0,
                (cropped - roi_voxels.mean()) / roi_voxels.std(),
                0
            )

        zoom_factors = [t / s for t, s in zip(roi_size, cropped.shape)]
        resized = zoom(cropped, zoom_factors, order=1)
        rois[roi_name] = resized.astype(np.float32)

    return rois


# ── Process all subjects ──────────────────────────────────────
df = pd.read_csv(COHORT)
done = skipped = failed = excluded = 0
fail_log = []

for _, row in df.iterrows():
    subject_id  = str(row['subject_id']).strip()
    mri_session = str(row['mri_session']).strip()

    if subject_id in EXCLUDE:
        print(f'EXCLUDED {subject_id} (corrupt source PET, matches Vo et al.)')
        excluded += 1
        continue

    out_path = f'{CACHE_DIR}/{subject_id}_PIB.npy'
    if os.path.exists(out_path):
        skipped += 1
        continue

    pet_path = f'{PET_REG}/{subject_id}{PET_SUFFIX}'
    if not os.path.exists(pet_path):
        print(f'MISSING v3 PET: {subject_id}')
        fail_log.append((subject_id, 'missing PET'))
        failed += 1
        continue

    seg_path = f'{FS_OUTPUT}/{mri_session}/mri/aparc.DKTatlas+aseg.deep.mgz'
    if not os.path.exists(seg_path):
        print(f'MISSING segmentation: {subject_id} ({mri_session})')
        fail_log.append((subject_id, 'missing seg'))
        failed += 1
        continue

    try:
        rois = extract_pet_rois(subject_id, mri_session)
        roi_array = np.stack(list(rois.values()))          # (6, 64, 64, 64)
        np.save(out_path, roi_array)
        done += 1
        if done % 20 == 0:
            print(f'  ... {done} extracted')
    except Exception as e:
        print(f'FAILED {subject_id}: {e}')
        fail_log.append((subject_id, str(e)))
        failed += 1

print(f'\nDone: {done}  Skipped(existing): {skipped}  Excluded: {excluded}  Failed: {failed}')
print(f'Cache: {CACHE_DIR}')
print(f'Files present: {len([f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")])}')

if fail_log:
    print('\nFailures:')
    for sid, reason in fail_log:
        print(f'  {sid}: {reason}')
