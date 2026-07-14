import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────
FS_OUTPUT  = 'E:/Oasis3/FastSurfer_output'
PET_REG    = 'E:/Oasis3/PET_registered_v2'
CACHE_DIR  = 'D:/mamba_model/preprocessed_cache_pet'
COHORT     = 'D:/mamba_model/thesis_cohort_final.csv'

os.makedirs(CACHE_DIR, exist_ok=True)

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
    pet_path = f'{PET_REG}/{subject_id}_PIB_in_MRI.nii.gz'

    seg_data = nib.load(seg_path).get_fdata()
    pet_data = nib.load(pet_path).get_fdata()

    # Handle 4D PET just in case
    if pet_data.ndim == 4:
        pet_data = pet_data.mean(axis=-1)

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
        z2 = min(seg_data.shape[0]-1, coords[0].max() + pad)
        y1 = max(0, coords[1].min() - pad)
        y2 = min(seg_data.shape[1]-1, coords[1].max() + pad)
        x1 = max(0, coords[2].min() - pad)
        x2 = min(seg_data.shape[2]-1, coords[2].max() + pad)

        cropped = roi_vol[z1:z2+1, y1:y2+1, x1:x2+1]

        # Z-score normalise (ROI voxels only)
        roi_voxels = cropped[cropped != 0]
        if len(roi_voxels) > 0 and roi_voxels.std() > 0:
            cropped = np.where(
                cropped != 0,
                (cropped - roi_voxels.mean()) / roi_voxels.std(),
                0
            )

        zoom_factors = [t/s for t, s in zip(roi_size, cropped.shape)]
        resized = zoom(cropped, zoom_factors, order=1)
        rois[roi_name] = resized.astype(np.float32)

    return rois

# ── Process all subjects ──────────────────────────────────────
df = pd.read_csv(COHORT)
done = 0
skipped = 0
failed = 0

for _, row in df.iterrows():
    subject_id  = row['subject_id']
    mri_session = row['mri_session']
    out_path    = f'{CACHE_DIR}/{subject_id}_PIB.npy'

    if os.path.exists(out_path):
        skipped += 1
        continue

    pet_path = f'{PET_REG}/{subject_id}_PIB_in_MRI.nii.gz'
    if not os.path.exists(pet_path):
        print(f'MISSING registered PET: {subject_id}')
        failed += 1
        continue

    try:
        rois = extract_pet_rois(subject_id, mri_session)
        roi_array = np.stack(list(rois.values()))
        np.save(out_path, roi_array)
        done += 1
        print(f'Done {subject_id} ({done})')
    except Exception as e:
        print(f'FAILED {subject_id}: {e}')
        failed += 1

print(f'\nDone: {done}, Skipped: {skipped}, Failed: {failed}')