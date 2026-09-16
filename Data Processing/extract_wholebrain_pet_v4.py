"""
extract_wholebrain_pet_v4.py

Rebuilds the whole-brain native-resolution PET cache from the v3
(last-9-frame) registered volumes, using the same brain mask as the MRI
whole-brain cache so the two modalities are finally consistent.

This version masks with FastSurfer's mask.mgz, matching
extract_wholebrain_mri_v2.py.

Source PET is PET_registered_v3/ -- the last-9-frame temporal average
(the amyloid binding window), registered to native MRI space.

Output: 256^3 float32, masked to brain, z-scored over non-zero voxels.
Expected brain fraction ~0.07, matching the MRI cache.

Run in Anaconda Prompt with the mamba_thesis environment active, AFTER
extract_wholebrain_mri_v2.py.
"""

import os
import glob
import shutil

import numpy as np
import nibabel as nib
import pandas as pd

# ── Config ────────────────────────────────────────────────────
OUT_ROOT   = "F:/mamba_model"
COHORT_CSV = "D:/mamba_model/thesis_cohort_clean.csv"
FS_OUTPUT  = "E:/Oasis3/FastSurfer_output"
PET_REG    = "E:/Oasis3/PET_registered_v3"

CACHE_DIR = f"{OUT_ROOT}/preprocessed_cache_wholebrain_pet_v4"
MRI_DIR   = f"{OUT_ROOT}/preprocessed_cache_wholebrain_mri_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

PET_SUFFIX = "_PIB_in_MRI_v3.nii.gz"
EXPECTED   = (256, 256, 256)

free_gb = shutil.disk_usage(OUT_ROOT[:3]).free / 1e9
print(f"Output: {CACHE_DIR}")
print(f"Free:   {free_gb:.1f} GB")
assert free_gb > 20, "need ~14GB for 200 volumes at 256^3 float32"

df = pd.read_csv(COHORT_CSV)
print(f"Cohort: {len(df)} subjects\n")

done = failed = 0
fail_log = []
fractions = []

for _, row in df.iterrows():
    sid = str(row["subject_id"]).strip()
    ses = str(row["mri_session"]).strip()

    out = f"{CACHE_DIR}/{sid}_PIB.npy"
    if os.path.exists(out):
        continue

    pet_p  = f"{PET_REG}/{sid}{PET_SUFFIX}"
    mask_p = f"{FS_OUTPUT}/{ses}/mri/mask.mgz"

    if not os.path.exists(pet_p):
        fail_log.append((sid, "no v3 PET")); failed += 1; continue
    if not os.path.exists(mask_p):
        fail_log.append((sid, "no mask.mgz")); failed += 1; continue

    try:
        vol = nib.load(pet_p).get_fdata()
        msk = nib.load(mask_p).get_fdata()

        if vol.ndim == 4:
            vol = vol.mean(axis=-1)
        if vol.shape != EXPECTED:
            fail_log.append((sid, f"shape {vol.shape}")); failed += 1; continue
        if msk.shape != vol.shape:
            fail_log.append((sid, "mask shape mismatch")); failed += 1; continue

        vol = vol * (msk > 0)
        vox = vol[vol != 0]
        if len(vox) == 0 or vox.std() == 0:
            fail_log.append((sid, "empty after mask")); failed += 1; continue

        frac = float((vol != 0).mean())
        fractions.append(frac)
        if frac < 0.03 or frac > 0.20:
            print(f"  ODD brain fraction {frac:.3f} for {sid}")

        vol = np.where(vol != 0, (vol - vox.mean()) / vox.std(), 0)
        np.save(out, vol.astype(np.float32))
        done += 1
        if done % 20 == 0:
            print(f"  ... {done}")

    except Exception as e:
        fail_log.append((sid, str(e))); failed += 1

print(f"\nDone: {done}  Failed: {failed}")
if fractions:
    f = np.array(fractions)
    print(f"Brain fraction: min {f.min():.3f}  median {np.median(f):.3f}  max {f.max():.3f}")
print(f"Files: {len(glob.glob(f'{CACHE_DIR}/*.npy'))}")
print(f"Size:  {sum(os.path.getsize(f) for f in glob.glob(f'{CACHE_DIR}/*.npy')) / 1e9:.1f} GB")

if fail_log:
    print("\nFailures:")
    for k, r in fail_log:
        print(f"  {k}: {r}")

# ── Consistency check against the MRI cache ───────────────────
mri_files = sorted(glob.glob(f"{MRI_DIR}/*.npy"))
pet_files = sorted(glob.glob(f"{CACHE_DIR}/*.npy"))
if mri_files and pet_files:
    m, p = np.load(mri_files[0]), np.load(pet_files[0])
    print("\n--- consistency check ---")
    for name, v in [("MRI v2", m), ("PET v4", p)]:
        print(f"  {name}: shape {v.shape}  brain frac {(v != 0).mean():.3f}  "
              f"mean(nonzero) {v[v != 0].mean():.4f}  std {v[v != 0].std():.3f}")
    print("\n  Both should show brain fraction ~0.07 and mean(nonzero) ~0.")
    print("  If they differ substantially, stop -- the modalities are not")
    print("  being masked the same way and the multimodal model will be unfair.")
