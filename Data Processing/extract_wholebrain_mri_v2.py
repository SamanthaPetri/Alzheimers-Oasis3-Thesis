"""
extract_wholebrain_mri_v2.py

Rebuilds the whole-brain native-resolution MRI cache with proper brain
masking.

Output: 256^3 float32, masked to brain, z-scored over non-zero voxels.
Expected brain fraction ~0.07.

Run in Anaconda Prompt with the mamba_thesis environment active.
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

CACHE_DIR = f"{OUT_ROOT}/preprocessed_cache_wholebrain_mri_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

EXPECTED = (256, 256, 256)

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
    ses = str(row["mri_session"]).strip()
    out = f"{CACHE_DIR}/{ses}.npy"
    if os.path.exists(out):
        continue

    orig_p = f"{FS_OUTPUT}/{ses}/mri/orig.nii.gz"
    mask_p = f"{FS_OUTPUT}/{ses}/mri/mask.mgz"

    if not os.path.exists(orig_p):
        fail_log.append((ses, "no orig.nii.gz")); failed += 1; continue
    if not os.path.exists(mask_p):
        fail_log.append((ses, "no mask.mgz")); failed += 1; continue

    try:
        vol = nib.load(orig_p).get_fdata()
        msk = nib.load(mask_p).get_fdata()

        if vol.shape != EXPECTED:
            fail_log.append((ses, f"shape {vol.shape}")); failed += 1; continue
        if msk.shape != vol.shape:
            fail_log.append((ses, "mask shape mismatch")); failed += 1; continue

        vol = vol * (msk > 0)
        vox = vol[vol != 0]
        if len(vox) == 0 or vox.std() == 0:
            fail_log.append((ses, "empty after mask")); failed += 1; continue

        frac = float((vol != 0).mean())
        fractions.append(frac)
        if frac < 0.03 or frac > 0.20:
            print(f"  ODD brain fraction {frac:.3f} for {ses}")

        vol = np.where(vol != 0, (vol - vox.mean()) / vox.std(), 0)
        np.save(out, vol.astype(np.float32))
        done += 1
        if done % 20 == 0:
            print(f"  ... {done}")

    except Exception as e:
        fail_log.append((ses, str(e))); failed += 1

print(f"\nDone: {done}  Failed: {failed}")
if fractions:
    f = np.array(fractions)
    print(f"Brain fraction: min {f.min():.3f}  median {np.median(f):.3f}  max {f.max():.3f}")
    print("  (expect ~0.07 -- the old unmasked cache was 0.66)")
print(f"Files: {len(glob.glob(f'{CACHE_DIR}/*.npy'))}")
print(f"Size:  {sum(os.path.getsize(f) for f in glob.glob(f'{CACHE_DIR}/*.npy')) / 1e9:.1f} GB")

if fail_log:
    print("\nFailures:")
    for k, r in fail_log:
        print(f"  {k}: {r}")

print("\nNEXT: run check_wb.py and look at the PNG before extracting PET.")
