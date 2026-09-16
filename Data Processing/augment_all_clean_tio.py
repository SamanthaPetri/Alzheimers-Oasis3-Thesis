"""
augment_all_clean_tio.py

Builds augmented caches for all four datasets from the 200-subject clean
cohort, using torchio, in one pass.

Datasets
--------
  roi_mri   MRI ROIs, 6 x 64^3, native space
  roi_pet   PET ROIs, 6 x 64^3, native space, last-9-frame (v3)
  wb_mri    MRI whole brain, 256^3, native, brain-masked (mask.mgz)
  wb_pet    PET whole brain, 256^3, native, brain-masked, last-9-frame

Written to F:/mamba_model/aug_clean_tio/<name>/ as
  {key}_orig.npy, {key}_aug1.npy, {key}_aug101.npy, {key}_aug42.npy
key = mri_session for MRI, subject_id for PET, matching the dataset classes.

Transform
---------
    RandomAffine(scales=0.95-1.05, degrees=7, translation=2,
                 default_pad_value=0)
    RandomFlip(axes=(0,))

Three deliberate constraints relative to torchio's defaults and to the
augmentation used by Vo et al.:

  * default_pad_value=0. Without this, torchio fills space rotated in from
    outside the volume with a non-zero value. On z-scored data (background
    exactly 0, tissue ~ +/-3) that fabricates voxels with magnitudes ABOVE
    real tissue, and they then pass the model's occupancy mask as valid
    tokens. Measured on a first attempt: whole-brain non-zero fraction went
    0.070 -> 0.149, with 1.4M invented voxels at median magnitude 2.68
    against real-tissue median 0.44.

  * No RandomElasticDeformation. The ROI crops are bounding boxes with 3
    voxels of padding, so local warping displaces anatomy outside the crop.

  * Flips on one axis only. The six ROIs are three bilateral pairs with
    fixed left/right indices and learned positional embeddings; a
    left-right flip would place a right-hemisphere structure at the index
    the model encodes as left.

Rotation is also reduced from torchio's +/-10 default to +/-7, because at
+/-10 a corner voxel of a 64^3 crop moves further than the 3-voxel padding.

Seeds [1, 101, 42], training subjects only, giving a 4x training set.

A padding check runs after each dataset and warns loudly if invented
voxels reappear.

Run in Anaconda Prompt with mamba_thesis active. Safe to interrupt and
re-run: existing files are skipped, truncated files removed first.
"""

import os
import glob
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torchio as tio
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────
COHORT_CSV = "D:/mamba_model/thesis_cohort_clean.csv"
OUT_ROOT   = "F:/mamba_model/aug_clean_tio"

AUG_SEEDS  = [1, 101, 42]
SPLIT_SEED = 42

DATASETS = {
    "roi_mri": dict(src="D:/mamba_model/preprocessed_cache_roi64_aug",
                    pattern="{key}_orig.npy", key="session", shape=(6, 64, 64, 64)),
    "roi_pet": dict(src="D:/mamba_model/preprocessed_cache_pet_v3",
                    pattern="{key}_PIB.npy",  key="subject", shape=(6, 64, 64, 64)),
    "wb_mri":  dict(src="F:/mamba_model/preprocessed_cache_wholebrain_mri_v2",
                    pattern="{key}.npy",      key="session", shape=(256, 256, 256)),
    "wb_pet":  dict(src="F:/mamba_model/preprocessed_cache_wholebrain_pet_v4",
                    pattern="{key}_PIB.npy",  key="subject", shape=(256, 256, 256)),
}

NAMES = ['L-Hippo', 'R-Hippo', 'L-Cereb-WM',
         'R-Cereb-WM', 'L-Cerebral-WM', 'R-Cerebral-WM']

# ── Cohort and split ──────────────────────────────────────────
df = pd.read_csv(COHORT_CSV)
sessions = df["mri_session"].values
labels   = df["outcome_label"].values

X_tv, X_test, y_tv, y_test = train_test_split(
    sessions, labels, test_size=0.2, random_state=SPLIT_SEED, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, random_state=SPLIT_SEED, stratify=y_tv)

session_to_subject = dict(zip(df["mri_session"], df["subject_id"]))

print(f"Cohort: {len(df)} subjects  "
      f"(train {len(X_train)} / val {len(X_val)} / test {len(X_test)})")
print(f"Class balance: {df['outcome_label'].value_counts().to_dict()}")

free_gb = shutil.disk_usage(OUT_ROOT[:3]).free / 1e9
print(f"F: free: {free_gb:.1f} GB")
assert free_gb > 100, "the two whole-brain caches need roughly 80 GB together"


# ── Transform ─────────────────────────────────────────────────
def augment(arr, seed):
    """(C, D, H, W) for ROIs, (D, H, W) for whole brain.
    torchio treats dim 0 as channels, so all six ROIs get one transform."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    t = tio.Compose([
        tio.RandomAffine(scales=(0.95, 1.05), degrees=7, translation=2,
                         default_pad_value=0),
        tio.RandomFlip(axes=(0,)),
    ])

    x = torch.from_numpy(arr).float()
    squeeze = (x.ndim == 3)
    if squeeze:
        x = x.unsqueeze(0)
    out = t(x)
    if squeeze:
        out = out.squeeze(0)
    return out.numpy().astype(np.float32)


def padding_check(o, a, label):
    """Warn if augmentation invented voxels where the original was zero."""
    new = (a != 0) & (o == 0)
    n = int(new.sum())
    if n == 0:
        print(f"    padding check ({label}): clean, no invented voxels")
        return True
    mag  = float(np.median(np.abs(a[new])))
    real = float(np.median(np.abs(o[o != 0])))
    print(f"    padding check ({label}): {n:,} new voxels, "
          f"median magnitude {mag:.3f} vs real tissue {real:.3f}")
    if mag > real * 0.5:
        print("    *** WARNING: padding is inventing tissue. "
              "default_pad_value is not taking effect. ***")
        return False
    print("    (small magnitudes -- interpolation at edges, acceptable)")
    return True


def build(name, cfg):
    src, pattern, keytype, shape = cfg["src"], cfg["pattern"], cfg["key"], cfg["shape"]
    dst = f"{OUT_ROOT}/{name}"
    os.makedirs(dst, exist_ok=True)
    expected_bytes = int(np.prod(shape)) * 4

    print(f"\n{'='*62}\n{name}\n  src {src}\n  dst {dst}")
    if not os.path.isdir(src):
        print(f"  SOURCE MISSING -- skipping"); return

    removed = 0
    for f in glob.glob(f"{dst}/*.npy"):
        if os.path.getsize(f) < expected_bytes - 1024:
            os.remove(f); removed += 1
    if removed:
        print(f"  removed {removed} truncated file(s)")

    def key_for(s):
        return s if keytype == "session" else session_to_subject[s]

    copied, missing = 0, []
    for ses in sessions:
        k = key_for(ses)
        out = f"{dst}/{k}_orig.npy"
        if os.path.exists(out):
            continue
        s = f"{src}/{pattern.format(key=k)}"
        if not os.path.exists(s):
            missing.append(k); continue
        arr = np.load(s)
        if arr.shape != shape:
            missing.append(f"{k} (shape {arr.shape})"); continue
        np.save(out, arr.astype(np.float32))
        copied += 1
        if copied % 25 == 0:
            print(f"    originals ... {copied}")
    print(f"  originals: {copied} copied, {len(missing)} missing")
    if missing:
        print(f"    {missing[:10]}")

    t0 = time.time()
    for seed in AUG_SEEDS:
        made, skipped = 0, 0
        for ses in X_train:
            k = key_for(ses)
            out = f"{dst}/{k}_aug{seed}.npy"
            if os.path.exists(out):
                continue
            orig = f"{dst}/{k}_orig.npy"
            if not os.path.exists(orig):
                skipped += 1; continue
            np.save(out, augment(np.load(orig), seed))
            made += 1
            if made % 25 == 0:
                print(f"    seed {seed} ... {made}")
        print(f"  seed {seed}: {made} new, {skipped} skipped "
              f"({(time.time()-t0)/60:.1f} min)")

    n_o = len(glob.glob(f"{dst}/*_orig.npy"))
    n_a = len(glob.glob(f"{dst}/*_aug*.npy"))
    size = sum(os.path.getsize(f) for f in glob.glob(f"{dst}/*.npy")) / 1e9
    print(f"  TOTAL: {n_o} orig + {n_a} aug = {n_o+n_a} files, {size:.1f} GB")

    cand = [k for k in (key_for(s) for s in X_train)
            if os.path.exists(f"{dst}/{k}_aug1.npy")]
    if not cand:
        return
    k = cand[0]
    o, a = np.load(f"{dst}/{k}_orig.npy"), np.load(f"{dst}/{k}_aug1.npy")
    print(f"  spot check {k}: identical {np.array_equal(o, a)} (want False)")
    padding_check(o, a, k)

    if o.ndim == 4:
        for i, nm in enumerate(NAMES):
            fo, fa = (o[i] != 0).mean(), (a[i] != 0).mean()
            flag = "  <-- >20% change" if abs(fa - fo) > fo * 0.2 else ""
            print(f"    {nm:14s} nonzero {fo:.3f} -> {fa:.3f}{flag}")
    else:
        fo, fa = (o != 0).mean(), (a != 0).mean()
        flag = "  <-- >20% change" if abs(fa - fo) > fo * 0.2 else ""
        print(f"    brain fraction {fo:.3f} -> {fa:.3f}{flag}")


start = time.time()
for name, cfg in DATASETS.items():
    build(name, cfg)

print(f"\n{'='*62}\nAll done in {(time.time()-start)/60:.1f} min")
for name in DATASETS:
    d = f"{OUT_ROOT}/{name}"
    if os.path.isdir(d):
        print(f"  {name:8s} {len(glob.glob(f'{d}/*.npy')):>5} files")
print("\nRead the padding checks above. Any WARNING means the caches are unusable.")
