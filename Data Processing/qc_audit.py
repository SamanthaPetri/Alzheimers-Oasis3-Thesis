"""
qc_audit.py

Audits caches for data-quality problems before the
cohort is finalised, so bad subjects can be dropped

Checks per subject:

  ROI caches (6 x 64^3)
    - empty regions            (failed segmentation of that structure)
    - broken z-scoring         (|mean| far from 0 -> std was ~0, scaling skipped)
    - non-finite values        (NaN / inf from a divide-by-zero)
    - constant regions         (std ~ 0 -> no information)

  Whole-brain caches (256^3)
    - brain fraction           (too low -> failed mask/registration;
                                too high -> not masked at all)
    - broken z-scoring
    - non-finite values

Writes qc_report.csv and prints the union of subjects to exclude.
Nothing is deleted -- this only reports.
"""

import os
import glob

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────
COHORT_CSV = "D:/mamba_model/thesis_cohort_final.csv"

ROI_MRI = "D:/mamba_model/preprocessed_cache_roi64_aug"          # {session}_orig.npy
ROI_PET = "D:/mamba_model/preprocessed_cache_pet_v3"             # {subject}_PIB.npy
WB_MRI  = "D:/mamba_model/preprocessed_cache_wholebrain_native_mri_aug"   # {session}_orig.npy
WB_PET  = "E:/mamba_model/preprocessed_cache_wholebrain_native_pet_v3"    # {subject}_PIB.npy

OUT_CSV = "D:/mamba_model/qc_report.csv"

# Thresholds
MEAN_TOL      = 0.10     # |mean of nonzero| above this = z-scoring failed
STD_MIN       = 1e-3     # region with less spread than this is constant
BRAIN_FRAC_LO = 0.10     # whole-brain: less than 10% nonzero = failed mask
BRAIN_FRAC_HI = 0.95     # whole-brain: more than 95% nonzero = not masked

ROI_NAMES = ["L-Hippo", "R-Hippo", "L-Cereb-WM", "R-Cereb-WM",
             "L-Cerebral-WM", "R-Cerebral-WM"]

df = pd.read_csv(COHORT_CSV)
print(f"Cohort: {len(df)} subjects\n")


def check_roi(path):
    """Returns (list of problems, detail dict) for a 6 x 64^3 ROI array."""
    if not os.path.exists(path):
        return ["file missing"], {}
    a = np.load(path)
    problems, detail = [], {}

    if a.shape != (6, 64, 64, 64):
        problems.append(f"shape {a.shape}")
        return problems, detail
    if not np.isfinite(a).all():
        problems.append("non-finite values")

    empty, constant, badnorm = [], [], []
    max_abs_mean = 0.0
    for i in range(6):
        vox = a[i][a[i] != 0]
        if len(vox) == 0:
            empty.append(ROI_NAMES[i]); continue
        m, s = float(vox.mean()), float(vox.std())
        max_abs_mean = max(max_abs_mean, abs(m))
        if s < STD_MIN:
            constant.append(ROI_NAMES[i])
        if abs(m) > MEAN_TOL:
            badnorm.append(ROI_NAMES[i])

    if empty:
        problems.append(f"empty: {','.join(empty)}")
    if constant:
        problems.append(f"constant: {','.join(constant)}")
    if badnorm:
        problems.append(f"unnormalised: {','.join(badnorm)}")

    detail["n_empty"] = len(empty)
    detail["max_abs_mean"] = round(max_abs_mean, 3)
    return problems, detail


def check_wholebrain(path):
    """Returns (list of problems, detail dict) for a 256^3 volume."""
    if not os.path.exists(path):
        return ["file missing"], {}
    a = np.load(path)
    problems, detail = [], {}

    if a.shape != (256, 256, 256):
        problems.append(f"shape {a.shape}")
        return problems, detail
    if not np.isfinite(a).all():
        problems.append("non-finite values")

    frac = float((a != 0).mean())
    detail["brain_frac"] = round(frac, 3)

    if frac < BRAIN_FRAC_LO:
        problems.append(f"brain fraction {frac:.3f} too low")
    elif frac > BRAIN_FRAC_HI:
        problems.append(f"brain fraction {frac:.3f} -- not masked?")

    vox = a[a != 0]
    if len(vox):
        m = float(vox.mean())
        detail["mean_nonzero"] = round(m, 3)
        if abs(m) > MEAN_TOL:
            problems.append(f"unnormalised (mean {m:.2f})")
    return problems, detail


# Run
rows = []
for _, r in df.iterrows():
    sid, ses = str(r["subject_id"]).strip(), str(r["mri_session"]).strip()

    p_rm, d_rm = check_roi(f"{ROI_MRI}/{ses}_orig.npy")
    p_rp, d_rp = check_roi(f"{ROI_PET}/{sid}_PIB.npy")
    p_wm, d_wm = check_wholebrain(f"{WB_MRI}/{ses}_orig.npy")
    p_wp, d_wp = check_wholebrain(f"{WB_PET}/{sid}_PIB.npy")

    rows.append({
        "subject_id": sid,
        "mri_session": ses,
        "label": r["outcome_label"],
        "roi_mri":  "; ".join(p_rm),
        "roi_pet":  "; ".join(p_rp),
        "wb_mri":   "; ".join(p_wm),
        "wb_pet":   "; ".join(p_wp),
        "roi_mri_empty": d_rm.get("n_empty", ""),
        "roi_pet_empty": d_rp.get("n_empty", ""),
        "roi_pet_maxmean": d_rp.get("max_abs_mean", ""),
        "wb_mri_frac": d_wm.get("brain_frac", ""),
        "wb_pet_frac": d_wp.get("brain_frac", ""),
    })

qc = pd.DataFrame(rows)
qc.to_csv(OUT_CSV, index=False)

# Report
cols = ["roi_mri", "roi_pet", "wb_mri", "wb_pet"]
qc["any_problem"] = qc[cols].apply(lambda r: any(v for v in r), axis=1)
bad = qc[qc["any_problem"]]

print(f"Subjects with at least one problem: {len(bad)} of {len(qc)}\n")
for _, r in bad.iterrows():
    issues = [f"{c}[{r[c]}]" for c in cols if r[c]]
    print(f"  {r['subject_id']} (label {r['label']}): " + "  ".join(issues))

print("\n--- per-check counts ---")
for c in cols:
    print(f"  {c:8s}: {(qc[c] != '').sum()} affected")

print("\n--- brain fraction distribution (sanity) ---")
for c in ["wb_mri_frac", "wb_pet_frac"]:
    v = pd.to_numeric(qc[c], errors="coerce").dropna()
    if len(v):
        print(f"  {c}: min {v.min():.3f}  median {v.median():.3f}  max {v.max():.3f}")

exclude = sorted(bad["subject_id"].tolist())
print(f"\n--- proposed exclusions ({len(exclude)}) ---")
print(exclude)

kept = qc[~qc["any_problem"]]
print(f"\nRemaining: {len(kept)} subjects")
print("Class balance:", kept["label"].value_counts().to_dict())
print(f"\nFull report: {OUT_CSV}")
print("\nReview the report before excluding -- some flags (e.g. a single")
print("unnormalised region) may be tolerable, others (empty hippocampus) are not.")
