# Mamba Model Version History

A chronological log of every model iteration in this project — what changed, why, and what it revealed.

## CNN-Mamba Hybrid Trial (Test work)

`ROIEncoderMamba`: a shared 3D CNN encoder reduces each of the 6 ROIs down to a single feature vector (one token per ROI, 6 tokens total), then a lightweight custom Mamba block learns relationships between those 6 tokens before classification.

- **Ablation baseline**: `FullBrainMamba` — same approach applied to a single masked whole-brain volume, no ROI separation
- **Extension**: `ROIStatsMamba` — adds pre-resize ROI statistics (volume, intensity, bounding box) into the same architecture
- 5-seed evaluation (`[42, 101, 1, 7, 123]`)

### Why It Was Not Used as the Final Architecture

**It is not a genuine Vision Mamba implementation.** Vision Mamba, as defined in the literature, tokenises an image by dividing it into small patches and processing each patch as its own token — this preserves fine-grained spatial structure across the whole sequence. `ROIEncoderMamba` does the opposite: a CNN collapses each entire ROI down to a single summary token *before* the Mamba block ever runs, so the sequence model only ever operates over 6 coarse tokens (one per region), never over patch-level spatial detail. This makes it a CNN–SSM hybrid architecture, not a Vision Mamba architecture 


**It is not a novel architecture.** CNN-feature-extraction-then-SSM designs of this kind are an established pattern already explored in prior published work.
(J. Ma, F. Li, and B. Wang, "U-Mamba: Enhancing Long-Range Dependency for Biomedical Image Segmentation," arXiv:2401.04722, 2024.)

## v1 — Initial Architecture (Not True Vision Mamba)

Custom `BiMambaWrapper` built on plain `mambapy.mamba.Mamba` (causal-only, single-direction). Runs a full forward-pass stack and a full backward-pass stack (on the reversed sequence) independently, then merges them once at the end.

- `d_model=64`, all 6 ROIs in one model, MRI + PET + multimodal
- **Known issue**: this "merge after two separate stacks" pattern matches Vim's own published *"Bidirectional Block"* ablation — which the authors found underperforms plain causal Mamba. Not a faithful Vision Mamba implementation.

**References**
- [`mambapy`](https://github.com/alxndrTL/mamba.py) — third-party Mamba implementation
- Zhu et al. (2024), *[Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)*
- Loshchilov & Hutter (2019), *[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)* (ICLR) — AdamW optimizer, used throughout

---

## v2 — Capacity Reduced

Same `BiMambaWrapper` as v1, with `d_model` cut from 64 to 32 to fight overfitting from an oversized positional embedding table.

- Same MRI + PET + multimodal structure

**References**
- [`mambapy`](https://github.com/alxndrTL/mamba.py) — unchanged from v1
- Zhang et al. (2017), *[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)* (ICLR) — general justification for weighing parameter count against dataset size, applied to this project's 126-subject cohort

---

## v2_cali — Threshold Calibration Experiment

Same `BiMambaWrapper` architecture as v2. Adds post-hoc classification threshold tuning — determines thresholds against the validation set instead of using a fixed 0.5 cutoff.

- **Result**: did not improve performance, overfit to the small (~42-subject) validation set. Not carried forward.

---

## v2_3seed — Formalized Multi-Seed Evaluation

Same `BiMambaWrapper` architecture. First version to run 3 fixed seeds (`[1, 7, 123]`) per configuration for proper mean ± std reporting, rather than single-run results.

---

## v3 — First Proper Vision Mamba Implementation

Replaces `BiMambaWrapper` with `mambapy.vim.VMamba`, matching Vim's published algorithm

- Introduces factorised positional embeddings (separate small ROI/depth/height/width embedding tables, added together), replacing one large per-patch-position table
- Adds background-token masking before the encoder, so empty (non-anatomical) patches don't leak into real tokens during the bidirectional scan
- Introduces dropout (0.4) before the final classifier, and early stopping on validation loss
- `d_model=32`, all 6 ROIs, MRI + PET + multimodal

**References**
- [`mambapy.vim.VMamba`](https://github.com/alxndrTL/mamba.py)
- Zhu et al. (2024), *Vision Mamba*
- Srivastava et al. (2014), *[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)* (JMLR) — justification for dropout given small dataset size
- Prechelt (1998), *Early Stopping — But When?* — justification for early-stopping regularization

---

## v4 — Clean Refactor of v3

Architecturally identical to v3 (same `VMamba` encoder, same factorised embeddings, same masking). Restructured with clearer comments and organisation only

- Serves as the base architecture for all `v4_*` variants below

---

## v4_avg — Feature Averaging

Same v4 architecture and branches, but combines the pooled MRI and PET feature vectors by averaging instead of concatenating before the classifier.
- Classifier input size stays at `d_model` (32) rather than `d_model*2` (64), since averaging keeps vector length unchanged unlike concatenation
- **Result**: accuracy matched concatenation exactly (65.1% ± 1.4% for both), but with substantially higher seed-to-seed variance in sensitivity and specificity (TPR std 13.7% vs 5.5%; TNR std 12.6% vs 4.8%) — same overall accuracy reached via a less stable trade-off between error types, consistent with averaging discarding modality-specific information that concatenation preserves

--- 

## v4_stats — Auxiliary ROI Statistics

Adds a `stats_proj` linear layer that fuses 6 pre-resize statistics per ROI (voxel count, mean/std intensity, bounding-box dimensions) into the pooled representation.

- Same `VMamba` backbone as v4; tests whether restoring information lost during the 64³ resize improves prediction
- **Result**: improved PET-only accuracy and reduced variance; did not improve MRI-only

---

## v4_ROIs — Individual / Paired-Region Ablation

Same v4 backbone, but trains separate models on single left+right hemisphere pairs (Hippocampus, Cerebellum-WM, Cerebral-WM) instead of all 6 ROIs jointly.

- Uses the original cropping (uniform pad=3 for all regions)
- Purpose: assess individual regional contribution to prediction (explainability)


---

## v4_ROIs_recropped — Region-Appropriate Padding (Paired Regions)

Same paired-region structure as v4_ROIs, using a new cache with region-specific crop padding: larger padding for small structures (hippocampus, pad=10) to reduce upsampling blur, moderate for cerebellum-WM (pad=6), unchanged for cerebral-WM (pad=3, already well-sized).

- **Result**: hippocampus improved modestly; cerebellum-WM got worse; cerebral-WM unchanged

**References**
- `FastSurfer` segmentation labels — Henschel et al. (2020), *[FastSurfer — A fast and accurate deep learning based neuroimaging pipeline](https://doi.org/10.1016/j.neuroimage.2020.117012)* (NeuroImage) — source of anatomical masks and z-score intensity normalization convention

---

## v4_recropped — Region-Appropriate Padding (Full 6-ROI Model)

Same full 6-ROI, MRI + PET + multimodal structure as v4, using the same custom-padding cache as v4_ROIs_recropped — applied to the complete 6-ROI model rather than individual pairs.

**References**
- Same as v4_ROIs_recropped

---

## v4_wholebrain — Whole-Brain Input Instead of ROIs

Same `VMamba` backbone; replaces the 6 discrete ROI inputs with a single resized whole-brain volume, resized to match the original 6-ROI setup's token count (3,072 tokens) exactly — so the comparison is compute-matched, not just "smaller input."

- Includes `WholeBrainModel` / `MultimodalWholeBrainModel` variants (MRI, PET, multimodal)
- **Purpose**: directly test the proposal's core hypothesis — anatomically-targeted ROI input vs. whole-brain input, holding compute fixed
- **Result**: whole-brain underperformed ROI-based input across all three configurations, most notably for MRI

**References**
- I. J. Khan et al., “Enhanced ROI guided deep learning model for Alzheimer's detection using 3D MRI images,” Informatics in Medicine Unlocked, vol. 56, p. 101665, 2025. 
- Token-matching calculation original to this project

---

## v4_wholebrain_fullres — Whole-Brain Input, Native Resolution
Same as v4_wholebrain, but without the token-matching downsample — MRI (256³, FastSurfer conformed) and PET (native registered resolution) are used as-is, with no resizing or ROI cropping/masking at all.
- Token count at 256³: 32,768 (vs 3,072 in the token-matched version)
---

## v5 — Capacity Test (d_model=64 Revisited)

Same v3/v4 `VMamba` backbone; `d_model` raised back to 64 (single-seed test).

- Motivation: the original justification for capping `d_model` at 32 (v2) was to curb the old oversized positional-embedding table. Since v3's factorised embeddings already fixed that independent of `d_model`, this checks whether 32 became an unnecessarily tight cap.
- **Result**: did not show improvement over `d_model=32` on the single seed tested

**References**
- Parameter-to-data ratio reasoning benchmarked against Zhu et al. (2024), *Vision Mamba* — Vim-Tiny's published parameter count (7M) and ImageNet training set size (1.28M images) used as a reference ratio
- Zhang et al. (2017), *Understanding Deep Learning Requires Rethinking Generalization* (ICLR)

---

## v6 — Cross-Modal Attention Fusion
Adds multi-head cross-attention between MRI and PET token sequences before pooling and concatenation, following Vo et al.'s MNA-net fusion order (attention first, then concatenation) rather than v4's plain concatenation.
- Also used to extract region-level attention weights for explainability, aggregated per anatomical region relative to the uniform-attention baseline (1/3,072)
- **Result**: single-seed accuracy did not clearly outperform base concatenation or probability fusion; attention weights ranked cerebral white matter highest and hippocampus lowest, matching the v4 accuracy ranking

**References**
J. Vo, N. Sharif and G. M. Hassan, “MNA-Net: Multimodal neuroimaging attention-based
architecture for cognitive decline prediction,” in Predictive Intelligence in Medicine (PRIME
2024), Lecture Notes in Computer Science, vol. 15155, Cham, Switzerland, 2025. 

---

## Results

All 3-seed results use seeds `[1, 7, 123]`, reported as mean ± sample standard deviation (ddof=1). v5 & v6 are single-seed (seed 1).

### Reference Baselines

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MNA-net (Vo et al.) | 82.9% | 85.7% | 80.0% |
| Whole-brain trial (CNN-Mamba hybrid) | 62.4% | — | — |
| ROIStatsMamba (CNN-Mamba hybrid, MRI-only) | ~72.0% | ~76.0% | ~69.0% |

### v4 — Baseline

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MRI-only | 68.3% ± 2.7% | 63.5% ± 7.3% | 73.0% ± 2.7% |
| PET-only | 62.7% ± 5.0% | 68.3% ± 12.0% | 57.1% ± 4.8% |
| Multimodal (joint-trained) | 65.1% ± 1.4% | 58.7% ± 5.5% | 71.4% ± 4.8% |

### v4_avg — Feature Averaging
| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| Multimodal, feature averaging | 65.1% ± 1.4% | 63.5% ± 13.7% | 66.7% ± 12.6% |

### v4_stats — Auxiliary ROI Statistics

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MRI-only + ROI statistics | 65.1% ± 1.4% | 71.4% ± 0.0% | 58.7% ± 2.8% |
| PET-only + ROI statistics | 66.7% ± 2.4% | 66.7% ± 0.0% | 66.7% ± 4.8% |

### v4_ROIs — Individual/Paired-Region Ablation (Uniform Padding)

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| Hippocampus pair only | 56.3% ± 7.3% | 60.3% ± 12.0% | 52.4% ± 12.6% |
| Cerebellum-WM pair only | 65.1% ± 2.7% | 50.8% ± 9.9% | 79.4% ± 15.3% |
| Cerebral-WM pair only | 69.0% ± 0.0% | 61.9% ± 8.2% | 76.2% ± 8.2% |

### v4_ROIs_recropped — Region-Appropriate Padding

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| Hippocampus pair only | 58.7% ± 7.3% | 60.3% ± 5.5% | 57.1% ± 19.0% |
| Cerebellum-WM pair only | 57.1% ± 2.4% | 49.2% ± 2.7% | 65.1% ± 7.3% |
| Cerebral-WM pair only (control, padding unchanged) | 69.0% ± 0.0% | 61.9% ± 8.2% | 76.2% ± 8.2% |

### v4_wholebrain — Token-Matched Whole-Brain Input

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MRI-only | 56.3% ± 2.7% | 65.1% ± 7.3% | 47.6% ± 12.6% |
| PET-only | 63.5% ± 1.4% | 52.4% ± 0.0% | 74.6% ± 2.7% |
| Multimodal | 63.5% ± 1.4% | 55.6% ± 2.7% | 71.4% ± 0.0% |

### v4_wholebrain_fullres — Native Whole-Brain Input

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MRI-only | 57.9% ± 6.9% | 68.3% ± 29.1% | 47.6% ± 42.3% |
| PET-only | 64.3% ± 0.0% | 57.1% ± 0.0% | 71.4% ± 0.0% |
| Multimodal | 69.0% ± 2.4% | 58.7% ± 2.7% | 79.4% ± 5.5% |

### v5 — Capacity Test

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MRI-only, `d_model=64` (single seed) | 66.7% | 66.7% | 66.7% |

### v6 — Cross-Modal Attention Fusion
| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| Multimodal, cross-attention + concatenation (single seed) | 64.3% | 61.9% | 66.7% |
