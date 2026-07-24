# Alzheimer's Prediction Using OASIS-3 Dataset

Predicting 10-year conversion from cognitively normal (CN) to mild cognitive impairment/Alzheimer's disease (MCI/AD) using multimodal MRI and PET imaging from the OASIS-3 dataset, via a patch-based Vision Mamba architecture applied to anatomically-defined brain regions.

**Repository structure**: `Data Processing/` (extraction, augmentation, ROI statistics) · `Models/` (v1–v5 model iterations, see per-folder READMEs for detail)

---

## Overview

This project implements and evaluates a **Vision Mamba** architecture — patch-based tokenisation of anatomically-targeted MRI/PET regions, processed by a bidirectional state-space model — for predicting AD conversion from a 210-subject cohort (126 train / 42 val / 42 test, stratified). Six architecture generations (v1–v5) were developed, each addressing a specific limitation identified in the previous version. **v4** is the project's primary reported architecture; all other versions are ablations or corrective iterations.

---

## Data Processing (Summary)

- **6 anatomically-defined ROIs** extracted per subject from FastSurfer segmentations: bilateral hippocampus, cerebellum white matter, cerebral white matter
- **Pipeline**: mask ROI → crop (+padding) → z-score normalise (ROI voxels only) → resize to 64×64×64 → cache
- **Augmentation**: 3 copies per training subject (random rotation ±15°, random flips), seeds `[1, 101, 42]`, expanding 126 → 504 training samples
- **PET preprocessing**: registered to MRI space via FLIRT; original 4D dynamic frames averaged into a single 3D volume before registration
- **ROI statistics** (v4_stats): pre-resize voxel count, mean/std intensity, and bounding-box dimensions extracted separately, restoring detail lost during the 64³ resize
- **Region-appropriate padding** (v4_recropped): padding tuned per region (hippocampus +10, cerebellum-WM +6, cerebral-WM +3) instead of one flat value, after visual/quantitative evidence that uniform padding disproportionately blurred smaller structures

Full detail in `Data Processing/` sub-folder READMEs.

---

## Model Version Comparison

| Version | Change from previous | Status |
|---|---|---|
| CNN-Mamba hybrid | Earlier, non-Vision-Mamba pipeline-validation trial | Not used |
| v1 | Initial architecture — custom bidirectional wrapper (not faithful Vision Mamba) | Superseded |
| v2 | `d_model` 64→32 to curb overfitting | Superseded |
| v2_cali | + threshold calibration | Not carried forward |
| v2_3seed | + formal multi-seed evaluation protocol | Superseded |
| v3 | Real `VMamba` encoder, factorised embeddings, background masking | Superseded |
| **v4** | **Clean refactor of v3 — primary reported architecture** | **Active** |
| v4_stats | + auxiliary ROI statistics | **Active** |
| v4_ROIs | Individual/paired-region ablation (uniform padding) | **Active** |
| v4_ROIs_recropped | Paired-region ablation, region-appropriate padding | **Active** |
| v4_recropped | Full 6-ROI model, region-appropriate padding | **Active** |
| v4_wholebrain | Whole-brain input, token-matched to v4 | Comparison |
| v5 | `d_model` 32→64 revisited (single seed) | Not carried forward |

Full version history, references, and result-by-result reasoning in `Models/README.md`.


<img width="1621" height="487" alt="Thesis drawio" src="https://github.com/user-attachments/assets/1ca135c0-8575-4b53-a501-c7d9471c2363" />

---

## v4 — Hyperparameters

**Learning rate (1e-4) / weight decay (1e-3)**: conservative learning rate reflects known early-training instability in Adam-family optimisers, particularly relevant given the model trains end-to-end from random initialisation on a small dataset (126 subjects) rather than fine-tuning a pretrained checkpoint [1]. Weight decay follows AdamW's decoupled formulation [2].

**ReduceLROnPlateau scheduling**: learning rate halved after 10 epochs without validation-loss improvement — standard adaptive scheduling, allowing larger early steps and finer late-stage convergence.

**Weight initialisation**: factorised positional embeddings scaled by `*0.02` at initialisation, matching the convention used in GPT-2's released implementation [3], to prevent positional signal from dominating the network before it has learned anything from the data. Directly fixed an observed ~20-epoch slow-convergence issue in early v3 training.

**`d_model=32`**: adopted to curb overfitting from an oversized positional embedding table in the original architecture (98,304 parameters, 63.6% of the model at `d_model=64`) [4]. Retained through v4 even after v3's factorised embeddings independently resolved that issue; revisited in v5 (`d_model=64`), which showed no improvement on the single seed tested.

**Dropout (0.4) / early stopping**: standard regularisation for a small-dataset regime [5][6].

**Data augmentation**: horizontal flipping is directly supported by Vim's own training [7]; rotation follows general medical/volumetric imaging practice [8].

**Multi-seed evaluation (`[1, 7, 123]`)**: reporting mean ± std across 3 fixed seeds, rather than a single run, follows documented evidence that random seed alone can produce different outcomes in neural network training [9]

**PET frame averaging before registration**: standard PET preprocessing practice, combining dynamic frames into a single reference volume prior to spatial registration [10].

**References**
[1] L. Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond," ICLR, 2020.
[2] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019.
[3] A. Radford et al., "Language Models Are Unsupervised Multitask Learners," OpenAI, 2019 (initialisation convention per released GPT-2 implementation).
[4] C. Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization," ICLR, 2017.
[5] N. Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR, 2014.
[6] L. Prechelt, "Early Stopping — But When?," in *Neural Networks: Tricks of the Trade*, Springer, 1998.
[7] L. Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model," arXiv:2401.09417, 2024.
[8] L. Henschel et al., "FastSurfer — A Fast and Accurate Deep Learning Based Neuroimaging Pipeline," *NeuroImage*, 2020.
[9] D. Picard, "torch.manual_seed(3407) is all you need," arXiv:2109.08203, 2021.
[10] Y. Li et al., "Diffusion Bridge Networks Simulate Clinical-grade PET from MRI for Dementia Diagnostics," arXiv:2510.15556, 2025.

---

## v4 Results (3-Seed Mean ± Std, Test Set)

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MRI-only | 68.3% ± 2.7% | 63.5% ± 7.3% | 73.0% ± 2.7% |
| PET-only | 62.7% ± 5.0% | 68.3% ± 12.0% | 57.1% ± 4.8% |
| Multimodal | 65.1% ± 1.4% | 58.7% ± 5.5% | 71.4% ± 4.8% |
| MRI-only + ROI stats | 65.1% ± 1.4% | 71.4% ± 0.0% | 58.7% ± 2.8% |
| PET-only + ROI stats | 66.7% ± 2.4% | 66.7% ± 0.0% | 66.7% ± 4.8% |
| Hippocampus pair only | 56.3% ± 7.3% | 60.3% ± 12.0% | 52.4% ± 12.6% |
| Cerebellum-WM pair only | 65.1% ± 2.7% | 50.8% ± 9.9% | 79.4% ± 15.3% |
| Cerebral-WM pair only | 69.0% ± 0.0% | 61.9% ± 8.2% | 76.2% ± 8.2% |
| Whole-brain MRI (token-matched) | 56.3% ± 2.7% | 65.1% ± 7.3% | 47.6% ± 12.6% |
| Whole-brain PET (token-matched) | 63.5% ± 1.4% | 52.4% ± 0.0% | 74.6% ± 2.7% |
| Whole-brain Multimodal (token-matched) | 63.5% ± 1.4% | 55.6% ± 2.7% | 71.4% ± 0.0% |

**For reference**: MNA-net baseline (Vo et al.) — 82.9% / 85.7% / 80.0% (single seed).

---

## Key Findings

- ROI-based input outperforms token-matched whole-brain input, particularly for MRI
- Auxiliary ROI statistics helped PET, not MRI — a modality-specific rather than uniform effect
- Region-appropriate crop padding improved hippocampal performance but not cerebellum-WM, suggesting padding needs depend on how tightly a structure fills its native bounding box, not just its absolute size
- All configurations remain below the MNA-net baseline, consistent with attention/SSM architectures' known need for larger training data than the 126-subject cohort available
