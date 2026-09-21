# Alzheimer's Prediction Using OASIS-3 Dataset

Predicting 10-year conversion from cognitively normal (CN) to mild cognitive
impairment/Alzheimer's disease (MCI/AD) using multimodal MRI and PET imaging
from the OASIS-3 dataset, via a patch-based Vision Mamba architecture [1]
applied to anatomically-defined brain regions.

**Repository structure**: `Data Processing/` (extraction, augmentation, quality
control) · `Models/` (architecture and comparisons) · `superseded/` (earlier
pipeline versions, archived).

> Earlier results are archived in `superseded/README.md`. They were produced
> on data containing PET frame-averaging, whole-brain masking, augmentation
> interpolation and cohort quality-control defects, all since corrected.

---

## Overview

This project implements and evaluates a Vision Mamba architecture: patch-based
tokenisation of anatomically-targeted MRI/PET regions for predicting AD conversion.

The proposed model pools tokens within each of six anatomical regions and fuses
the MRI and PET summaries at each region through a shared attention layer. It
reaches **78.3% ± 6.3% accuracy at 0.68 GFLOPs**.

## Proposed Model:
<img width="2182" height="782" alt="thesis_proposed" src="https://github.com/user-attachments/assets/e39d3525-5269-4fbb-93e8-8a6deca2e8f8" />

## Initial Model:
<img width="1622" height="487" alt="Architecture diagram" src="https://github.com/user-attachments/assets/a45dc910-c8a0-443a-a624-58e0fac3d95a" />

---

## Cohort

**200 subjects** (101 converters / 99 stable), split 120 / 40 / 40 at
`random_state=42`, stratified by outcome.

Ten subjects were excluded from the original 210 following quality control:

| Subject | Issue |
|---|---|
| OAS30041, OAS30234, OAS30241, OAS30379, OAS30662, OAS30867, OAS30919, OAS31132 | Both hippocampi absent from the DKT parcellation |
| OAS31103 | Right hippocampus absent |
| OAS30065 | Corrupt source PET (single volume; also excluded by Vo et al. [2]) |

---

## Data Processing

**ROI extraction.** Six regions per subject from FastSurfer [3] segmentations:
bilateral hippocampus, cerebellar white matter, cerebral white matter.
Pipeline: mask by DKT label, crop with 3-voxel padding, z-score over ROI
voxels only, resize to 64³ → cache.

**PET temporal averaging.** The final 9 frames of the dynamic PIB acquisition
are averaged, corresponding to the late amyloid binding window [4].

**PET registration.** Rigid (6-DOF) FLIRT to the subject's own native MRI
space, so ROI extraction uses the same segmentation for both modalities.
Cropping by segmentation label excludes everything outside the structure.

**Augmentation.** torchio [5], three copies per training subject at seeds
`[1, 101, 42]`, expanding 120 to 480 training samples. Settings differ from
torchio's defaults as follows:

- **`default_pad_value=0`.** Without it, space rotated in from outside the
  volume is filled with a non-zero value, which on z-scored data passes the
  occupancy mask as tissue.
- **No elastic deformation.** The ROI crops have only 3 voxels of padding.
- **Single-axis flips, no left–right flip.** Avoids altering left-right ROIs.
- **Rotation ±7°**. Enough rotation to ensure ROIs not outisde cropped area.

---

## Hyperparameters

**Learning rate (1e-4) / weight decay (1e-3)**: a conservative learning rate
given early-training instability in Adam-family optimisers [6] and training
from random initialisation on 120 subjects. Weight decay uses AdamW's decoupled
formulation [7].

**ReduceLROnPlateau scheduling**: learning rate halved after 10 epochs without
validation-loss improvement.

**Positional embedding initialisation**: factorised positional embeddings are
scaled by 0.02 at initialisation. Introduced after early runs showed slow
convergence over the first ~20 epochs.

**`d_model=32`**: chosen to limit overfitting, since the factorised positional
embedding tables account for a substantial share of total parameters and scale
directly with `d_model` [8].

**Dropout (0.4) / early stopping**: regularisation for a small training set
[9][10]. A minimum-epoch floor is applied before early stopping can trigger,
after runs in earlier versions stopped with near-initial weights.

**Multi-seed evaluation (`[1, 7, 123]`)**: mean ± sample standard deviation
across 3 fixed seeds, since the random seed alone can change training
outcomes [11].

**GFLOPs**: counted for one forward pass at batch size 1, covering the
convolutions and linear layers, the attention, and the Mamba selective scan.

---

## Results: proposed model

Mean ± sample standard deviation across seeds `[1, 7, 123]`. Each test subject
is 2.5 percentage points.

| Model | Accuracy | TPR | TNR | Params | GFLOPs |
|---|---|---|---|---|---|
| **Late per-region attention (proposed)** | **78.3% ± 6.3%** | 75.0% | 81.7% | 94,786 | 0.68 |
| Late per-region concatenation | 70.8% ± 2.9% | 75.0% | 66.7% | 90,562 | 0.68 |
| Global mean pooling (initial model) | 66.7% ± 1.4% | 60.0% | 73.3% | 89,922 | 0.68 |

Per-seed accuracy for the proposed model is 85.0%, 77.5% and 72.5%

The proposed model requires both modalities, so it has no MRI-only or PET-only
form. Modality comparisons are reported from the initial model and the
comparison models below.

---

## Initial model and comparisons

Identical data, split and seeds throughout. Only the stated component differs.

| Variant | Modality | Accuracy | TPR | TNR | Params | GFLOPs |
|---|---|---|---|---|---|---|
| **Global mean pooling** (initial model) | MRI | 67.5% ± 4.3% | 60.0% | 75.0% | 44,962 | 0.34 |
| | PET | 65.8% ± 3.8% | 61.7% | 70.0% | 44,962 | 0.34 |
| | Multimodal | 66.7% ± 1.4% | 60.0% | 73.3% | 89,922 | 0.68 |
| **Transformer encoder** | MRI | 52.5% ± 4.3% | 45.0% | 60.0% | 42,914 | 2.67 |
| | PET | 67.5% ± 6.6% | 60.0% | 75.0% | 42,914 | 2.67 |
| | Multimodal | 58.3% ± 5.2% | 56.7% | 60.0% | 85,826 | 5.34 |
| **3D CNN tokenisation** (from scratch) | MRI | 66.7% ± 1.4% | 75.0% | 58.3% | 71,330 | 0.20 |
| | PET | 64.2% ± 3.8% | 56.7% | 71.7% | 71,330 | 0.20 |
| | Multimodal | 68.3% ± 3.8% | 68.3% | 68.3% | 142,658 | 0.41 |
| **Pretrained ResNet-10 tokenisation** | MRI | 66.7% ± 8.8% | 68.3% | 65.0% | 14,399,714 | 106.2 |
| | PET | 69.2% ± 6.3% | 60.0% | 78.3% | 14,399,714 | 106.2 |
| | Multimodal | 70.8% ± 3.8% | 53.3% | 88.3% | 28,799,426 | 212.4 |
| **Early cross-modal attention** | Multimodal | 64.2% ± 2.9% | 65.0% | 63.3% | 98,370 | 3.15 |


**Transformer encoder.** Replaces `VimEncoder` with `nn.TransformerEncoder` at
matched depth and width, keeping the same patch tokenisation. Lower than Mamba
on MRI (52.5% vs 67.5%) and multimodal (58.3% vs 66.7%); slightly higher on PET
(67.5% vs 65.8%). It costs 2.67 GFLOPs against Mamba's 0.34 at the same
sequence length.

**Tokenisation arms.** Both replace the 512-patch-per-ROI tokenisation with a
CNN producing one token per region, shortening the sequence from 3,072 to 6.
The from-scratch CNN performs comparably to patch tokenisation across all three
modalities. The MedicalNet-pretrained ResNet-10 uses about 320× the parameters
and 312× the FLOPs of the initial model for its largest gain of 4.1 points on
multimodal.

**Early cross-modal attention.** MRI and PET tokens attend to each other before
pooling, rather than being pooled independently and concatenated. Applying attention across all 3,072 tokens rather
than to six pairs of regional summaries costs 3.15 GFLOPs against the proposed
model's 0.68, a difference of 2.47 GFLOPs in the attention products alone.

---

## Region-pair ablation

Each bilateral pair trained in isolation: 2 ROIs, 1,024 tokens, 0.11 GFLOPs
unimodal / 0.23 multimodal.

| Region pair | MRI | PET | Multimodal | Multimodal vs best unimodal |
|---|---|---|---|---|
| **Hippocampus** | 61.7% ± 1.4% | 66.7% ± 2.9% | **72.5% ± 0.0%** | +5.8 |
| Cerebellar WM | 59.2% ± 7.6% | 61.7% ± 2.9% | 60.8% ± 3.8% | −0.9 |
| Cerebral WM | 60.8% ± 2.9% | 66.7% ± 3.8% | 70.0% ± 5.0% | +3.3 |

Hippocampus multimodal (72.5%) is 5.8 points above the six-region initial model
(66.7%) at about a third of the FLOPs.

---

## Region attention weights

**Proposed model — per-region modality attention.** At each region the MRI and
PET summaries attend to each other, so each weight is the share one modality
draws from the other. 50% means the region relies on both equally.

| Region | MRI query to PET | PET query to MRI |
|---|---|---|
| L-Hippocampus | 50.0% | 49.8% |
| R-Hippocampus | 50.4% | 49.5% |
| L-Cerebellar-WM | 50.0% | 49.7% |
| R-Cerebellar-WM | 49.9% | 49.9% |
| L-Cerebral-WM | 50.4% | 49.7% |
| R-Cerebral-WM | 50.3% | 50.0% |

Every value is within half a percentage point of 50% in both directions, so
the attention expresses no modality preference. The accuracy gain over
concatenation therefore does not come from the model weighting one modality
above the other at any region.


---

## Whole-brain comparison

Native-resolution 256³ volumes, brain-masked with `mask.mgz`, 8³ patches
(32,768 tokens). Same architecture as the initial model, so only the input
representation differs. Batch size 1 rather than 4, as activation memory
scales with sequence length.

| Input | Modality | Accuracy | TPR | TNR | GFLOPs | Train time |
|---|---|---|---|---|---|---|
| Whole brain (32,768 tokens) | MRI | 70.0%¹ | 75.0% | 65.0% | 3.62 | 298 min |
| | PET | 62.5%¹ | 50.0% | 75.0% | 3.62 | 395 min |
| | Multimodal | 62.5%¹ | 55.0% | 70.0% | 7.25 | 568 min |
| Six-ROI input, initial model (3,072 tokens) | MRI | 67.5% ± 4.3% | 60.0% | 75.0% | 0.34 | 8 min |
| | PET | 65.8% ± 3.8% | 61.7% | 70.0% | 0.34 | 9 min |
| | Multimodal | 66.7% ± 1.4% | 60.0% | 73.3% | 0.68 | 15 min |

¹ Single seed. At 300–570 minutes per run the three-seed protocol was not
feasible; these figures are indicative only.

Whole-brain input costs about ten times the FLOPs of ROI input. Whole-brain
accuracy is higher on MRI and lower on PET and multimodal.

ROI models train from an in-memory cache while whole-brain volumes are streamed
from disk, so training times are not directly comparable. FLOPs are the
preferred efficiency measure.

The encoder is the `mamba.py` implementation
[12], a pure-PyTorch parallel scan rather than the fused CUDA kernel of the
official release, so times are specific to it; FLOPs and parameter counts are
unaffected.

The proposed model has no whole-brain equivalent: per-region fusion requires
anatomical regions, which a whole-brain volume does not define.

---

## Key findings

- **Mamba scored higher than a transformer encoder on MRI and multimodal** at
  matched depth, width and tokenisation (by 15.0 and 8.4 points), at an eighth
  of the FLOPs, and 1.7 points lower on PET.
- **Keeping region summaries separate and adding per-region modality attention
  gave the highest accuracy**, 78.3% against 66.7% for the initial model, for
  4,224 extra parameters and 0.0001 GFLOPs.
- **ROI input costs about a tenth of the FLOPs of whole-brain input.**
  Whole-brain accuracy was not consistently higher (single seed).
- **Hippocampus gives the highest region-pair accuracy**, with multimodal input
  reaching 72.5%, above the six-region initial model, at a third of the FLOPs.
- **Where attention is applied matters more than whether it is used.**

---

## References

[1] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, "Vision Mamba: Efficient visual representation learning with bidirectional state space model," arXiv:2401.09417, 2024.

[2] J. Vo, N. Sharif, and G. M. Hassan, "MNA-net: Multimodal neuroimaging attention-based architecture for cognitive decline prediction," in *Predictive Intelligence in Medicine (PRIME)*, LNCS 15155, Springer, 2025, pp. 86–98.

[3] L. Henschel, S. Conjeti, S. Estrada, K. Diers, B. Fischl, and M. Reuter, "FastSurfer—A fast and accurate deep learning based neuroimaging pipeline," *NeuroImage*, vol. 219, Art. no. 117012, 2020.

[4] Y. Li, R. Buchert, B. Schmitz-Koep, T. Grimmer, B. Ommer, D. M. Hedderich, I. Yakushev, and C. Wachinger, "Diffusion bridge networks simulate clinical-grade PET from MRI for dementia diagnostics," arXiv:2510.15556, 2025.

[5] F. Pérez-García, R. Sparks, and S. Ourselin, "TorchIO: A Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning," *Comput. Methods Programs Biomed.*, vol. 208, Art. no. 106236, 2021.

[6] L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han, "On the variance of the adaptive learning rate and beyond," in *Proc. ICLR*, 2020.

[7] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.

[8] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning requires rethinking generalization," in *Proc. ICLR*, 2017.

[9] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929–1958, 2014.

[10] L. Prechelt, "Early stopping—But when?," in *Neural Networks: Tricks of the Trade*, G. B. Orr and K.-R. Müller, Eds. Berlin, Germany: Springer, 1998, pp. 55–69.

[11] D. Picard, "Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision," arXiv:2109.08203, 2021.

[12] A. Torres-Leguet, "mamba.py: A simple, hackable and efficient Mamba implementation in pure PyTorch and MLX," 2024. [Online]. Available: https://github.com/alxndrTL/mamba.py
