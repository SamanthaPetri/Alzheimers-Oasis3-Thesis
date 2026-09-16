# Alzheimer's Prediction Using OASIS-3 Dataset

Predicting 10-year conversion from cognitively normal (CN) to mild cognitive
impairment/Alzheimer's disease (MCI/AD) using multimodal MRI and PET imaging
from the OASIS-3 dataset, via a patch-based Vision Mamba architecture [1]
applied to anatomically-defined brain regions.

**Repository structure**: `Data Processing/` (extraction, augmentation, quality
control) · `Models/` (architecture and ablations) · `superseded/` (earlier
pipeline versions, archived).

> Earlier results are archived in `superseded/README.md`. They were produced
> on data containing PET frame-averaging, whole-brain masking, augmentation
> interpolation and cohort quality-control defects, all since corrected.

---

## Overview

This project implements and evaluates a Vision Mamba architecture: patch-based
tokenisation of anatomically-targeted MRI/PET regions, processed by a
bidirectional state-space model, for predicting AD conversion.

The proposed model pools tokens within each of six anatomical regions and fuses
the MRI and PET summaries at each region through a shared attention layer. It
reaches **78.3% ± 6.3% accuracy at 0.47 GFLOPs**, with no convolutional backbone
and no pretrained weights.

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

Five of the nine segmentation failures also had PET regions where z-scoring was
skipped because the region standard deviation was zero, leaving raw intensities
two orders of magnitude above the rest of the cohort.

Full audit in `Data Processing/qc_audit.py`; the exclusion list is
`excluded_subjects.csv`.

---

## Data Processing

**ROI extraction.** Six regions per subject from FastSurfer [3] segmentations:
bilateral hippocampus, cerebellar white matter, cerebral white matter.
Pipeline: mask by DKT label → crop with 3-voxel padding → z-score over ROI
voxels only → resize to 64³ → cache.

**PET temporal averaging.** The final 9 frames of the dynamic PIB acquisition
are averaged, corresponding to the late amyloid binding window [4]. OASIS-3 PIB
acquisitions vary in frame count (25–53 in this cohort); the fixed last-9
selection follows Vo et al. [2], who apply the same rule without frame-count
adjustment. Frame selection is logged per subject in
`logs_v3/frame_selection_v3.csv`.

**PET registration.** Rigid (6-DOF) FLIRT to the subject's own native MRI
space, so ROI extraction uses the same segmentation for both modalities.
Cropping by segmentation label excludes everything outside the structure, so
no separate skull-stripping is applied for ROI inputs.

**Whole-brain volumes.** Masked with FastSurfer's `mask.mgz` and z-scored over
non-zero voxels, giving a ~7% non-zero fraction matched between modalities. The
DKT segmentation is not used for this purpose because it excludes roughly 19%
of brain volume, mostly cortical ribbon and CSF.

**Augmentation.** torchio [5], three copies per training subject at seeds
`[1, 101, 42]`, expanding 120 → 480 training samples. Settings differ from
torchio's defaults as follows:

- **`default_pad_value=0`.** Without it, space rotated in from outside the
  volume is filled with a non-zero value, which on z-scored data passes the
  occupancy mask as tissue.
- **No elastic deformation.** The ROI crops have only 3 voxels of padding.
- **Single-axis flips, no left–right flip.** The six ROIs form three bilateral
  pairs with fixed left/right indices and learned positional embeddings.
- **Rotation ±7°**. Enough flip to ensure ROIs not outisde cropped area.

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

---

## Results — proposed model

Mean ± sample standard deviation across seeds `[1, 7, 123]`. Each test subject
is 2.5 percentage points.

| Model | Accuracy | TPR | TNR | Params | GFLOPs |
|---|---|---|---|---|---|
| **Per-region modality attention** | **78.3% ± 6.3%** | 75.0% | 81.7% | 94,786 | 0.47 |
| Per-region concatenation (control) | 72.5% ± 6.6% | 75.0% | 70.0% | 90,562 | 0.47 |
| Baseline, mean pooling | 66.7% ± 1.4% | 60.0% | 73.3% | 89,922 | 0.47 |

The concatenation control separates the two changes relative to the baseline:
keeping the six region summaries separate rather than averaging them
(66.7% → 72.5%), then adding the attention layer (72.5% → 78.3%). Each step is
5.8 points, smaller than the seed standard deviation of either per-region
variant.

Per-seed accuracy for the proposed model is 85.0%, 77.5% and 72.5%, the widest
range of any configuration.

The proposed model requires both modalities, so it has no MRI-only or PET-only
form. Modality comparisons are reported from the baseline and ablations below.

---

## Baseline and ablations

Identical data, split and seeds throughout. Only the stated component differs.

| Variant | Modality | Accuracy | TPR | TNR | Params | GFLOPs |
|---|---|---|---|---|---|---|
| **Vision Mamba** (baseline) | MRI | 67.5% ± 4.3% | 60.0% | 75.0% | 44,962 | 0.24 |
| | PET | 65.8% ± 3.8% | 61.7% | 70.0% | 44,962 | 0.24 |
| | Multimodal | 66.7% ± 1.4% | 60.0% | 73.3% | 89,922 | 0.47 |
| **Transformer encoder** | MRI | 52.5% ± 4.3% | 45.0% | 60.0% | 42,914 | 0.21 |
| | PET | 67.5% ± 6.6% | 60.0% | 75.0% | 42,914 | 0.21 |
| | Multimodal | 58.3% ± 5.2% | 56.7% | 60.0% | 85,826 | 0.41 |
| **ResNet-10 + Mamba** (random init) | MRI | 72.5% ± 4.3% | 71.7% | 73.3% | 14,399,714 | 106.4 |
| | PET | 70.0% ± 4.3% | 61.7% | 78.3% | 14,399,714 | 106.4 |
| | Multimodal | 66.7% ± 5.8% | 58.3% | 75.0% | 28,799,426 | 212.7 |
| **ResNet-10 + Mamba** (MedicalNet) | MRI | 66.7% ± 8.8% | 68.3% | 65.0% | 14,399,714 | 106.4 |
| | PET | 69.2% ± 6.3% | 60.0% | 78.3% | 14,399,714 | 106.4 |
| | Multimodal | 70.8% ± 3.8% | 53.3% | 88.3% | 28,799,426 | 212.7 |
| **Cross-modal attention** | Multimodal | 64.2% ± 6.3% | 65.0% | 63.3% | 98,370 | 0.47 |

**Transformer.** Replaces `VimEncoder` with `nn.TransformerEncoder` at matched
depth and width, keeping the same patch tokenisation. Lower than Mamba on MRI
(52.5% vs 67.5%) and multimodal (58.3% vs 66.7%); slightly higher on PET
(67.5% vs 65.8%).

**ResNet-10 arms.** Architecturally identical; the only difference is whether
MedicalNet's pretrained weights are loaded. The randomly initialised trunk
scores higher on MRI (72.5% vs 66.7%) and PET (70.0% vs 69.2%), and lower on
multimodal (66.7% vs 70.8%). Several pretrained runs reached their best
validation loss by epoch 3–4. The pretrained trunk was fine-tuned unfrozen at
the same learning rate as the rest of the network. Both arms use about 320× the
parameters and 440–450× the FLOPs of the baseline, and both replace the patch
tokenisation with 6 region tokens, so neither isolates the convolutional front
end alone.

**Cross-modal attention.** MRI and PET tokens attend to each other before
pooling, rather than being pooled independently and concatenated, following the
fusion order of MNA-net [2]. Accuracy is 64.2% against 66.7% for the baseline,
and it is the slowest ROI variant at 24 ms per sample, since it computes a
3,072 × 3,072 attention matrix per head. The region weights below come from
this model.

---

## Region-pair ablation

Each bilateral pair trained in isolation: 2 ROIs, 1,024 tokens, 0.079 GFLOPs
unimodal / 0.158 multimodal.

| Region pair | MRI | PET | Multimodal | Multimodal vs best unimodal |
|---|---|---|---|---|
| **Hippocampus** | 61.7% ± 1.4% | 66.7% ± 2.9% | **72.5% ± 0.0%** | +5.8 |
| Cerebellar WM | 59.2% ± 7.6% | 61.7% ± 2.9% | 60.8% ± 3.8% | −0.9 |
| Cerebral WM | 60.8% ± 2.9% | 66.7% ± 3.8% | 70.0% ± 5.0% | +3.3 |

Hippocampus multimodal (72.5%) is 5.8 points above the full six-region baseline
(66.7%) at about a third of the FLOPs, a margin of roughly two test subjects.
The ±0.0% reflects identical accuracy, not identical predictions: all three
seeds classified 29 of 40 correctly but on different subjects (85–90% pairwise
prediction agreement; predicted-positive counts of 17, 23 and 19).

---

## Region attention weights

From the cross-modal attention model: the mean attention each key token
received across all queries, grouped by region. A uniform distribution would
give 1/3072 = 0.000326 per token.

| Region | MRI | vs uniform | PET | vs uniform |
|---|---|---|---|---|
| R-Cerebral-WM | 0.000449 | +37.9% | 0.000445 | +36.6% |
| L-Cerebral-WM | 0.000446 | +36.9% | 0.000444 | +36.3% |
| R-Cerebellar-WM | 0.000281 | −13.7% | 0.000277 | −15.0% |
| L-Cerebellar-WM | 0.000280 | −13.9% | 0.000276 | −15.1% |
| R-Hippocampus | 0.000251 | −22.8% | 0.000258 | −20.8% |
| L-Hippocampus | 0.000246 | −24.3% | 0.000254 | −22.0% |

Left and right differ by under 2% for every pair, values agree to three
significant figures across seeds, and MRI and PET tokens show the same
ordering. This ordering does not match the region-pair ablation, where
hippocampus gives the highest multimodal accuracy. The weights have not been
compared against per-region counts of occupied tokens and are reported as
descriptive only.

The proposed model's per-region modality attention weights are within half a
percentage point of 50% in every region and both directions, so they show no
modality preference and are not reported as an interpretability result.

---

## Whole-brain comparison

Native-resolution 256³ volumes, brain-masked with `mask.mgz`, 8³ patches →
32,768 tokens. Same architecture as the ROI baseline, so only the input
representation differs.

| Input | Modality | Accuracy | TPR | TNR | GFLOPs | Train time |
|---|---|---|---|---|---|---|
| Whole brain (32,768 tokens) | MRI | 70.0%¹ | 75.0% | 65.0% | 2.52 | 298 min |
| | PET | 62.5%¹ | 50.0% | 75.0% | 2.52 | 395 min |
| | Multimodal | 62.5%¹ | 55.0% | 70.0% | 5.05 | 568 min |
| ROI baseline (3,072 tokens) | MRI | 67.5% ± 4.3% | 60.0% | 75.0% | 0.24 | 8 min |
| | PET | 65.8% ± 3.8% | 61.7% | 70.0% | 0.24 | 9 min |
| | Multimodal | 66.7% ± 1.4% | 60.0% | 73.3% | 0.47 | 15 min |

¹ Single seed. At 300–570 minutes per run the three-seed protocol was not
feasible; these figures are indicative only.

Whole-brain input costs about ten times the FLOPs of ROI input. Whole-brain
accuracy is higher on MRI and lower on PET and multimodal, all within the
resolution of a single seed.

ROI models train from an in-memory cache while whole-brain volumes are streamed
from disk, so training times are not directly comparable. FLOPs are the
preferred efficiency measure.

The proposed model has no whole-brain equivalent: per-region fusion requires
anatomical regions, which a whole-brain volume does not define.

---

## Comparison to the published baseline

MNA-net [2] reports 82.9% / 85.7% / 80.0% on the same OASIS-3 prediction task
from a single seed, using 54 pretrained 3D ResNet-10 encoders (27 uniform
patches per modality). The published cohort (204 subjects) differs from the 200 used here; the
reproduction below uses the 209-subject cohort produced by Vo's own code.

| Model | Accuracy | TPR | TNR | Cost |
|---|---|---|---|---|
| MNA-net (published) | 82.9% | 85.7% | 80.0% | 54 × ResNet-10 |
| Per-region modality attention | 78.3% ± 6.3% | 75.0% | 81.7% | 0.47 GFLOPs |
| Hippocampus region pair | 72.5% ± 0.0% | 71.7% | 73.3% | 0.158 GFLOPs |

A separate reproduction using Vo's own frozen encoders and matched seeds
reached 80.4% ± 2.3% with PET skull-stripped by BET followed by SynthStrip; the best single seed reached 83.3%. Replacing the stage-3 concatenation with a Mamba sequence model gave 75.0%.

---

## Limitations

The test set contains 40 subjects, so each is worth 2.5 percentage points and
differences below roughly 5 points cannot be resolved. Results are from three
seeds on a single fixed split; cross-validation was not performed.

The proposed model has the widest seed range of any configuration
(72.5–85.0%), against ±1.4% for the baseline.

Whole-brain results are single-seed. The MedicalNet arm was fine-tuned without
freezing or layer-wise learning rates.

The Mamba encoder is the `mamba.py` implementation [12], which uses a
pure-PyTorch parallel scan rather than the fused CUDA kernel of the official
release. Reported times are specific to this implementation; FLOPs and
parameter counts are unaffected.

---

## Key findings

- **Mamba scored higher than a transformer encoder on MRI and multimodal** at
  matched depth, width and tokenisation (by 15.0 and 8.4 points), and 1.7
  points lower on PET.
- **Keeping region summaries separate and adding per-region modality attention
  gave the highest accuracy.** Each change added 5.8 points over the previous
  configuration, with seed standard deviations of 6.3–6.6.
- **ROI input costs about a tenth of the FLOPs of whole-brain input.**
  Whole-brain accuracy was not consistently higher (single seed).
- **Multimodal fusion improved on the best unimodal arm for hippocampus (+5.8)
  and cerebral WM (+3.3), but not cerebellar WM (−0.9).**
- **MedicalNet pretraining did not improve unimodal accuracy** over the same
  ResNet-10 trained from random initialisation.
- **Cross-modal attention weights rank cerebral WM highest and hippocampus
  lowest**, while region-pair accuracy ranks hippocampus highest.
- **The proposed model reaches 78.3% at 0.47 GFLOPs**, 4.6 points below the
  published MNA-net result on a different cohort.

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
