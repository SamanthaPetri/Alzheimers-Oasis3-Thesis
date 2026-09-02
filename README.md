# Alzheimer's Prediction Using OASIS-3 Dataset

Predicting 10-year conversion from cognitively normal (CN) to mild cognitive
impairment/Alzheimer's disease (MCI/AD) using multimodal MRI and PET imaging
from the OASIS-3 dataset, via a patch-based Vision Mamba architecture applied
to anatomically-defined brain regions.

**Repository structure**: `Data Processing/` (extraction, augmentation, quality
control) · `Models/` (v7 architecture and ablations) · `superseded/` (earlier
pipeline versions, archived).

> Earlier results are archived in `superseded/README.md`. They were produced
> on data containing PET frame-averaging, whole-brain masking, augmentation
> interpolation and cohort quality-control defects, all since corrected.

---

## Overview

This project implements and evaluates a **Vision Mamba** architecture —
patch-based tokenisation of anatomically-targeted MRI/PET regions, processed by
a bidirectional state-space model — for predicting AD conversion.

**v7** is the current architecture. It is a pure Vision Mamba: no convolutional
front end, no pretrained weights. Six anatomical ROIs are cut into 8³ patches
giving 3,072 tokens per modality, which pass directly to a bidirectional Mamba
encoder.

<img width="1622" height="487" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/a45dc910-c8a0-443a-a624-58e0fac3d95a" />

---

## Cohort

**200 subjects** (101 converters / 99 stable), split 120 / 40 / 40 at
`random_state=42`, stratified by outcome.

Ten subjects were excluded from the original 210 following quality control:

| Subject | Issue |
|---|---|
| OAS30041, OAS30234, OAS30241, OAS30379, OAS30662, OAS30867, OAS30919, OAS31132 | Both hippocampi absent from the DKT parcellation |
| OAS31103 | Right hippocampus absent |
| OAS30065 | Corrupt source PET (single volume; also excluded by Vo et al.) |

Five of the nine segmentation failures additionally had PET regions where
z-scoring was skipped because the region standard deviation was zero, leaving
raw intensities two orders of magnitude above the rest of the cohort.

Exclusion improves class balance from 107/103 to 101/99. Full audit in
`Data Processing/qc_audit.py`; the exclusion list is `excluded_subjects.csv`.

---

## Data Processing

**ROI extraction.** Six anatomically-defined regions per subject from
FastSurfer segmentations: bilateral hippocampus, cerebellar white matter,
cerebral white matter. Pipeline: mask by DKT label → crop with 3-voxel padding
→ z-score over ROI voxels only → resize to 64³ → cache.

**PET temporal averaging.** The final 9 frames of the dynamic PIB acquisition
are averaged — the 5-minute late frames constituting the amyloid binding
window. Earlier frames are perfusion-weighted and dilute the specific signal.
OASIS-3 PIB acquisitions vary in frame count (25–53 in this cohort); the fixed
last-9 selection follows Vo et al., who apply the same rule without
frame-count adjustment. Frame selection is logged per subject in
`logs_v3/frame_selection_v3.csv`.

**PET registration.** Rigid (6-DOF) FLIRT to the subject's own native MRI
space, so ROI extraction uses the same segmentation for both modalities. No
separate skull-stripping is required: cropping by segmentation label excludes
everything outside the structure by construction.

**Whole-brain volumes.** Masked with FastSurfer's `mask.mgz` and z-scored over
non-zero voxels, giving a ~7% brain fraction matched between modalities. The
DKT segmentation is deliberately *not* used for this purpose — it excludes
roughly 19% of brain volume, mostly cortical ribbon and CSF, and cortical
thinning is a primary structural marker of AD progression.

**Augmentation.** torchio, three copies per training subject at seeds
`[1, 101, 42]`, expanding 120 → 480 training samples

Three constraints relative to torchio's defaults, each with a specific reason:

- **`default_pad_value=0`.** Without it, space rotated in from outside the
  volume is filled with a non-zero value. On z-scored data this fabricates
  voxels *above* the magnitude of real tissue, which then pass the occupancy
  mask as valid tokens.
- **No elastic deformation.** The ROI crops are bounding boxes with 3 voxels
  of padding; local warping displaces anatomy outside the crop.
- **Single-axis flips.** The six ROIs form three bilateral pairs with fixed
  left/right indices and learned positional embeddings. A left–right flip
  would place a right-hemisphere structure at the index the model encodes as
  left.

Rotation is reduced from torchio's ±10° default to ±7° because at ±10 a corner
voxel of a 64³ crop moves further than the available padding.

---

## Hyperparameters

**Learning rate (1e-4) / weight decay (1e-3)**: conservative learning rate
reflects known early-training instability in Adam-family optimisers,
particularly relevant given the model trains end-to-end from random
initialisation on a small dataset (120 subjects) rather than fine-tuning a
pretrained checkpoint [1]. Weight decay follows AdamW's decoupled
formulation [2].

**ReduceLROnPlateau scheduling**: learning rate halved after 10 epochs without
validation-loss improvement.

**Weight initialisation**: factorised positional embeddings scaled by `*0.02`
at initialisation to prevent positional signal from dominating the network
before it has learned anything from the data [3]. Directly fixed an observed
~20-epoch slow-convergence issue in early training.

**`d_model=32`**: chosen to curb overfitting, since the positional embedding
table accounted for 98,304 parameters — 63.6% of the model at `d_model=64` —
which was disproportionate given the training set size [4].

**Dropout (0.4) / early stopping**: standard regularisation for a small-dataset
regime [5][6]. A minimum-epoch floor is applied before early stopping can
trigger, after several runs in earlier versions terminated during the
symmetry-breaking phase and retained near-initial weights.

**Data augmentation**: flipping is directly supported by Vim's own training
[7]; affine transformation follows general medical/volumetric imaging
practice [8]. The specific constraints applied here are justified above.

**Multi-seed evaluation (`[1, 7, 123]`)**: reporting mean ± std across 3 fixed
seeds, rather than a single run, follows documented evidence that random seed
alone can produce different outcomes in neural network training [9].

**PET frame selection**: averaging the late frames of a dynamic PIB
acquisition isolates specific amyloid binding from early perfusion-weighted
signal [10].

**References**

[1] L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han, "On the variance of the adaptive learning rate and beyond," in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[2] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[3] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[4] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning requires rethinking generalization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2017.

[5] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," J. Mach. Learn. Res., vol. 15, no. 56, pp. 1929–1958, 2014.

[6] L. Prechelt, "Early stopping—But when?," in Neural Networks: Tricks of the Trade, G. B. Orr and K.-R. Müller, Eds. Berlin, Germany: Springer, 1998, pp. 55–69.

[7] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, "Vision Mamba: Efficient visual representation learning with bidirectional state space model," arXiv preprint arXiv:2401.09417, 2024.

[8] L. Henschel, S. Conjeti, S. Estrada, K. Diers, B. Fischl, and M. Reuter, "FastSurfer—A fast and accurate deep learning based neuroimaging pipeline," NeuroImage, vol. 219, Art. no. 117012, 2020, doi: 10.1016/j.neuroimage.2020.117012.

[9] D. Picard, "Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision," arXiv preprint arXiv:2109.08203, 2021.

[10] Y. Li, R. Buchert, B. Schmitz-Koep, T. Grimmer, B. Ommer, D. M. Hedderich, I. Yakushev, and C. Wachinger, "Diffusion bridge networks simulate clinical-grade PET from MRI for dementia diagnostics," arXiv preprint arXiv:2510.15556, 2025.

[11] F. Pérez-García, R. Sparks, and S. Ourselin, "TorchIO: A Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning," Comput. Methods Programs Biomed., vol. 208, Art. no. 106236, 2021.

[12] J. Vo, N. Sharif, and G. M. Hassan, "MNA-net: Multimodal neuroimaging attention-based architecture for cognitive decline prediction," in Predictive Intelligence in Medicine (PRIME), LNCS 15155, Springer, 2025, pp. 86–98.

---

## v7 Results — ROI Vision Mamba

Mean ± sample standard deviation across seeds `[1, 7, 123]`. Each test subject
is 2.5 percentage points, so differences under ~5 points are within the
resolution of one or two subjects.

| Modality | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| MRI-only | 67.5% ± 4.3% | 60.0% ± 8.7% | 75.0% ± 15.0% |
| PET-only | 65.8% ± 3.8% | 61.7% ± 2.9% | 70.0% ± 5.0% |
| Multimodal | 66.7% ± 1.4% | 60.0% ± 5.0% | 73.3% ± 7.6% |

---

## Ablations

Identical data, split and seeds throughout. Only the stated component differs.

| Variant | Modality | Accuracy | TPR | TNR | Params | GFLOPs |
|---|---|---|---|---|---|---|
| **Vision Mamba** (baseline) | MRI | 67.5% ± 4.3% | 60.0% | 75.0% | 44,962 | 0.24 |
| | PET | 65.8% ± 3.8% | 61.7% | 70.0% | 44,962 | 0.24 |
| | Multimodal | 66.7% ± 1.4% | 60.0% | 73.3% | 89,922 | 0.47 |
| **Transformer encoder** | MRI | 52.5% ± 4.3% | 45.0% | 60.0% | 42,914 | 0.21 |
| | PET | 67.5% ± 6.6% | 60.0% | 75.0% | 42,914 | 0.21 |
| | Multimodal | 58.3% ± 5.2% | 56.7% | 60.0% | 85,826 | 0.41 |
| **CNN (scratch) + Mamba** | MRI | 66.7% ± 1.4% | 75.0% | 58.3% | 71,330 | 0.20 |
| | PET | 64.2% ± 3.8% | 56.7% | 71.7% | 71,330 | 0.20 |
| | Multimodal | 68.3% ± 3.8% | 68.3% | 68.3% | 142,658 | 0.41 |
| **MedicalNet + Mamba** | MRI | 66.7% ± 8.8% | 68.3% | 65.0% | 14,399,714 | 106.4 |
| | PET | 69.2% ± 6.3% | 60.0% | 78.3% | 14,399,714 | 106.4 |
| | Multimodal | 70.8% ± 3.8% | 53.3% | 88.3% | 28,799,426 | 212.7 |
| **Cross-modal attention** | Multimodal | 64.2% ± 6.3% | 65.0% | 63.3% | 98,370 | 0.47 |

**Transformer.** Replaces `VimEncoder` with `nn.TransformerEncoder` at matched
depth and width, keeping the same patch tokenisation. Worse than Mamba on MRI
(52.5% vs 67.5%) and multimodal (58.3% vs 66.7%), comparable on PET. Attention
is O(n²) in sequence length against Mamba's linear scan, so the comparison is
also unfavourable on cost.

**CNN arms.** Each ROI becomes a single token (6 rather than 3,072), so these
change the tokenisation as well as adding a convolutional front end — not a
pure "CNN vs no CNN" comparison.

**MedicalNet.** Highest multimodal accuracy of the ablations, but at 320× the
parameters and 450× the FLOPs of the baseline, with 88.3% specificity against
53.3% sensitivity. Several runs reached best validation loss by epoch 3–4,
indicating near-immediate memorisation of the 480 training samples.

**Cross-modal attention.** MRI and PET tokens attend to each other before
pooling, rather than being pooled independently and concatenated — the fusion
order used by MNA-net [12]. It does not improve accuracy (64.2% vs 66.7%) and
is the slowest ROI variant at 24 ms per sample, since it materialises a
3,072 × 3,072 attention matrix per head. Its value is interpretability: the
region weights below come from this model.

---

## Region-Pair Ablation

Each bilateral pair trained in isolation — 2 ROIs, 1,024 tokens, 0.079 GFLOPs
unimodal / 0.158 multimodal.

| Region pair | MRI | PET | Multimodal | Fusion gain |
|---|---|---|---|---|
| **Hippocampus** | 61.7% ± 1.4% | 66.7% ± 2.9% | **72.5% ± 0.0%** | +5.8 |
| Cerebellar WM | 59.2% ± 7.6% | 61.7% ± 2.9% | 60.8% ± 3.8% | −0.9 |
| Cerebral WM | 60.8% ± 2.9% | 66.7% ± 3.8% | 70.0% ± 5.0% | +3.3 |

Multimodal fusion helps where the region carries amyloid signal and does not
where it lacks one. Cerebellar white matter is the standard PET reference
region, selected precisely because it is spared in Alzheimer's, so its flat
response is anatomically expected.

Hippocampus multimodal exceeds the full six-region model at a third of the
computational cost, though the margin is two test subjects. The ±0.0% is
arithmetic rather than stability: all three seeds classified 29 of 40 correctly
but on *different* subjects (85–90% pairwise prediction agreement;
predicted-positive counts of 17, 23 and 19).

---

## Whole-Brain Comparison

Native-resolution whole-brain input, brain-masked with `mask.mgz`, 8³ patches
→ 32,768 tokens.

| Configuration | Modality | Accuracy | TPR | TNR | GFLOPs | Train time |
|---|---|---|---|---|---|---|
| Whole-brain Vision Mamba | MRI | 70.0%¹ | 75.0% | 65.0% | 2.52 | 298 min |
| | PET | 62.5%¹ | 50.0% | 75.0% | 2.52 | 395 min |
| | Multimodal | 62.5%¹ | 55.0% | 70.0% | 5.05 | 568 min |
| Whole-brain Transformer | MRI | *pending* | | | | |
| | PET | *pending* | | | | |
| | Multimodal | *pending* | | | | |
| Whole-brain CNN (scratch) + Mamba | MRI | *pending* | | | | |
| | PET | *pending* | | | | |
| | Multimodal | *pending* | | | | |
| Whole-brain MedicalNet + Mamba | MRI | *pending* | | | | |
| | PET | *pending* | | | | |
| | Multimodal | *pending* | | | | |

¹ Single seed. At ~300–570 minutes per seed this condition was not run to the
full three-seed protocol; the figures are indicative only.

Whole-brain input costs roughly ten times the FLOPs and forty times the
training time of the ROI
