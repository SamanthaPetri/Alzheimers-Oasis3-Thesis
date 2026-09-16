# Alzheimer's Prediction Using OASIS-3 Dataset

Predicting 10-year conversion from cognitively normal (CN) to mild cognitive
impairment/Alzheimer's disease (MCI/AD) using multimodal MRI and PET imaging
from the OASIS-3 dataset, via a patch-based Vision Mamba architecture applied
to anatomically-defined brain regions.

**Repository structure**: `Data Processing/` (extraction, augmentation, quality
control) · `Models/` (architecture and ablations) · `superseded/` (earlier
pipeline versions, archived).

> Earlier results are archived in `superseded/README.md`. They were produced
> on data containing PET frame-averaging, whole-brain masking, augmentation
> interpolation and cohort quality-control defects, all since corrected.

---

## Overview

This project implements and evaluates a **Vision Mamba** architecture
(patch-based tokenisation of anatomically-targeted MRI/PET regions, processed by
a bidirectional state-space model) for predicting AD conversion.

The proposed model pools tokens within each of six anatomical regions and fuses
the MRI and PET summaries at each region through a shared attention layer,
reaching **78.3% accuracy at 0.47 GFLOPs**. There is no convolutional feature
extractor and no pretrained weights.

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
| OAS30065 | Corrupt source PET (single volume; also excluded by Vo et al. [12]) |

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
`[1, 101, 42]`, expanding 120 → 480 training samples.

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

**`d_model=32`**: chosen to curb overfitting, since the factorised positional
embedding tables account for a substantial share of total parameters and scale
directly with `d_model` — a concern given the training set size [4].

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

---

## Results - proposed model

Per-region modality attention. Mean ± sample standard deviation across seeds
`[1, 7, 123]`. Each test subject is 2.5 percentage points, so differences under
~5 points are within the resolution of one or two subjects.

| Model | Accuracy | TPR | TNR | Params | GFLOPs |
|---|---|---|---|---|---|
| **Per-region modality attention** | **78.3% ± 6.3%** | 75.0% | 81.7% | 94,786 | 0.47 |
| Per-region concatenation (control) | 72.5% ± 6.6% | 75.0% | 70.0% | 90,562 | 0.47 |
| Baseline, mean pooling | 66.7% ± 1.4% | 60.0% | 73.3% | 89,922 | 0.47 |

Two changes contribute roughly equally to the 11.6-point gain over the
baseline: keeping the six region summaries separate rather than averaging them
(66.7% → 72.5%), and adding the attention layer (72.5% → 78.3%). The
concatenation row is the control that separates them.

The proposed model is multimodal by construction (per-region fusion requires
both modalities) so it has no MRI-only or PET-only form. Modality comparisons
are reported from the baseline and ablations below.

This variant also shows the widest seed spread of any configuration (range
72.5–85.0%), discussed under *Limitations*.

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
depth and width, keeping the same patch tokenisation. Worse than Mamba on MRI
(52.5% vs 67.5%) and multimodal (58.3% vs 66.7%), comparable on PET. Attention
is O(n²) in sequence length against Mamba's linear scan, so the comparison is
also unfavourable on cost.

**ResNet-10 arms.** Architecturally identical; the only difference is whether
MedicalNet's pretrained weights are loaded. Pretraining does not transfer on
this task - the randomly initialised model is better on MRI (72.5% vs 66.7%)
and PET (70.0% vs 69.2%), and worse only on multimodal (66.7% vs 70.8%).
Several pretrained runs reached best validation loss by epoch 3–4, indicating
near-immediate memorisation of the 480 training samples. Both arms cost 320×
the parameters and 450× the FLOPs of the baseline, and both replace the patch
tokenisation with 6 region tokens, so neither isolates the convolutional front
end alone.

**Cross-modal attention.** MRI and PET tokens attend to each other before
pooling, rather than being pooled independently and concatenated — the fusion
order used by MNA-net [12]. It does not improve accuracy (64.2% vs 66.7%) and
is the slowest ROI variant at 24 ms per sample, since it materialises a
3,072 × 3,072 attention matrix per head. Its value is interpretability: the
region weights below come from this model.

---

## Region-pair ablation

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

Hippocampus multimodal exceeds the full six-region baseline at a third of the
computational cost, though the margin is two test subjects. The ±0.0% is
arithmetic rather than stability: all three seeds classified 29 of 40 correctly
but on *different* subjects (85–90% pairwise prediction agreement;
predicted-positive counts of 17, 23 and 19).

---

## Region attention weights

Cross-modal attention, measured as the mean attention each key token *received*
across all queries, grouped by region. This quantity is not normalised by
softmax's row-sum-to-one property, unlike a per-query weighting. Baseline with
no preference is 1/3072 = 0.000326.

| Region | MRI | vs baseline | PET | vs baseline |
|---|---|---|---|---|
| R-Cerebral-WM | 0.000449 | +37.9% | 0.000445 | +36.6% |
| L-Cerebral-WM | 0.000446 | +36.9% | 0.000444 | +36.3% |
| R-Cerebellar-WM | 0.000281 | −13.7% | 0.000277 | −15.0% |
| L-Cerebellar-WM | 0.000280 | −13.9% | 0.000276 | −15.1% |
| R-Hippocampus | 0.000251 | −22.8% | 0.000258 | −20.8% |
| L-Hippocampus | 0.000246 | −24.3% | 0.000254 | −22.0% |

Three properties support this as an anatomical signal rather than an artefact
of token ordering: left and right differ by under 2% for every pair; values
reproduce to three significant figures across all three seeds; and the MRI and
PET branches, trained independently, converge on the same ordering.

Attention magnitude and standalone predictive value do **not** align. Cerebral
white matter receives the most attention, but hippocampus gives the best
region-pair accuracy in the multimodal case. A region can be attended to
without being the most useful in isolation.

The proposed model's own per-region attention weights are uniform to within
half a percentage point in every region and both directions, so its accuracy
gain comes from the attention layer's learned projections rather than from any
learned modality preference. Those weights are not reported as an
interpretability result.

---

## Whole-brain comparison

Native-resolution 256³ volumes, brain-masked with `mask.mgz`, 8³ patches →
32,768 tokens. Same architecture as the ROI baseline, so this isolates the
input representation.

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

Whole-brain input costs roughly ten times the FLOPs and forty times the
training time for no consistent accuracy gain, which is the basis for the
region-based approach. The proposed model has no whole-brain equivalent:
per-region fusion operates on anatomical regions, which a whole-brain volume
does not define.

ROI models train from an in-memory cache while whole-brain volumes are streamed
from disk, so training times between the two are not directly comparable.
FLOPs and inference latency are hardware-independent and should be preferred
for efficiency claims.

---

## Comparison to the published baseline

MNA-net [12] reports 82.9% / 85.7% / 80.0% on the same OASIS-3 task, using 54
pretrained 3D ResNet-10 encoders over 27 uniform patches.

| Model | Accuracy | TPR | TNR | Cost |
|---|---|---|---|---|
| MNA-net (published) | 82.9% | 85.7% | 80.0% | 54 × ResNet-10 |
| Per-region modality attention | 78.3% ± 6.3% | 75.0% | 81.7% | 0.47 GFLOPs |
| Hippocampus region pair | 72.5% ± 0.0% | 71.7% | 73.3% | 0.158 GFLOPs |

A separate reproduction using Vo's own frozen encoders and matched seeds
recovered 80.4% ± 2.3%, with the best single seed reaching
83.3%. Substituting Mamba for the concatenation in his patch-fusion stage gave
75.0%, no better than attention.

---

## Limitations

The test set contains 40 subjects, so each is worth 2.5 percentage points and
differences below roughly 5 points cannot be resolved. Results are reported
over three seeds on a single fixed split; cross-validation would give a more
reliable estimate and is the main methodological improvement available.

The proposed model shows the widest seed spread of any configuration (72.5–85.0%),
against ±1.4% for the baseline. The wider 384-dimensional readout gives the
classifier more freedom and correspondingly more sensitivity to
initialisation.

Whole-brain results are single-seed and should be read as indicative. The
Mamba encoder uses a pure-PyTorch parallel scan rather than the official fused
CUDA kernel, so reported times are specific to this implementation; FLOPs and
parameter counts are unaffected.

---

## Key findings

- **Mamba outperforms attention at matched depth and width.** Holding
  tokenisation fixed, the transformer is 15 points worse on MRI and 8 points
  worse on multimodal at comparable parameter count.
- **Region structure should not be averaged away.** Keeping six region
  summaries separate rather than mean-pooling them adds 5.8 points; adding
  per-region modality attention adds a further 5.8.
- **Fewer, anatomically-defined tokens work better than more.** Two ROIs
  (1,024 tokens) and six ROIs (3,072) both outperform whole-brain (32,768) at
  monotonically decreasing cost.
- **Multimodal fusion helps only where the region carries amyloid signal.**
  Hippocampus +5.8 and cerebral WM +3.3 over their better unimodal arm;
  cerebellar WM, the PET reference region, −0.9.
- **MedicalNet pretraining does not transfer.** At matched architecture, the
  randomly initialised ResNet-10 is better on both unimodal settings.
- **Attention magnitude and predictive value diverge.** Cross-modal attention
  ranks cerebral WM highest and hippocampus lowest — bilaterally symmetric and
  stable across seeds — while region-pair accuracy favours hippocampus.
- **The proposed model reaches 78.3% at 0.47 GFLOPs**, within 4.6 points of a
  published baseline requiring 54 pretrained convolutional networks.

---

## References

[1] L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han, "On the variance of the adaptive learning rate and beyond," in *Proc. ICLR*, 2020.

[2] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.

[3] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[4] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning requires rethinking generalization," in *Proc. ICLR*, 2017.

[5] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929–1958, 2014.

[6] L. Prechelt, "Early stopping—But when?," in *Neural Networks: Tricks of the Trade*, G. B. Orr and K.-R. Müller, Eds. Berlin, Germany: Springer, 1998, pp. 55–69.

[7] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, "Vision Mamba: Efficient visual representation learning with bidirectional state space model," arXiv:2401.09417, 2024.

[8] L. Henschel, S. Conjeti, S. Estrada, K. Diers, B. Fischl, and M. Reuter, "FastSurfer—A fast and accurate deep learning based neuroimaging pipeline," *NeuroImage*, vol. 219, Art. no. 117012, 2020.

[9] D. Picard, "Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision," arXiv:2109.08203, 2021.

[10] Y. Li, R. Buchert, B. Schmitz-Koep, T. Grimmer, B. Ommer, D. M. Hedderich, I. Yakushev, and C. Wachinger, "Diffusion bridge networks simulate clinical-grade PET from MRI for dementia diagnostics," arXiv:2510.15556, 2025.

[11] F. Pérez-García, R. Sparks, and S. Ourselin, "TorchIO: A Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning," *Comput. Methods Programs Biomed.*, vol. 208, Art. no. 106236, 2021.

[12] J. Vo, N. Sharif, and G. M. Hassan, "MNA-net: Multimodal neuroimaging attention-based architecture for cognitive decline prediction," in *Predictive Intelligence in Medicine (PRIME)*, LNCS 15155, Springer, 2025, pp. 86–98.

[13] A. Torres--Leguet, "mamba.py: A simple, hackable and efficient Mamba implementation in pure PyTorch and MLX," 2024. [Online]. Available: https://github.com/alxndrTL/mamba.py
