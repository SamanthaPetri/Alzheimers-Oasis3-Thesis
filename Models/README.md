# Models

Vision Mamba architectures for 10-year CN → MCI/AD conversion prediction.
All notebooks read the caches produced by `Data Processing/`.

**Cohort**: 200 subjects, 120 / 40 / 40 split at `random_state=42`, stratified.
**Seeds**: `[1, 7, 123]`, reported as mean ± sample standard deviation [1].
**Hardware**: RTX 3070 Ti.

Every notebook shares the same structure: cells 0–5 are identical across all
of them (imports, config, encoder, models, data, training loop), followed by
one cell per seed and a summary.

---

## Architecture

The baseline and proposed models share a common front end. Six anatomical ROIs
are each cut into 8³ patches by a single `Conv3d`, giving 512 tokens per region
and 3,072 per modality. Factorised positional embeddings: depth, height,
width, plus a per-ROI embedding are added, scaled by 0.02 at initialisation
[2] to prevent positional signal dominating before the network has learned
anything from the data. An occupancy mask excludes patches containing no
tissue. Tokens then pass to a bidirectional `VMamba` encoder [3].

In these models there is no convolutional feature extractor and no pretrained
weights: the patch embedding is a single strided convolution, so all
representation learning happens in the state-space encoder. The CNN ablations
replace this front end and are described below.

The two models differ in what happens after the encoder.

**Baseline:** All 3,072 tokens are mean-pooled over valid positions into a
single 32-dimensional vector per modality. For multimodal input, two
independent branches produce one vector each, concatenated to 64 before the
classifier.

**Proposed model - per-region modality attention:** Tokens are pooled *within*
each of the six regions, giving six 32-dimensional summaries per modality. At
each region, that region's MRI and PET summaries are stacked and passed through
a shared multi-head attention layer, producing six fused 64-dimensional
vectors. These are concatenated to 384 for classification. This follows
MNA-net's fusion design [9], applied at anatomical rather than positional
granularity. The mechanism requires both modalities and has no unimodal form.

`d_model=32`, `n_layers=2`, `d_state=16`, dropout 0.4 [4], AdamW at lr 1e-4
with weight decay 1e-3 [5], `ReduceLROnPlateau`, early stopping on validation
loss [6] with a minimum-epoch floor.

`d_model=32` was chosen to curb overfitting, since the factorised positional
embedding tables account for a substantial share of total parameters and scale
directly with `d_model` — a concern given the 120-subject training set [7].

---

## Notebooks

| Notebook | Variant |
|---|---|
| `final_mamba_v7_perregion_att.ipynb` | Per-region modality attention (proposed model) |
| `final_mamba_v7_perregion_concat.ipynb` | Per-region concatenation (control) |
| `final_mamba_v7.ipynb` | Baseline ROI Vision Mamba |
| `final_mamba_v7_indv_rois.ipynb` | Region-pair ablation |
| `final_mamba_v7_trans.ipynb` | Transformer encoder |
| `final_mamba_v7_CNN_Untrain.ipynb` | ResNet-10 tokenisation, random initialisation |
| `final_mamba_v7_CNN_Pretrain.ipynb` | ResNet-10 tokenisation, MedicalNet pretrained |
| `final_mamba_v7_attention.ipynb` | Cross-modal attention |
| `final_mamba_v7_wholebrain.ipynb` | Whole-brain Vision Mamba |
| `mnanet_comparison.ipynb` | MNA-net replication and Mamba fusion |

### `final_mamba_v7_perregion_att.ipynb`

The proposed model, as described above. One attention layer is reused across
all six regions, following MNA-net [9].

### `final_mamba_v7_perregion_concat.ipynb`

Control for the proposed model. Identical up to the region summaries and
identical in readout width (6 × 64 = 384 into the classifier), but the MRI and
PET summaries at each region are concatenated directly rather than passed
through attention.

This separates the two changes the proposed model makes relative to the
baseline: keeping the six region summaries separate rather than averaging them,
and adding the attention layer. Each contributes roughly half the total gain.

### `final_mamba_v7.ipynb`

The baseline, and the only ROI configuration evaluated across all three
modality settings with the standard mean-pooled readout.

### `final_mamba_v7_indv_rois.ipynb`

Each bilateral pair trained in isolation — hippocampus (ROI indices 0–1),
cerebellar WM (2–3), cerebral WM (4–5). Two ROIs give 1,024 tokens instead of
3,072. Nine conditions, three regions × three modalities, three seeds each.

The dataset class loads the full 6-ROI array and slices the requested pair, so
no separate caches are required.

### `final_mamba_v7_trans.ipynb`

Replaces `VimEncoder` with `nn.TransformerEncoder` at matched depth, width and
head count, holding the patch tokenisation fixed. This isolates the sequence
model as the only variable. Attention is O(n²) in sequence length against
Mamba's linear scan [3], so the comparison is unfavourable on cost as well as
accuracy.

### `final_mamba_v7_CNN_Untrain.ipynb` and `final_mamba_v7_CNN_Pretrain.ipynb`

Both replace the patch embedding with MedicalNet's ResNet-10 trunk [8] up to
`layer4`, producing one token per ROI — 6 tokens rather than 3,072. The two
notebooks are architecturally identical; the only difference is whether the
pretrained weights are loaded or the trunk is randomly initialised, making this
a clean test of pretraining with architecture and capacity matched.

The pretrained variant requires the MedicalNet repository on the Python path
and `resnet_10_23dataset.pth`.

Both change the tokenisation as well as adding a convolutional front end, so
neither isolates the convolutional encoder alone.

### `final_mamba_v7_attention.ipynb`

Cross-modal attention: MRI and PET token sequences attend to each other before
pooling, matching MNA-net's fusion order [9]. Multimodal only — the mechanism
requires both modalities present.

Region weights are extracted as the mean attention each **key** token
*received*, averaged over all queries, then aggregated by region. Averaging
over the query dimension matters: the key dimension is normalised to sum to 1
by softmax and so cannot express preference, while the query-averaged quantity
can.

Memory note: this materialises a 3,072 × 3,072 attention matrix per head, so
`BATCH_SIZE = 1` may be required.

### `final_mamba_v7_wholebrain.ipynb`

Substitutes native-resolution 256³ volumes (32,768 tokens) for the six 64³ ROI
crops (3,072 tokens), with everything downstream unchanged. This establishes
the cost of whole-brain input relative to region-based input and justifies the
ROI approach, rather than serving as a competitive configuration.

The proposed model has no whole-brain equivalent: per-region fusion operates on
anatomical regions, which a whole-brain volume does not define.

Single seed. At roughly 300–570 minutes per run the three-seed protocol was not
feasible, and the figures should be read as indicative.

### `mnanet_comparison.ipynb`

Three parts, all on the MNI-space SynthStripped data:

1. **Replication** — Vo's frozen 27 + 27 patch encoders, his modality
   attention, and his stage-3 concatenation into a single dense layer [9].
2. **Mamba fusion** — identical stages 1 and 2, but the 27 patches become a
   sequence with positional embeddings and pass through Mamba rather than
   being concatenated.
3. **Pure Vision Mamba** — the baseline architecture on the same MNI volumes,
   no CNN and no pretrained weights.

Parts 1 and 2 use the 209-subject MNA-net cohort rather than the 200-subject
clean cohort, since a fair reproduction requires Vo's subject set.

---

## Reported metrics

Each seed records accuracy, sensitivity (TPR), specificity (TNR), parameter
count, GFLOPs from a single forward pass, per-sample inference latency
(CUDA-synchronised, averaged over 20 batches), total training wall-clock, and
the epoch at which validation loss bottomed out.

A warning fires if `best_epoch < 5`, flagging runs that never left
initialisation — a failure mode that occurred repeatedly in earlier versions
where early stopping triggered during the symmetry-breaking phase.

ROI models train from an in-memory cache while whole-brain volumes are streamed
from disk, so training times between the two are not directly comparable.
GFLOPs and inference latency are hardware-independent and should be preferred
for efficiency claims.

The Mamba encoder is the `mamba.py` implementation [10], which uses a
pure-PyTorch parallel scan rather than the fused CUDA kernel of the official
release. Reported training and inference times are therefore specific to this
implementation; FLOPs and parameter counts are unaffected.

---

## References

[1] D. Picard, "Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision," arXiv:2109.08203, 2021.

[2] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[3] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, "Vision Mamba: Efficient visual representation learning with bidirectional state space model," arXiv:2401.09417, 2024.

[4] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929–1958, 2014.

[5] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.

[6] L. Prechelt, "Early stopping — But when?," in *Neural Networks: Tricks of the Trade*, G. B. Orr and K.-R. Müller, Eds. Berlin, Germany: Springer, 1998, pp. 55–69.

[7] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning requires rethinking generalization," in *Proc. ICLR*, 2017.

[8] S. Chen, K. Ma, and Y. Zheng, "Med3D: Transfer learning for 3D medical image analysis," arXiv:1904.00625, 2019.

[9] J. Vo, N. Sharif, and G. M. Hassan, "MNA-net: Multimodal neuroimaging attention-based architecture for cognitive decline prediction," in *Predictive Intelligence in Medicine (PRIME 2024)*, LNCS 15155, Springer, 2025, pp. 86–98.

[10] A. Torres--Leguet, "mamba.py: A simple, hackable and efficient Mamba implementation in pure PyTorch and MLX," 2024. [Online]. Available: https://github.com/alxndrTL/mamba.py
