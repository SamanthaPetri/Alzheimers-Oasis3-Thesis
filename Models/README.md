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

**v7 — pure Vision Mamba.** No convolutional front end, no pretrained weights.
Six anatomical ROIs, each cut into 8³ patches by a single `Conv3d`, giving
3,072 tokens per modality. Factorised positional embeddings — depth, height,
width, plus a per-ROI embedding — scaled by 0.02 at initialisation [2] to
prevent positional signal dominating before the network has learned anything
from the data. An occupancy mask excludes patches containing no tissue. Tokens
pass to a bidirectional `VMamba` encoder [3], are mean-pooled over valid
positions, and classified.

Multimodal models run two independent branches and concatenate before the
classifier.

`d_model=32`, `n_layers=2`, `d_state=16`, dropout 0.4 [4], AdamW at lr 1e-4
with weight decay 1e-3 [5], `ReduceLROnPlateau`, early stopping on validation
loss [6] with a minimum-epoch floor.

`d_model=32` was chosen to curb overfitting: the positional embedding table
accounted for 63.6% of parameters at `d_model=64`, disproportionate for a
120-subject training set [7]. Revisiting it at 64 in an earlier version showed
no improvement.

---

## Notebooks

| Notebook | Variant |
|---|---|
| `final_mamba_v7.ipynb` | Baseline ROI Vision Mamba |
| `final_mamba_v7_indv_rois.ipynb` | Region-pair ablation |
| `final_mamba_v7_trans.ipynb` | Transformer encoder |
| `final_mamba_v7-CNN_Scratch.ipynb` | From-scratch CNN tokenisation |
| `final_mamba_v7_CNN_Pretrain.ipynb` | MedicalNet pretrained tokenisation |
| `final_mamba_v7_attention.ipynb` | Region-attention pooling |
| `final_mamba_v7_attention_1.ipynb` | Cross-modal attention |
| `final_mamba_v7_wholebrain.ipynb` | Whole-brain Vision Mamba |
| `final_mamba_v7_wholebrain_trans.ipynb` | Whole-brain transformer |
| `final_mamba_v7_wholebrain_CNN_Scratch.ipynb` | Whole-brain, from-scratch CNN |
| `final_mamba_v7_wholebrain_CNN_Pretrain.ipynb` | Whole-brain, MedicalNet |
| `mnanet_comparison.ipynb` | MNA-net replication and Mamba fusion |

The whole-brain notebooks mirror the ROI ones exactly, substituting the
native-resolution 256³ volumes (32,768 tokens) for the six 64³ ROI crops
(3,072 tokens); at roughly ten times the FLOPs and forty times the training
time, they serve as a cost comparison rather than a competitive configuration.

### `final_mamba_v7.ipynb`

The baseline. MRI-only, PET-only and multimodal.

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

### `final_mamba_v7-CNN_Scratch.ipynb`

Replaces the patch embedding with a shallow 3D CNN producing one token per ROI
— 6 tokens rather than 3,072. This changes the tokenisation as well as adding
a convolutional front end, so it is not a pure "CNN vs no CNN" comparison.

### `final_mamba_v7_CNN_Pretrain.ipynb`

Same structure, with MedicalNet's pretrained ResNet-10 trunk [8] up to
`layer4` as the encoder.

Requires the MedicalNet repository on the Python path and
`resnet_10_23dataset.pth`.

### `final_mamba_v7_attention.ipynb`

Cross-modal attention: MRI and PET token sequences attend to each other before
pooling, matching MNA-net's fusion order [9]. Multimodal only — the mechanism
requires both modalities present.

Region weights are extracted as the mean attention each **key** token
*received*, averaged over all queries, then aggregated by region. Averaging
over the query dimension matters: the key dimension is normalised to sum to 1
by softmax and so cannot express preference, while the query-averaged
quantity can.

Memory note: this materialises a 3,072 × 3,072 attention matrix per head, so
`BATCH_SIZE = 1` may be required.

### `mnanet_comparison.ipynb`

Three parts, all on the MNI-space SynthStripped data:

1. **Replication** — Vo's frozen 27 + 27 patch encoders, his modality
   attention, and his stage-3 concatenation into a single dense layer [9].
2. **Mamba fusion** — identical stages 1 and 2, but the 27 patches become a
   sequence with positional embeddings and pass through Mamba rather than
   being concatenated.
3. **Pure Vision Mamba** — the v7 architecture on the same MNI volumes, no CNN
   and no pretrained weights.

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

ROI models train from an in-memory cache while whole-brain volumes are
streamed from disk, so training times between the two are not directly
comparable. GFLOPs and inference latency are hardware-independent and should
be preferred for efficiency claims.

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
