# Models

Vision Mamba models for 10-year CN to MCI/AD conversion prediction.
All notebooks read the caches produced by `Data Processing/`.

**Cohort**: 210 OASIS-3 subjects reduced to 200 after quality control,
120 / 40 / 40 stratified split.
**Seeds**: `[1, 7, 123]`, reported as mean ± sample standard deviation [1].

**Batch size**: 4. Whole-brain input uses 1 due to larger token size.

**Hardware**: RTX 3070 Ti.

Cells 0, 1, 4 and 5 hold the imports, configuration, data pipeline and
training loop and are near-identical throughout. Cells 2 and 3 hold the
encoder and the model variant, and are what differ between experiments. These
are followed by one cell per seed, a summary, and a GFLOPs cell.

---

## Architecture

The initial and proposed models share a common front end. Six anatomical ROIs
are each cut into 8³ patches by a single `Conv3d`, giving 512 tokens per region
and 3,072 per modality. Factorised positional embeddings (depth, height, width
and a per-ROI embedding) are added, scaled by 0.02 at initialisation. An
occupancy mask marks patches containing no tissue; these are zeroed and
excluded from pooling. Tokens then pass to a bidirectional Vision Mamba
encoder (`mambapy.vim.VMamba`) [2].

These models have no convolutional backbone and no pretrained weights: the
patch embedding is a single strided convolution. The CNN ablations replace this
front end and are described below.

The two models differ in what happens after the encoder.

**Initial model - global mean pooling:** All 3,072 tokens are mean-pooled over
valid positions into a single 32-dimensional vector per modality. For
multimodal input, two independent branches produce one vector each,
concatenated to 64 before the classifier.

**Proposed model - late per-region attention:** Tokens are pooled within each
of the six regions, giving six 32-dimensional summaries per modality. At each
region, that region's MRI and PET summaries are stacked and passed through a
shared multi-head attention layer, producing six fused 64-dimensional vectors.
These are concatenated to 384 for classification. This follows MNA-net's fusion
design [3], applied to anatomical regions rather than uniform patches. It
requires both modalities.

`d_model=32`, `n_layers=2`, `d_state=16`, dropout 0.4 [4], AdamW at lr 1e-4
with weight decay 1e-3 [5], `ReduceLROnPlateau`, early stopping on validation
loss [6] with a minimum-epoch floor of 25.

`d_model=32` was chosen to limit overfitting [7].

---

## Notebooks

| Notebook | Variant |
|---|---|
| `01_initial_model.ipynb` | Global mean pooling, MRI / PET / multimodal |
| `02_encoder_transformer.ipynb` | Transformer encoder in place of Vim |
| `03a_input_wholebrain.ipynb` | Whole-brain 256³ input |
| `03b_input_region_pairs.ipynb` | Each bilateral ROI pair in isolation |
| `04a_tokenisation_cnn_scratch.ipynb` | 3D CNN tokenisation, trained from scratch |
| `04b_tokenisation_resnet10_pretrained.ipynb` | ResNet-10 tokenisation, MedicalNet pretrained |
| `05a_fusion_early_crossmodal.ipynb` | Early cross-modal attention |
| `05b_fusion_late_concat.ipynb` | Late per-region concatenation |
| `05c_fusion_late_attention.ipynb` | Late per-region attention (proposed model) |
| `06_mnanet_comparison.ipynb` | MNA-net replication and Mamba fusion |

### `01_initial_model.ipynb`

The first ROI Vision Mamba model, evaluated on MRI, PET and multimodal input.

### `02_encoder_transformer.ipynb`

Replaces `VimEncoder` with `nn.TransformerEncoder` at matched depth, width and
sequence length; tokenisation, pooling and classification are unchanged. Two
layers, four heads, feed-forward dimension four times the model dimension,
pre-layer normalisation.

### `03a_input_wholebrain.ipynb`

Substitutes 256³ volumes (32,768 tokens) for the six 64³ ROI crops (3,072
tokens), with everything else unchanged. ROI identity embeddings are omitted,
as no separate regions are defined.

Single seed. At roughly 300–570 minutes per run the three-seed protocol was not
feasible.

### `03b_input_region_pairs.ipynb`

Each bilateral pair trained in isolation: hippocampus,
cerebellar WM, cerebral WM . Two ROIs give 1,024 tokens instead of
3,072. Three regions × three modalities.

### `04a_tokenisation_cnn_scratch.ipynb`

Replaces the patch embedding with a convolutional front end producing one
token per ROI: 6 tokens rather than 3,072. `04a` uses a three-layer `Conv3d`
stack with ReLU and adaptive average pooling, trained from scratch. `04b` uses
MedicalNet's ResNet-10 trunk [8] with `resnet_10_23dataset.pth`, features from
the final residual stage pooled and projected to 32 dimensions, fine-tuned
unfrozen at the same learning rate as the rest of the network.

### `04b_tokenisation_resnet10_pretrained.ipynb`
Same tokenisation change as `04a`, one token per ROI instead of 512 patches,
but the CNN starts from pretrained weights rather than random initialisation.
The tokeniser is MedicalNet's ResNet-10 trunk [8], trained by Tencent on 23
public 3D medical segmentation datasets covering several modalities and
structures, including brain and hippocampal MRI but no PET.

### `05a_fusion_early_crossmodal.ipynb`

MRI and PET token sequences attend to each other before pooling.
All 3,072 tokens from each modality pass through
two four-head attention layers; in one direction PET tokens weight the MRI
tokens, in the other MRI tokens weight the PET tokens. The resulting sequences
are mean-pooled and concatenated to 64. Multimodal only.

Region weights are extracted as the attention each key token received, averaged
over all queries and then by region.

### `05b_fusion_late_concat.ipynb`

Identical to the proposed model, but with no attention layer: MRI and PET
summaries at each region are concatenated directly. The two models differ by
4,224 parameters, which is the attention layer.

### `05c_fusion_late_attention.ipynb`

The proposed model, as described above. One attention layer is shared across
all six regions.

### `06_mnanet_comparison.ipynb`

Three parts, all on the MNI-space SynthStripped data:

1. **Replication**: Vo et al. replication [3].
2. **Mamba fusion**: the 27 patches become a sequence with positional
   embeddings and pass through Mamba rather than being concatenated.
3. **Pure Vision Mamba**: Vim on the same MNI volumes.

These use the 209-subject MNA-net cohort rather than the 200-subject clean
cohort, to match Vo et al.'s subject set. Single seed.

---

## Reported metrics

Each seed records accuracy, sensitivity (TPR), specificity (TNR), parameter
count, per-sample inference latency, total training wall-clock, and the epoch
at which validation loss reached its minimum. A warning fires if
`best_epoch < 5`.

ROI models train from an in-memory cache while whole-brain volumes are streamed
from disk, so training times between the two are not directly comparable.
GFLOPs and parameter counts are hardware-independent and are used for
efficiency comparisons; latency and training time depend on hardware.

The Mamba encoder is the `mamba.py` implementation [9], which uses a
pure-PyTorch parallel scan rather than the fused CUDA kernel of the official
release. Reported training and inference times are therefore specific to this
implementation; FLOPs and parameter counts are unaffected.

### GFLOPs

Each notebook ends with a GFLOPs cell that adds up the arithmetic in one
forward pass, for a single subject. It counts three things:

- the convolutions and the linear layers;
- the attention, where there is any;
- the Mamba scan.

| Configuration | Unimodal | Multimodal |
|---|---|---|
| Global mean pooling (initial model) | 0.34 | 0.68 |
| Transformer encoder | 2.67 | 5.34 |
| Whole-brain input | 3.62 | 7.25 |
| Single ROI pair | 0.11 | 0.23 |
| 3D CNN tokenisation | 0.20 | 0.41 |
| Pretrained ResNet-10 tokenisation | 106.2 | 212.4 |
| Early cross-modal attention | N/A | 3.15 |
| Late per-region concatenation | N/A | 0.68 |
| Late per-region attention (proposed) | N/A | 0.68 |

---

## References

[1] D. Picard, "Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision," arXiv:2109.08203, 2021.

[2] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, "Vision Mamba: Efficient visual representation learning with bidirectional state space model," arXiv:2401.09417, 2024.

[3] J. Vo, N. Sharif, and G. M. Hassan, "MNA-net: Multimodal neuroimaging attention-based architecture for cognitive decline prediction," in *Predictive Intelligence in Medicine (PRIME 2024)*, LNCS 15155, Springer, 2025, pp. 86–98.

[4] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929–1958, 2014.

[5] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.

[6] L. Prechelt, "Early stopping—But when?," in *Neural Networks: Tricks of the Trade*, G. B. Orr and K.-R. Müller, Eds. Berlin, Germany: Springer, 1998, pp. 55–69.

[7] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning requires rethinking generalization," in *Proc. ICLR*, 2017.

[8] S. Chen, K. Ma, and Y. Zheng, "Med3D: Transfer learning for 3D medical image analysis," arXiv:1904.00625, 2019.

[9] A. Torres-Leguet, "mamba.py: A simple, hackable and efficient Mamba implementation in pure PyTorch and MLX," 2024. [Online]. Available: https://github.com/alxndrTL/mamba.py
