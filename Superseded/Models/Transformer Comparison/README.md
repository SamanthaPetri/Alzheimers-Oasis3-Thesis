# Transformer Comparison (Not Part of Primary Results)

Notebooks swapping the Vision Mamba encoder for a standard multi-layer Transformer encoder, keeping everything else in the pipeline identical (patch embedding, ROI/position embeddings, masking, pooling, classifier, training protocol, seeds). Run to test whether the performance gap versus the MNA-Net baseline is specific to Mamba.

**Included variants**
- `Trans_comp` — full 6-region model, MRI/PET/multimodal (matches v4 structure)
- `Trans_comp_wholebrain` — whole-brain input, downsampled to match the 3,072-token ROI setup
- `Trans_comp_wholebrain_fullres` — whole-brain input, native resolution (32,768 tokens), no downsampling


## Implementation

Only the sequence-relating mechanism changed — `VimEncoder` replaced with PyTorch's built-in `torch.nn.TransformerEncoder` (multi-head self-attention + feed-forward, stacked in layers) [1].

## Results (3-seed mean ± std, Test Set)

| Configuration | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|---|---|---|
| 6-region MRI-only | 54.0% ± 3.6% | 58.7% ± 5.5% | 49.2% ± 7.1% |
| 6-region PET-only | 56.3% ± 5.5% | 71.4% ± 34.2% | 41.3% ± 38.5% |
| 6-region Multimodal | 54.0% ± 1.4% | 50.8% ± 14.4% | 57.1% ± 14.4% |
| Whole-brain MRI (token-matched) | 48.4% ± 1.4% | 52.4% ± 8.2% | 44.4% ± 7.3% |
| Whole-brain PET (token-matched) | 63.5% ± 1.4% | 58.7% ± 2.7% | 68.3% ± 2.7% |
| Whole-brain Multimodal (token-matched) | 60.3% ± 6.9% | 61.9% ± 8.2% | 58.7% ± 22.0% |
| Whole-brain MRI (native resolution) | TBD | TBD | TBD |
| Whole-brain PET (native resolution) | TBD | TBD | TBD |
| Whole-brain Multimodal (native resolution) | TBD | TBD | TBD |

## Result summary

The Transformer consistently underperformed Mamba, with substantially higher seed-to-seed variance and clear overfitting in training curves.

This is treated as supporting evidence that the gap to the MNA-Net baseline reflects a broader data-scarcity limitation affecting sequence-based architectures generally, not a weakness specific to Mamba.


**References**

[1] PyTorch `torch.nn.TransformerEncoder` documentation. [Online]. Available: 
    https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
