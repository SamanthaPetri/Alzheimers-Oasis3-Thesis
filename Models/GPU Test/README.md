# GPU Memory Sanity Check

Check confirming the Vision Mamba architecture fits within available GPU 
memory before committing to full training runs. Not part of the training pipeline.

## What It Does

Measures peak GPU memory after one forward + backward pass at three scales:

1. Small sanity check — `d_model=32`, `n_layers=1`, batch size 2
2. Larger configuration — `d_model=64`, `n_layers=2`, batch size 8
3. Multimodal at the larger configuration (MRI + PET branches) — same settings, batch size 8

Multimodal is tested explicitly since running two branches at once is where OOM errors are most likely.

## Results (RTX 3070 Ti, 8 GB VRAM)

| Configuration | Peak memory |
|---|---|
| `d_model=32`, sanity check | 0.31 GB |
| `d_model=64`, single modality | 3.76 GB |
| `d_model=64`, multimodal | 6.73 GB |

## Note

`d_model=32`: used in v2–v4 (and all `v4_*` variants) to curb overfitting from an oversized 
positional embedding table. Kept as default even after v3's factorised embeddings independently 
fixed that same issue.

`d_model=64`: used in v1 (original architecture) and revisited in v5 to check if the 32 cap was 
still needed post-v3 — showed no improvement.

This notebook tests `d_model=64` for GPU headroom only; actual training used 32. It also predates 
the real `mambapy.vim.VMamba` encoder (v3+), so figures are a rough upper bound, not exact.
