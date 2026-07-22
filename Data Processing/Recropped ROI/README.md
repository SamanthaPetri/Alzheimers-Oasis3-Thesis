# ROI Extraction — Region-Appropriate Padding

A variant of the main 6-ROI extraction pipeline using region-specific crop padding, rather than a single fixed value applied uniformly.

## Motivation

The original pipeline applied +3 voxel padding to every ROI regardless of size. This is reasonable for larger structures but disproportionately affects smaller, more curved regions such as the hippocampus, where a tight crop requires substantial upsampling to reach 64³ and results in visible loss of detail. This effect was confirmed through direct visual and quantitative comparison (see the main model history for the pad=3 vs. pad=10 hippocampus comparison).

## Padding Applied

| Region | Padding (voxels) |
|---|---|
| Hippocampus (L/R) | 10 |
| Cerebellum White Matter (L/R) | 6 |
| Cerebral White Matter (L/R) | 3 (unchanged; already appropriately sized) |

## Pipeline

Identical to the main extraction otherwise: mask → region-specific padded crop → z-score normalise (ROI voxels only) → resize to 64×64×64 → cache. Augmentation uses the same fixed seeds (`[1, 101, 42]`) as the rest of the project for consistency.
