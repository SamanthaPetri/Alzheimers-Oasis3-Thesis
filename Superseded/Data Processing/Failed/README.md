# ROI Image Attempt — PET Image Issue
## Issue
**PET registration artefact (sharp lines across images)**

Visual inspection revealed a sharp line splitting several PET ROI images, with signal on one side and empty voxels on the other.

Traced back through the registered and pre-registration volumes and confirmed as a **FLIRT registration failure**, not a data-loading bug: the original PIB PET scan (`128 × 128 × 109 × 26`, ~2.3mm voxels) was substantially smaller and lower-resolution than the target MRI space (`256³`, 1mm isotropic), and still 4D at the point registration ran. FLIRT defaulted to using only the first time frame with this mismatched field of view, placing the registered PET brain into only part of the output volume.

## Fix

PET time frames are averaged into a single 3D volume **before** FLIRT registration (not after), giving FLIRT a proper 3D target and resolving the field-of-view mismatch.

**References**

[1] Y. Li, R. Buchert, B. Schmitz-Koep, T. Grimmer, B. Ommer, D. M. Hedderich, 
    I. Yakushev, and C. Wachinger, "Diffusion Bridge Networks Simulate Clinical-grade 
    PET from MRI for Dementia Diagnostics," arXiv:2510.15556, 2025. [Online]. 
    Available: https://arxiv.org/abs/2510.15556

[2] "Processing and Analysing PET Brain Images," AAIC 2024 Neuroimaging Analysis 
    Workshop tutorial. [Online]. Available: 
    https://healthbioscienceideas.github.io/aaic2024-neuroimaging-workshop/pet-imaging.html
