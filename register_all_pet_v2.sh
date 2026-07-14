#!/bin/bash

FS_OUTPUT="/mnt/e/Oasis3/FastSurfer_output"
PET_DIR="/mnt/e/pet scans"
AVG_DIR="/mnt/e/Oasis3/PET_averaged"
OUT_DIR="/mnt/e/Oasis3/PET_registered_v2"
COHORT="/mnt/d/mamba_model/thesis_cohort_final.csv"

mkdir -p "$AVG_DIR"
mkdir -p "$OUT_DIR"

tail -n +2 "$COHORT" | while IFS=',' read -r subject_id baseline_day conversion_day pet_day mri_day mri_session outcome_label; do

    pet_day_padded=$(printf "%04d" "$pet_day")
    PET_SESSION=$(ls "$PET_DIR" | grep "${subject_id}_PIB_d${pet_day_padded}" | head -1)

    if [ -z "$PET_SESSION" ]; then
        echo "SKIP $subject_id — no PIB session found"
        continue
    fi

    # Handle both folder structures
    PET_NIFTI="$PET_DIR/$PET_SESSION/pet1/NIFTI"
    if [ ! -d "$PET_NIFTI" ]; then
        PET_NIFTI="$PET_DIR/$PET_SESSION/pet1"
    fi

    PET_FILE=$(ls "$PET_NIFTI"/*.nii.gz 2>/dev/null | head -1)
    if [ -z "$PET_FILE" ]; then
        echo "SKIP $subject_id — no NIfTI found"
        continue
    fi

    MRI_REF="$FS_OUTPUT/$mri_session/mri/orig.nii.gz"
    if [ ! -f "$MRI_REF" ]; then
        echo "SKIP $subject_id — no orig.nii.gz"
        continue
    fi

    OUT_NII="$OUT_DIR/${subject_id}_PIB_in_MRI.nii.gz"
    OUT_MAT="$OUT_DIR/${subject_id}_pet2mri.mat"
    AVG_FILE="$AVG_DIR/${subject_id}_PIB_averaged.nii.gz"

    if [ -f "$OUT_NII" ]; then
        echo "SKIP $subject_id — already registered"
        continue
    fi

    echo "Averaging $subject_id..."
    fslmaths "$PET_FILE" -Tmean "$AVG_FILE"

    echo "Registering $subject_id..."
    flirt -in "$AVG_FILE" \
          -ref "$MRI_REF" \
          -out "$OUT_NII" \
          -omat "$OUT_MAT" \
          -dof 6 2>/dev/null

    echo "Done $subject_id"
done

echo "All done!"