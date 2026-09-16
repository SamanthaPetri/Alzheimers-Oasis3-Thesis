#!/bin/bash
# ---------------------------------------------------------------------------
# register_all_pet_v3_late.sh
#
# v3 PET pipeline: LATE-FRAME temporal averaging + rigid registration to MRI.
#   PET_averaged_late_v3/    3D late-frame means
#   PET_registered_v3/       registered volumes  (*_PIB_in_MRI_v3.nii.gz)
#                            transforms         (*_pet2mri_v3.mat)
#                            per-subject FLIRT logs
#   logs_v3/                 frame log, skip log, run summary
#
# Safe to interrupt and re-run: already-registered subjects are skipped.
# ---------------------------------------------------------------------------

set -u

# ---- config ---------------------------------------------------------------
FS_OUTPUT="/mnt/e/Oasis3/FastSurfer_output"
PET_DIR="/mnt/e/pet scans"
COHORT="/mnt/d/mamba_model/thesis_cohort_final.csv"

AVG_DIR="/mnt/e/Oasis3/PET_averaged_late_v3"
OUT_DIR="/mnt/e/Oasis3/PET_registered_v3"
LOG_DIR="/mnt/e/Oasis3/logs_v3"

N_LATE=9          # number of trailing frames to average (the 9x5min frames)
DOF=6             # rigid — within-subject PET->MRI

SUFFIX="_v3"      # appended to every output filename

# ---- setup ----------------------------------------------------------------
mkdir -p "$AVG_DIR" "$OUT_DIR" "$LOG_DIR"

FRAME_LOG="$LOG_DIR/frame_selection_v3.csv"
SKIP_LOG="$LOG_DIR/skipped_v3.txt"
RUN_LOG="$LOG_DIR/run_v3.txt"

# fresh headers only if starting clean
if [ ! -f "$FRAME_LOG" ]; then
    echo "subject_id,n_volumes,frames_used,start_index,mode" > "$FRAME_LOG"
fi
: > "$SKIP_LOG"

echo "=== PET v3 (late-frame averaging) started $(date) ===" | tee -a "$RUN_LOG"
echo "N_LATE=$N_LATE  DOF=$DOF" | tee -a "$RUN_LOG"
echo "avg  -> $AVG_DIR"  | tee -a "$RUN_LOG"
echo "reg  -> $OUT_DIR"  | tee -a "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

n_done=0
n_skip=0
n_short=0
n_already3d=0

# ---- main loop ------------------------------------------------------------
tail -n +2 "$COHORT" | while IFS=',' read -r subject_id baseline_day conversion_day pet_day mri_day mri_session outcome_label; do

    # strip any stray carriage returns from a Windows-authored CSV
    subject_id=$(echo "$subject_id" | tr -d '\r')
    mri_session=$(echo "$mri_session" | tr -d '\r')
    pet_day=$(echo "$pet_day" | tr -d '\r')

    [ -z "$subject_id" ] && continue

    pet_day_padded=$(printf "%04d" "$pet_day")

    PET_SESSION=$(ls "$PET_DIR" | grep "${subject_id}_PIB_d${pet_day_padded}" | head -1)
    if [ -z "$PET_SESSION" ]; then
        echo "SKIP $subject_id - no PIB session found" | tee -a "$SKIP_LOG"
        continue
    fi

    # handle both folder structures
    PET_NIFTI="$PET_DIR/$PET_SESSION/pet1/NIFTI"
    if [ ! -d "$PET_NIFTI" ]; then
        PET_NIFTI="$PET_DIR/$PET_SESSION/pet1"
    fi

    PET_FILE=$(ls "$PET_NIFTI"/*.nii.gz 2>/dev/null | head -1)
    if [ -z "$PET_FILE" ]; then
        echo "SKIP $subject_id - no NIfTI found" | tee -a "$SKIP_LOG"
        continue
    fi

    MRI_REF="$FS_OUTPUT/$mri_session/mri/orig.nii.gz"
    if [ ! -f "$MRI_REF" ]; then
        echo "SKIP $subject_id - no orig.nii.gz" | tee -a "$SKIP_LOG"
        continue
    fi

    # ---- v3 output names (distinct from v2) ----
    AVG_FILE="$AVG_DIR/${subject_id}_PIB_late${SUFFIX}.nii.gz"
    TMP_FILE="$AVG_DIR/${subject_id}_tmp_late${SUFFIX}.nii.gz"
    OUT_NII="$OUT_DIR/${subject_id}_PIB_in_MRI${SUFFIX}.nii.gz"
    OUT_MAT="$OUT_DIR/${subject_id}_pet2mri${SUFFIX}.mat"
    FLIRT_LOG="$OUT_DIR/${subject_id}_flirt${SUFFIX}.log"

    if [ -f "$OUT_NII" ]; then
        echo "SKIP $subject_id - already registered (v3)"
        continue
    fi

    # ---- late-frame temporal average ----
    NVOLS=$(fslnvols "$PET_FILE" 2>/dev/null)
    if [ -z "$NVOLS" ]; then
        echo "SKIP $subject_id - fslnvols failed on $PET_FILE" | tee -a "$SKIP_LOG"
        continue
    fi

    echo "Averaging $subject_id (${NVOLS} volumes)..."

    if [ "$NVOLS" -le 1 ]; then
        # already a 3D volume - nothing to average
        cp "$PET_FILE" "$AVG_FILE"
        echo "${subject_id},${NVOLS},${NVOLS},0,already_3d" >> "$FRAME_LOG"
        echo "  NOTE $subject_id: already 3D, copied as-is"

    elif [ "$NVOLS" -lt "$N_LATE" ]; then
        # fewer frames than the late window - average what exists, but flag it
        fslmaths "$PET_FILE" -Tmean "$AVG_FILE"
        echo "${subject_id},${NVOLS},${NVOLS},0,short_all_frames" >> "$FRAME_LOG"
        echo "  WARNING $subject_id: only ${NVOLS} frames (< ${N_LATE}), averaged all" | tee -a "$RUN_LOG"

    else
        START=$((NVOLS - N_LATE))
        fslroi "$PET_FILE" "$TMP_FILE" "$START" "$N_LATE"
        fslmaths "$TMP_FILE" -Tmean "$AVG_FILE"
        rm -f "$TMP_FILE"
        echo "${subject_id},${NVOLS},${N_LATE},${START},late_window" >> "$FRAME_LOG"
        echo "  $subject_id: used frames ${START}-$((NVOLS-1)) of ${NVOLS}"
    fi

    if [ ! -f "$AVG_FILE" ]; then
        echo "SKIP $subject_id - averaging produced no output" | tee -a "$SKIP_LOG"
        continue
    fi

    # ---- rigid registration to native MRI ----
    echo "Registering $subject_id..."
    flirt -in "$AVG_FILE" \
          -ref "$MRI_REF" \
          -out "$OUT_NII" \
          -omat "$OUT_MAT" \
          -dof "$DOF" 2>"$FLIRT_LOG"

    if [ -f "$OUT_NII" ]; then
        # registration succeeded - drop the empty log to keep the folder clean
        [ -s "$FLIRT_LOG" ] || rm -f "$FLIRT_LOG"
        echo "Done $subject_id"
    else
        echo "FAILED $subject_id - see $FLIRT_LOG" | tee -a "$SKIP_LOG"
    fi

done

# ---- summary --------------------------------------------------------------
echo "" | tee -a "$RUN_LOG"
echo "=== finished $(date) ===" | tee -a "$RUN_LOG"

N_REG=$(ls "$OUT_DIR"/*_PIB_in_MRI${SUFFIX}.nii.gz 2>/dev/null | wc -l)
N_AVG=$(ls "$AVG_DIR"/*_PIB_late${SUFFIX}.nii.gz 2>/dev/null | wc -l)
N_COHORT=$(( $(wc -l < "$COHORT") - 1 ))

echo "cohort subjects : $N_COHORT" | tee -a "$RUN_LOG"
echo "late-frame means: $N_AVG"    | tee -a "$RUN_LOG"
echo "registered (v3) : $N_REG"    | tee -a "$RUN_LOG"

if [ -s "$SKIP_LOG" ]; then
    echo "" | tee -a "$RUN_LOG"
    echo "skipped/failed:" | tee -a "$RUN_LOG"
    cat "$SKIP_LOG" | tee -a "$RUN_LOG"
fi

echo "" | tee -a "$RUN_LOG"
echo "frame log: $FRAME_LOG" | tee -a "$RUN_LOG"
echo "  (check the 'mode' column for short_all_frames / already_3d subjects)" | tee -a "$RUN_LOG"
