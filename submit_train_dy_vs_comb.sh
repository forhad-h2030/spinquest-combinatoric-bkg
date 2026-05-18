#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=18G
#SBATCH --time=03:00:00
#SBATCH -o train_dy_comb_%j.out
#SBATCH -e train_dy_comb_%j.err
set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

# ---- knobs ----
export NBOOT="${NBOOT:-1}"                 # bootstrap replicas (looped inside this job)
export SPLIT_SEED="${SPLIT_SEED:-12345}"   # base split seed
export OUT_ROOT="${OUT_ROOT:-outputs_${SLURM_JOB_ID}}"
export STANDARDIZE="${STANDARDIZE:-0}"
export THRESHOLD="${THRESHOLD:-0.90}"
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"

mkdir -p "$OUT_ROOT"

SCRIPT="scripts/train_dy_vs_comb.py"
RUN_NAME="dy_comb_raw"

echo "[INFO] job_id=${SLURM_JOB_ID}"
echo "[INFO] script=${SCRIPT}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] nboot=${NBOOT} split_seed=${SPLIT_SEED}"
echo "[INFO] epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE} standardize=${STANDARDIZE}"
echo "[INFO] out_root=${OUT_ROOT}"

for (( BOOT_IDX=0; BOOT_IDX<NBOOT; BOOT_IDX++ )); do
  BOOT_SEED=$(( SPLIT_SEED + 1000*0 + BOOT_IDX ))  # RUN_IDX is 0 since only one run

  echo "[INFO] ---- BOOT_IDX=${BOOT_IDX} BOOT_SEED=${BOOT_SEED} ----"

  BOOT_TAG="$(printf "boot_%03d" "${BOOT_IDX}")"
  export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
  mkdir -p "$RUN_DIR"

  export OUT_DIR="$RUN_DIR"

  export MPLCONFIGDIR="$RUN_DIR/mplconfig"
  mkdir -p "$MPLCONFIGDIR"

  export BOOT_IDX BOOT_SEED

  apptainer exec --nv --cleanenv \
    --env PYTHONNOUSERSITE=1 \
    --env MPLBACKEND=Agg \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --env OUT_DIR="$OUT_DIR" \
    --env BOOT_SEED="$BOOT_SEED" \
    --env SPLIT_SEED="$SPLIT_SEED" \
    --env EPOCHS="$EPOCHS" \
    --env LR="$LR" \
    --env BATCH_SIZE="$BATCH_SIZE" \
    --env STANDARDIZE="$STANDARDIZE" \
    --env THRESHOLD="$THRESHOLD" \
    --env QT_QPA_PLATFORM=offscreen \
    --env DISPLAY= \
    --env QT_PLUGIN_PATH= \
    --env QT_QPA_PLATFORM_PLUGIN_PATH= \
    --env XDG_RUNTIME_DIR=/tmp \
    "$CONTAINERDIR/pytorch-2.7.0.sif" \
    python3 "$SCRIPT"
done

# examples:
# sbatch submit_train_dy_vs_comb.sh
# sbatch --export=ALL,NBOOT=5 submit_train_dy_vs_comb.sh
