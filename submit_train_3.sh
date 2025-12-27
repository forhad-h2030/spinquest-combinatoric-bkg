#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --array=0-2
#SBATCH -o train_%A_%a.out
#SBATCH -e train_%A_%a.err
set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

export SPLIT_SEED="${SPLIT_SEED:-12345}"
export OUT_ROOT="${OUT_ROOT:-outputs_array_${SLURM_ARRAY_JOB_ID}}"
export BASE_SEED="${BASE_SEED:-42}"
export STANDARDIZE="${STANDARDIZE:-0}"
export THRESHOLD="${THRESHOLD:-0.90}"
export EPOCHS="${EPOCHS:-5}"          # start with 5, later set to 200
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"

mkdir -p "$OUT_ROOT"

SCRIPTS=(
  "scripts/train_jpsi_vs_nonjpsi.py"
  "scripts/train_psip_vs_nonpsip.py"
  "scripts/train_dy_vs_comb.py"
)

RUNS=(
  "jpsi_vs_nonjpsi"
  "psip_vs_nonpsip"
  "dy_comb_raw"
)

IDX="${SLURM_ARRAY_TASK_ID}"
SCRIPT="${SCRIPTS[$IDX]}"
RUN_NAME="${RUNS[$IDX]}"

echo "[INFO] job_id=${SLURM_JOB_ID} task_id=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] script=${SCRIPT}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE} standardize=${STANDARDIZE}"
echo "[INFO] out_root=${OUT_ROOT}"

export RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

export OUT_DIR="$RUN_DIR"

export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR="$MPLCONFIGDIR" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$CONTAINERDIR/pytorch-2.7.0.sif" \
  python3 "$SCRIPT"

