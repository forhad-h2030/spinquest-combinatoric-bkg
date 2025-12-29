#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --array=0-29
#SBATCH -o train_%A_%a.out
#SBATCH -e train_%A_%a.err
set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

# ---- knobs ----
export NBOOT="${NBOOT:-10}"                 # bootstrap replicas per run
export SPLIT_SEED="${SPLIT_SEED:-12345}"    # base split seed (will be offset per boot)
export OUT_ROOT="${OUT_ROOT:-outputs_array_${SLURM_ARRAY_JOB_ID}}"
export BASE_SEED="${BASE_SEED:-42}"
export STANDARDIZE="${STANDARDIZE:-0}"
export THRESHOLD="${THRESHOLD:-0.90}"
export EPOCHS=100
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

# map linear task id -> (run, boot)
RUN_IDX=$(( IDX / NBOOT ))
BOOT_IDX=$(( IDX % NBOOT ))

if (( RUN_IDX < 0 || RUN_IDX >= ${#RUNS[@]} )); then
  echo "[ERROR] RUN_IDX=${RUN_IDX} out of range. Check --array and NBOOT."
  exit 2
fi

SCRIPT="${SCRIPTS[$RUN_IDX]}"
RUN_NAME="${RUNS[$RUN_IDX]}"

# unique per-bootstrap seed (stable, reproducible)
BOOT_SEED=$(( SPLIT_SEED + 1000*RUN_IDX + BOOT_IDX ))

echo "[INFO] job_id=${SLURM_JOB_ID} task_id=${IDX}"
echo "[INFO] RUN_IDX=${RUN_IDX} BOOT_IDX=${BOOT_IDX}"
echo "[INFO] script=${SCRIPT}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] boot_seed=${BOOT_SEED}"
echo "[INFO] epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE} standardize=${STANDARDIZE}"
echo "[INFO] out_root=${OUT_ROOT}"

# keep run order, but put each bootstrap in its own folder
BOOT_TAG="$(printf "boot_%03d" "${BOOT_IDX}")"
export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
mkdir -p "$RUN_DIR"

export OUT_DIR="$RUN_DIR"

export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

# pass boot seed to python scripts (they must consume it)
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

# sbatch --export=ALL,NBOOT=2 submit_train_3.sh
