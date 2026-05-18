#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=12:30:00
#SBATCH --array=0-0
#SBATCH -o train_multiclass_transformer_%A_%a.out
#SBATCH -e train_multiclass_transformer_%A_%a.err

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

export NBOOT="${NBOOT:-1}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
export OUT_ROOT="${OUT_ROOT:-outputs_multiclass_transformer_${SLURM_ARRAY_JOB_ID}}"
export STANDARDIZE="${STANDARDIZE:-0}"

export EPOCHS="${EPOCHS:-400}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"

# Transformer-specific hyperparameters
export D_MODEL="${D_MODEL:-64}"           # token embedding dimension
export N_HEADS="${N_HEADS:-4}"            # attention heads (D_MODEL must be divisible by N_HEADS)
export N_ENCODER_LAYERS="${N_ENCODER_LAYERS:-4}"  # number of encoder blocks
export DIM_FF="${DIM_FF:-256}"            # feedforward inner dimension
export DROPOUT="${DROPOUT:-0.1}"

export RUN_NAME="${RUN_NAME:-transformer_jpsi_psip_dy_comb}"
SCRIPT="${SCRIPT:-scripts/train_multiclass_transformer.py}"

mkdir -p "$OUT_ROOT"

IDX="${SLURM_ARRAY_TASK_ID}"
BOOT_IDX=$(( IDX % NBOOT ))
BOOT_SEED=$(( SPLIT_SEED + BOOT_IDX ))

echo "[INFO] job_id=${SLURM_JOB_ID} task_id=${IDX}"
echo "[INFO] boot_idx=${BOOT_IDX} boot_seed=${BOOT_SEED}"
echo "[INFO] script=${SCRIPT}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE} standardize=${STANDARDIZE}"
echo "[INFO] d_model=${D_MODEL} n_heads=${N_HEADS} n_encoder_layers=${N_ENCODER_LAYERS} dim_ff=${DIM_FF} dropout=${DROPOUT}"
echo "[INFO] out_root=${OUT_ROOT}"

BOOT_TAG="$(printf "boot_%03d" "${BOOT_IDX}")"
export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
mkdir -p "$RUN_DIR"

export OUT_DIR="$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

export BOOT_IDX BOOT_SEED

: "${CONTAINERDIR:?CONTAINERDIR is not set. On Rivanna, pytorch/2.7.0 usually sets it. If not, run: sbatch --export=ALL,CONTAINERDIR=/path/to/containers ...}"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"
if [[ ! -f "$SIF" ]]; then
  echo "[ERROR] container not found: $SIF"
  exit 2
fi

apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR="$MPLCONFIGDIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env BOOT_SEED="$BOOT_SEED" \
  --env SPLIT_SEED="$SPLIT_SEED" \
  --env SEED="$BOOT_SEED" \
  --env EPOCHS="$EPOCHS" \
  --env LR="$LR" \
  --env BATCH_SIZE="$BATCH_SIZE" \
  --env STANDARDIZE="$STANDARDIZE" \
  --env D_MODEL="$D_MODEL" \
  --env N_HEADS="$N_HEADS" \
  --env N_ENCODER_LAYERS="$N_ENCODER_LAYERS" \
  --env DIM_FF="$DIM_FF" \
  --env DROPOUT="$DROPOUT" \
  --env RUN_NAME="$RUN_NAME" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 "$SCRIPT"

echo "[INFO] Training complete! Results in: $RUN_DIR"
