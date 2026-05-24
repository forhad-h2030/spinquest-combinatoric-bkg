#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH -o resnet_%A_%a.out
#SBATCH -e resnet_%A_%a.err

# ── ResNet diagnostic sweep: 4 architectures × 1 seed = 4 jobs, 300 epochs ───
#
#  1 seed only — diagnostic run to find the best architecture.
#  Once a winner is identified, re-run with 3 seeds for final results.
#
#  All variants use ce_ls loss (sweep winner) + use_class_weights=True.
#  num_layers = number of residual blocks (each block = 2 linear layers + skip).
#  Total linear layers = 1 (input proj) + num_blocks×2 + 1 (head).
#
#  Variant 0  resnet_d512_b4   hidden=512,  4 blocks → 10 linear layers
#  Variant 1  resnet_d512_b6   hidden=512,  6 blocks → 14 linear layers
#  Variant 2  resnet_d512_b8   hidden=512,  8 blocks → 18 linear layers
#  Variant 3  resnet_d1024_b4  hidden=1024, 4 blocks → 10 layers, wider
#
#  Submit:
#    sbatch multi_class_resnet.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

export N_BOOT=1
export SPLIT_SEED="${SPLIT_SEED:-42}"
export OUT_ROOT="${OUT_ROOT:-outputs_resnet_${SLURM_ARRAY_JOB_ID}}"
export EPOCHS="${EPOCHS:-300}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"
export LR_MIN="${LR_MIN:-1e-6}"
export STANDARDIZE="${STANDARDIZE:-1}"
export MODEL_TYPE="resnet"
export LOSS_TYPE="${LOSS_TYPE:-ce_ls}"
export FOCAL_GAMMA="2.0"
export LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
export FLAT="1"
SCRIPT="scripts/train_multiclass.py"

# ── variant definitions ───────────────────────────────────────────────────────
VARIANT_NAMES=( "resnet_d512_b4"  "resnet_d512_b6"  "resnet_d512_b8"  "resnet_d1024_b4" )
HIDDEN_DIMS=(    512               512               512               1024              )
NUM_BLOCKS=(     4                 6                 8                 4                 )
DROPOUTS=(       "0.1"             "0.1"             "0.1"             "0.1"             )

# ── map array task → variant + boot ──────────────────────────────────────────
IDX="${SLURM_ARRAY_TASK_ID}"
VARIANT_IDX=$(( IDX / N_BOOT ))
BOOT_IDX=$(( IDX % N_BOOT ))
BOOT_SEED=$(( SPLIT_SEED + BOOT_IDX ))

RUN_NAME="${VARIANT_NAMES[$VARIANT_IDX]}"
export HIDDEN_DIM="${HIDDEN_DIMS[$VARIANT_IDX]}"
export NUM_LAYERS="${NUM_BLOCKS[$VARIANT_IDX]}"
export DROPOUT="${DROPOUTS[$VARIANT_IDX]}"

echo "[INFO] job_id=${SLURM_JOB_ID} task_id=${IDX}"
echo "[INFO] variant=${RUN_NAME}  boot_idx=${BOOT_IDX}  boot_seed=${BOOT_SEED}"
echo "[INFO] model=${MODEL_TYPE}  hidden_dim=${HIDDEN_DIM}  num_blocks=${NUM_LAYERS}  dropout=${DROPOUT}"
echo "[INFO] loss=${LOSS_TYPE}  label_smoothing=${LABEL_SMOOTHING}"
echo "[INFO] epochs=${EPOCHS}  lr=${LR}  lr_min=${LR_MIN}  batch=${BATCH_SIZE}"
echo "[INFO] out_root=${OUT_ROOT}"

BOOT_TAG="$(printf "boot_%03d" "${BOOT_IDX}")"
export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
mkdir -p "$RUN_DIR"
export OUT_DIR="$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

export BOOT_IDX BOOT_SEED

: "${CONTAINERDIR:?CONTAINERDIR is not set}"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"
if [[ ! -f "$SIF" ]]; then
  echo "[ERROR] container not found: $SIF"
  exit 2
fi

apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR="$MPLCONFIGDIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env BOOT_SEED="$BOOT_SEED" \
  --env SPLIT_SEED="$SPLIT_SEED" \
  --env EPOCHS="$EPOCHS" \
  --env LR="$LR" \
  --env LR_MIN="$LR_MIN" \
  --env BATCH_SIZE="$BATCH_SIZE" \
  --env STANDARDIZE="$STANDARDIZE" \
  --env MODEL_TYPE="$MODEL_TYPE" \
  --env HIDDEN_DIM="$HIDDEN_DIM" \
  --env NUM_LAYERS="$NUM_LAYERS" \
  --env DROPOUT="$DROPOUT" \
  --env FLAT="$FLAT" \
  --env LOSS_TYPE="$LOSS_TYPE" \
  --env FOCAL_GAMMA="$FOCAL_GAMMA" \
  --env LABEL_SMOOTHING="$LABEL_SMOOTHING" \
  --env RUN_NAME="$RUN_NAME" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 "$SCRIPT"

echo "[INFO] done. results in: $RUN_DIR"
