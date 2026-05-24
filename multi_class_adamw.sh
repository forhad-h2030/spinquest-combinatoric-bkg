#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH -o adamw_%A_%a.out
#SBATCH -e adamw_%A_%a.err

# ── AdamW + OneCycleLR diagnostic: 2 variants × 1 seed = 2 jobs ──────────────
#
#  Variant 0  adamw_onecycle_dnn     DNN flat 512×4,  AdamW + OneCycleLR
#  Variant 1  adamw_onecycle_resnet  ResNet 512×4 blocks, AdamW + OneCycleLR
#
#  Both use ce_ls loss + class weights (sweep winners).
#  OneCycleLR: 10% warmup, max_lr=5e-4, cosine anneal to ~5e-8.
#
#  Submit:
#    sbatch multi_class_adamw.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

export SPLIT_SEED="${SPLIT_SEED:-42}"
export OUT_ROOT="${OUT_ROOT:-outputs_adamw_${SLURM_ARRAY_JOB_ID}}"
export EPOCHS="${EPOCHS:-300}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"
export LR_MIN="${LR_MIN:-1e-6}"
export STANDARDIZE="${STANDARDIZE:-1}"
export OPTIMIZER="adamw"
export SCHEDULER="onecycle"
export LOSS_TYPE="ce_ls"
export LABEL_SMOOTHING="0.05"
export FOCAL_GAMMA="2.0"
export DROPOUT="0.1"
export FLAT="1"
SCRIPT="scripts/train_multiclass.py"

VARIANT_NAMES=( "adamw_onecycle_dnn"  "adamw_onecycle_resnet" )
MODEL_TYPES=(   "dnn"                 "resnet"                )
HIDDEN_DIMS=(   512                   512                     )
NUM_LAYERS_=(   4                     4                       )

IDX="${SLURM_ARRAY_TASK_ID}"
RUN_NAME="${VARIANT_NAMES[$IDX]}"
export MODEL_TYPE="${MODEL_TYPES[$IDX]}"
export HIDDEN_DIM="${HIDDEN_DIMS[$IDX]}"
export NUM_LAYERS="${NUM_LAYERS_[$IDX]}"
export BOOT_SEED="$SPLIT_SEED"

echo "[INFO] job_id=${SLURM_JOB_ID} task_id=${IDX}"
echo "[INFO] variant=${RUN_NAME}  model=${MODEL_TYPE}  optimizer=${OPTIMIZER}  scheduler=${SCHEDULER}"
echo "[INFO] hidden_dim=${HIDDEN_DIM}  num_layers=${NUM_LAYERS}  dropout=${DROPOUT}"
echo "[INFO] loss=${LOSS_TYPE}  label_smoothing=${LABEL_SMOOTHING}"
echo "[INFO] epochs=${EPOCHS}  lr=${LR}  batch=${BATCH_SIZE}"
echo "[INFO] out_root=${OUT_ROOT}"

BOOT_TAG="boot_000"
export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
mkdir -p "$RUN_DIR"
export OUT_DIR="$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

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
  --env OPTIMIZER="$OPTIMIZER" \
  --env SCHEDULER="$SCHEDULER" \
  --env RUN_NAME="$RUN_NAME" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 "$SCRIPT"

echo "[INFO] done. results in: $RUN_DIR"
