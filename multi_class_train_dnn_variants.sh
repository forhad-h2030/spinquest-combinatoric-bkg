#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=2:30:00
#SBATCH --array=0-2
#SBATCH -o train_dnn_variants_%A_%a.out
#SBATCH -e train_dnn_variants_%A_%a.err

# ── DNN variants: 3 architectures × 3 bootstrap seeds ────────────────────────
#
#  Variant A — flat wide   : hidden=512, layers=4, dropout=0.1  (no halving)
#  Variant B — deeper      : hidden=256, layers=6, dropout=0.2
#  Variant C — large flat  : hidden=1024, layers=4, dropout=0.2 (no halving)
#
#  Array mapping: task = variant_idx * N_BOOT + boot_idx
#    tasks 0-2  → variant A
#    tasks 3-5  → variant B
#    tasks 6-8  → variant C
#
#  Submit:
#    sbatch --array=0-8 multi_class_train_dnn_variants.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

export N_BOOT=1
export SPLIT_SEED="${SPLIT_SEED:-42}"
export OUT_ROOT="${OUT_ROOT:-outputs_dnn_variants_${SLURM_ARRAY_JOB_ID}}"
export EPOCHS="${EPOCHS:-300}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"
export LR_MIN="${LR_MIN:-1e-6}"
export STANDARDIZE="${STANDARDIZE:-1}"
SCRIPT="scripts/train_multiclass.py"

# ── variant definitions ───────────────────────────────────────────────────────
VARIANT_NAMES=("dnn_flat_w512_d01"   "dnn_deep_w256_d02"   "dnn_wide_w1024_d02")
HIDDEN_DIMS=(   512                    256                    1024               )
NUM_LAYERS=(    4                      6                      4                  )
DROPOUTS=(      0.1                    0.2                    0.2                )
FLATS=(         1                      0                      1                  )

# ── map array task → variant + boot ──────────────────────────────────────────
IDX="${SLURM_ARRAY_TASK_ID}"
VARIANT_IDX=$(( IDX / N_BOOT ))
BOOT_IDX=$(( IDX % N_BOOT ))
BOOT_SEED=$(( SPLIT_SEED + BOOT_IDX ))

RUN_NAME="${VARIANT_NAMES[$VARIANT_IDX]}"
export HIDDEN_DIM="${HIDDEN_DIMS[$VARIANT_IDX]}"
export NUM_LAYERS="${NUM_LAYERS[$VARIANT_IDX]}"
export DROPOUT="${DROPOUTS[$VARIANT_IDX]}"
export FLAT="${FLATS[$VARIANT_IDX]}"

echo "[INFO] job_id=${SLURM_JOB_ID} task_id=${IDX}"
echo "[INFO] variant=${RUN_NAME}  boot_idx=${BOOT_IDX}  boot_seed=${BOOT_SEED}"
echo "[INFO] hidden_dim=${HIDDEN_DIM}  num_layers=${NUM_LAYERS}  dropout=${DROPOUT}  flat=${FLAT}"
echo "[INFO] epochs=${EPOCHS}  lr=${LR}  lr_min=${LR_MIN}  standardize=${STANDARDIZE}"
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
  --env HIDDEN_DIM="$HIDDEN_DIM" \
  --env NUM_LAYERS="$NUM_LAYERS" \
  --env DROPOUT="$DROPOUT" \
  --env FLAT="$FLAT" \
  --env RUN_NAME="$RUN_NAME" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 "$SCRIPT"

echo "[INFO] done. results in: $RUN_DIR"
