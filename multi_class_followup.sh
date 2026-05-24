#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH -o followup_%A_%a.out
#SBATCH -e followup_%A_%a.err

# ── Follow-up: sweep winners only, 300 epochs, 1 seed = 2 jobs ───────────────
#
#  Sweep result: ce_base and ce_ls tied at test_acc~0.848, focal variants worse.
#
#  Variant 0  ce_base   CE + class weights,     flat 512×4, drop=0.1
#  Variant 1  ce_ls     CE + label_smooth=0.05, flat 512×4, drop=0.1
#
#  Submit:
#    sbatch multi_class_followup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

export N_BOOT=1
export SPLIT_SEED="${SPLIT_SEED:-42}"
export OUT_ROOT="${OUT_ROOT:-outputs_followup_${SLURM_ARRAY_JOB_ID}}"
export EPOCHS="${EPOCHS:-300}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"
export LR_MIN="${LR_MIN:-1e-6}"
export STANDARDIZE="${STANDARDIZE:-1}"
SCRIPT="scripts/train_multiclass.py"

# ── variant definitions ───────────────────────────────────────────────────────
VARIANT_NAMES=( "ce_base"  "ce_ls"  )
LOSS_TYPES=(    "ce"       "ce_ls"  )
FOCAL_GAMMAS=(  "2.0"      "2.0"   )
LABEL_SMOOTHS=( "0.0"      "0.05"  )
HIDDEN_DIMS=(   512        512     )
NUM_LAYERS_=(   4          4       )
DROPOUTS=(      "0.1"      "0.1"   )
FLATS=(         1          1       )

# ── map array task → variant + boot ──────────────────────────────────────────
IDX="${SLURM_ARRAY_TASK_ID}"
VARIANT_IDX=$(( IDX / N_BOOT ))
BOOT_IDX=$(( IDX % N_BOOT ))
BOOT_SEED=$(( SPLIT_SEED + BOOT_IDX ))

RUN_NAME="${VARIANT_NAMES[$VARIANT_IDX]}"
export LOSS_TYPE="${LOSS_TYPES[$VARIANT_IDX]}"
export FOCAL_GAMMA="${FOCAL_GAMMAS[$VARIANT_IDX]}"
export LABEL_SMOOTHING="${LABEL_SMOOTHS[$VARIANT_IDX]}"
export HIDDEN_DIM="${HIDDEN_DIMS[$VARIANT_IDX]}"
export NUM_LAYERS="${NUM_LAYERS_[$VARIANT_IDX]}"
export DROPOUT="${DROPOUTS[$VARIANT_IDX]}"
export FLAT="${FLATS[$VARIANT_IDX]}"

echo "[INFO] job_id=${SLURM_JOB_ID} task_id=${IDX}"
echo "[INFO] variant=${RUN_NAME}  boot_idx=${BOOT_IDX}  boot_seed=${BOOT_SEED}"
echo "[INFO] loss=${LOSS_TYPE}  focal_gamma=${FOCAL_GAMMA}  label_smoothing=${LABEL_SMOOTHING}"
echo "[INFO] hidden_dim=${HIDDEN_DIM}  num_layers=${NUM_LAYERS}  dropout=${DROPOUT}  flat=${FLAT}"
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
