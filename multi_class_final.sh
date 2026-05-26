#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# ── Final production run: AdamW+OneCycleLR DNN × 3 seeds ─────────────────────
#
#  Best model from architecture search:
#    Flat DNN 512×4, CE+LS (ε=0.05), AdamW (wd=1e-2), OneCycleLR
#    (max_lr=5e-4, 10% warmup, cosine decay, stepped per batch)
#
#  Input: final-tuned ROOT-derived .npy files in data/ml_input_final/
#    (generate first with: sbatch extract_final_features.sh)
#
#  Output: outputs_final_<jobid>/adamw_onecycle_dnn/boot_00{0,1,2}/
#
#  Submit (after extraction job completes):
#    sbatch multi_class_final.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

# ── Three distinct seeds for independent bootstrap realisations ───────────────
SEEDS=(42 123 456)
IDX="${SLURM_ARRAY_TASK_ID}"
BOOT_SEED="${SEEDS[$IDX]}"
BOOT_TAG=$(printf "boot_%03d" "$IDX")

# ── Model config (frozen from architecture search) ────────────────────────────
export OUT_ROOT="${OUT_ROOT:-outputs_final_${SLURM_ARRAY_JOB_ID}}"
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
export MODEL_TYPE="dnn"
export HIDDEN_DIM="512"
export NUM_LAYERS="4"

# ── Data: final-tuned .npy files ──────────────────────────────────────────────
export DATA_DIR="${SLURM_SUBMIT_DIR}/data/ml_input_final"
export JPSI_FILE="features_mc_jpsi_tuned_final.npy"
export PSIP_FILE="features_mc_psip_tuned_final.npy"
export DY_FILE="features_mc_dy_tuned_final.npy"
export COMB_FILE="features_mc_comb_tuned_final.npy"

RUN_NAME="adamw_onecycle_dnn"
SCRIPT="scripts/train_multiclass.py"

export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
mkdir -p "$RUN_DIR"
export OUT_DIR="$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Redirect all output into the model directory
exec 1>"$RUN_DIR/slurm_${SLURM_ARRAY_JOB_ID}_${IDX}.out" \
     2>"$RUN_DIR/slurm_${SLURM_ARRAY_JOB_ID}_${IDX}.err"

export SPLIT_SEED="$BOOT_SEED"
export BOOT_SEED="$BOOT_SEED"

echo "[INFO] job_id=${SLURM_JOB_ID}  task=${IDX}  seed=${BOOT_SEED}  tag=${BOOT_TAG}"
echo "[INFO] model=DNN flat 512x4  optimizer=adamw  scheduler=onecycle  loss=ce_ls"
echo "[INFO] data_dir=${DATA_DIR}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] epochs=${EPOCHS}  lr=${LR}  batch=${BATCH_SIZE}"

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
  --env DATA_DIR="$DATA_DIR" \
  --env JPSI_FILE="$JPSI_FILE" \
  --env PSIP_FILE="$PSIP_FILE" \
  --env DY_FILE="$DY_FILE" \
  --env COMB_FILE="$COMB_FILE" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 "$SCRIPT"

echo "[INFO] done. results in: $RUN_DIR"
