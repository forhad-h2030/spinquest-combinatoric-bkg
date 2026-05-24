#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o infer_transformer_%j.out
#SBATCH -e infer_transformer_%j.err

# ── Usage ────────────────────────────────────────────────────────────────────
# sbatch --export=ALL,CKPT_PATH=<path/to/.best.pth> infer_transformer.sh
#
# Optional overrides:
#   DATA_DIR=<path>   (default: repo/data/ml_input)
#   OUT_DIR=<path>    (default: same directory as CKPT_PATH)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

: "${CKPT_PATH:?CKPT_PATH is required. Pass via --export=ALL,CKPT_PATH=<path>}"

export DATA_DIR="${DATA_DIR:-data/ml_input}"
export OUT_DIR="${OUT_DIR:-$(dirname "$CKPT_PATH")}"

echo "[INFO] job_id   : ${SLURM_JOB_ID}"
echo "[INFO] ckpt     : ${CKPT_PATH}"
echo "[INFO] data_dir : ${DATA_DIR}"
echo "[INFO] out_dir  : ${OUT_DIR}"

mkdir -p "$OUT_DIR"

MPLCONFIGDIR="${OUT_DIR}/mplconfig"
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
  --env CKPT_PATH="$CKPT_PATH" \
  --env DATA_DIR="$DATA_DIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 scripts/infer_multiclass_transformer.py

echo "[INFO] done. results in: ${OUT_DIR}"
