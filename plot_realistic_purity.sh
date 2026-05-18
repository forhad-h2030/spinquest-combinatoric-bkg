#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -o plot_realistic_purity_%j.out
#SBATCH -e plot_realistic_purity_%j.err

# ── Usage ─────────────────────────────────────────────────────────────────────
# sbatch --export=ALL,BUNDLE_PATH=<path/to/test_bundle.npz> plot_realistic_purity.sh
#
# Optional overrides:
#   OUT_DIR   output dir (default: same dir as bundle)
#   N_BOOT    bootstrap iterations (default: 100)
#   N_JPSI    experimental J/psi count  (default: 500)
#   N_PSIP    experimental psi(2S) count (default: 150)
#   N_DY      experimental DY count     (default: 500)
#   N_COMB    experimental comb count   (default: 2000)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

: "${BUNDLE_PATH:?BUNDLE_PATH is required. Pass via --export=ALL,BUNDLE_PATH=<path>}"

export OUT_DIR="${OUT_DIR:-$(dirname "$BUNDLE_PATH")}"
export N_BOOT="${N_BOOT:-100}"
export N_JPSI="${N_JPSI:-500}"
export N_PSIP="${N_PSIP:-150}"
export N_DY="${N_DY:-500}"
export N_COMB="${N_COMB:-2000}"

echo "[INFO] job_id     : ${SLURM_JOB_ID}"
echo "[INFO] bundle     : ${BUNDLE_PATH}"
echo "[INFO] out_dir    : ${OUT_DIR}"
echo "[INFO] N_BOOT     : ${N_BOOT}"
echo "[INFO] exp counts : J/psi=${N_JPSI} psi(2S)=${N_PSIP} DY=${N_DY} Comb=${N_COMB}"

mkdir -p "$OUT_DIR"
MPLCONFIGDIR="${OUT_DIR}/mplconfig"
mkdir -p "$MPLCONFIGDIR"

: "${CONTAINERDIR:?CONTAINERDIR is not set}"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"

apptainer exec --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR="$MPLCONFIGDIR" \
  --env BUNDLE_PATH="$BUNDLE_PATH" \
  --env OUT_DIR="$OUT_DIR" \
  --env N_BOOT="$N_BOOT" \
  --env N_JPSI="$N_JPSI" \
  --env N_PSIP="$N_PSIP" \
  --env N_DY="$N_DY" \
  --env N_COMB="$N_COMB" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 scripts/plot_realistic_purity.py

echo "[INFO] done. results in: ${OUT_DIR}"
