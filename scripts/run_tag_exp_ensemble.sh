#!/bin/bash
set -euo pipefail

# --- paths (edit if needed) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT="${INPUT:-$REPO_ROOT/data/raw_input/exp_tgt_data.root}"
TREE="${TREE:-tree}"
OUTPUT="${OUTPUT:-$REPO_ROOT/data/raw_input/exp_tagged_tgt_data.root}"

# point these at the outputs_array job you want to use
OUT_ARRAY="${OUT_ARRAY:-$REPO_ROOT/outputs_array_6876064}"

THR="${THR:-0.80}"
MASS_MIN="${MASS_MIN:-2.0}"
MASS_MAX="${MASS_MAX:-5.0}"

JPSI_DIR="${JPSI_DIR:-$OUT_ARRAY/jpsi_vs_nonjpsi}"
PSIP_DIR="${PSIP_DIR:-$OUT_ARRAY/psip_vs_nonpsip}"
DY_DIR="${DY_DIR:-$OUT_ARRAY/dy_comb_raw}"

module purge
module load gcc/11.4.0 openmpi/4.1.4
module load root/6.32.06
module load pytorch/2.7.0

# optional: if you need apptainer module for your site workflow, uncomment:
# module load apptainer/1.3.4

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] INPUT=$INPUT"
echo "[INFO] OUTPUT=$OUTPUT"
echo "[INFO] OUT_ARRAY=$OUT_ARRAY"
echo "[INFO] THR=$THR  MASS=[$MASS_MIN,$MASS_MAX]"
echo "[INFO] JPSI_DIR=$JPSI_DIR"
echo "[INFO] PSIP_DIR=$PSIP_DIR"
echo "[INFO] DY_DIR=$DY_DIR"
echo "[INFO] python=$(which python3)"
python3 -c "import ROOT; print('[INFO] ROOT version:', ROOT.gROOT.GetVersion())"

# --- run ---
python3 "$REPO_ROOT/scripts/tag_exp_with_ml_ensemble.py" \
  --input "$INPUT" \
  --tree "$TREE" \
  --output "$OUTPUT" \
  --mass-min "$MASS_MIN" \
  --mass-max "$MASS_MAX" \
  --thr "$THR" \
  --jpsi-dir "$JPSI_DIR" \
  --psip-dir "$PSIP_DIR" \
  --dy-dir   "$DY_DIR" \
  --write-std

echo "[DONE] wrote: $OUTPUT"

