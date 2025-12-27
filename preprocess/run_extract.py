#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # .../spinquest-combinatoric-bkg
sys.path.insert(0, str(REPO_ROOT))

from utils.extract_dimu_features import extract_features

def main():
    input_root = Path("../data/raw_input/MC_DY_Pythia8_Target_July18.root")
    tree_name  = "tree"

    X, feature_names, meta = extract_features(
        input_root,
        tree_name=tree_name,
        output_path=None,   # don't write npz inside
        mass_min=1.0,
        mass_max=6.0,
        verbose_every=20000,
    )

    out_npy = Path("../data/ml_input/features_dy.npy")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, X)

    print(f"✅ Saved: {out_npy}  shape={X.shape}")

if __name__ == "__main__":
    main()

