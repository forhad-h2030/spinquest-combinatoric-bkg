#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # .../spinquest-combinatoric-bkg
sys.path.insert(0, str(REPO_ROOT))

from utils.extract_dimu_features import extract_features

def main():
    #input_root = Path("../data/raw_input/mc_raw_psip_compact_feb12_tunedI.root")
    #input_root = Path("../data/raw_input/mc_comb_muon_gun_march15.root")
    input_root = Path("../data/raw_input/mc_dy_target_pythia8_clsDNNmulti_v1.root")
    tree_name  = "tree"

    X, feature_names, meta = extract_features(
        input_root,
        tree_name=tree_name,
        output_path=None,   
        mass_min=2.0,
        mass_max=6.0,
        verbose_every=20000,
    )

    #out_npy = Path("../data/ml_input/mc_raw_jpsi_compact_feb12_tuned_itr2.npy")
    out_npy = Path("../data/ml_input/mc_dy_target_pythia8_clsDNNmulti_v1.npy")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, X)

    print(f"Saved: {out_npy}  shape={X.shape}")

if __name__ == "__main__":
    main()

