#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # .../spinquest-combinatoric-bkg
sys.path.insert(0, str(REPO_ROOT))

from utils.extract_dimu_features import extract_features


def stem_no_suffix(p: Path) -> str:
    return p.name[:-len("".join(p.suffixes))] if p.suffixes else p.stem

MASS_RANGES = [
    (2.0, 4.0, "ml_input_multiclass_M_24"),
    (4.0, 6.0, "ml_input_multiclass_M_46"),
]


def main():
    #data_dir = (REPO_ROOT / "data" / "raw_input").resolve()
    data_dir = (REPO_ROOT / "data" / "raw_input_tuned_I").resolve()
    tree_name = "tree"

    # Pre-create all output directories
    out_dirs = {}
    for mass_min, mass_max, out_dir_name in MASS_RANGES:
        out_dir = (REPO_ROOT / "data" / out_dir_name).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[out_dir_name] = (mass_min, mass_max, out_dir)

    root_files = sorted(data_dir.glob("*.root"))
    if not root_files:
        raise FileNotFoundError(f"No .root files found in: {data_dir}")

    print(f"[INFO] Found {len(root_files)} ROOT files in {data_dir}")
    print(f"[INFO] Mass ranges: { [(mn, mx) for mn, mx, _ in MASS_RANGES] }")

    for i, input_root in enumerate(root_files, start=1):
        tag = stem_no_suffix(input_root)
        print(f"\n[{i}/{len(root_files)}] Reading: {input_root.name}")

        # Read full file once with no mass cut
        X, feature_names, meta = extract_features(
            input_root,
            tree_name=tree_name,
            output_path=None,
            mass_min=2.0,
            mass_max=6.0,
            verbose_every=20000,
        )

        mass_col = feature_names.index("rec_dimu_M")
        mass = X[:, mass_col]

        # Split and save into each output directory
        for mass_min, mass_max, out_dir in out_dirs.values():
            mask = (mass >= mass_min) & (mass < mass_max)
            X_cut = X[mask]
            out_npy = out_dir / f"features_{tag}.npy"
            np.save(out_npy, X_cut)
            print(f"  [{mass_min}–{mass_max} GeV]  shape={X_cut.shape}  →  {out_npy}")

    print("\n[DONE] All files processed.")


if __name__ == "__main__":
    main()
