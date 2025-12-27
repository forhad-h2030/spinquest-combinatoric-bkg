#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # .../spinquest-combinatoric-bkg
sys.path.insert(0, str(REPO_ROOT))

from utils.extract_dimu_features import extract_features


def stem_no_suffix(p: Path) -> str:
    return p.name[:-len("".join(p.suffixes))] if p.suffixes else p.stem


def main():
    data_dir = (REPO_ROOT / "data" / "raw_input").resolve()
    out_dir  = (REPO_ROOT / "data" / "ml_input").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_name = "tree"

    root_files = sorted(data_dir.glob("*.root"))
    if not root_files:
        raise FileNotFoundError(f"No .root files found in: {data_dir}")

    print(f"[INFO] Found {len(root_files)} ROOT files in {data_dir}")

    for i, input_root in enumerate(root_files, start=1):
        tag = stem_no_suffix(input_root)
        out_npy = out_dir / f"features_{tag}.npy"

        print(f"\n[{i}/{len(root_files)}] Processing: {input_root.name}")
        X, feature_names, meta = extract_features(
            input_root,
            tree_name=tree_name,
            output_path=None,   
            mass_min=1.0,
            mass_max=6.0,
            verbose_every=20000,
        )

        np.save(out_npy, X)
        print(f"Saved: {out_npy}  shape={X.shape}")

    print("\n[DONE] All files processed.")


if __name__ == "__main__":
    main()

