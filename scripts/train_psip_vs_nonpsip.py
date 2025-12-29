#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import os


REPO_ROOT = Path(__file__).resolve().parents[1]  # .../spinquest-combinatoric-bkg
sys.path.insert(0, str(REPO_ROOT))

from utils.core_train_binary import TrainConfig, set_seed, load_npy, train_binary_task, predict_prob
from utils.plots_binary import plot_confusion_binary, plot_prob_hists, plot_overlays
from utils.features import FEATURE_NAMES


# POS = psi'
# NEG = non-psi'
POS_FILE = "features_compact_mc_psip_target_pythia8_acc.npy"
NEG_FILE = "features_compact_mc_non_psip_target_pythia8_acc.npy"

DATA_DIR = REPO_ROOT / "data" / "ml_input"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "models")))


CFG = TrainConfig(
    epochs=int(os.environ.get("EPOCHS", "3")),
    lr=float(os.environ.get("LR", "5e-4")),
    batch_size=int(os.environ.get("BATCH_SIZE", "1024")),
    seed=int(os.environ.get("BOOT_SEED", os.environ.get("SPLIT_SEED", "42"))),
    standardize=bool(int(os.environ.get("STANDARDIZE", "0"))),
)

THR = float(os.environ.get("THRESHOLD", "0.90"))

def main():
    set_seed(CFG.seed)

    run_name  = "psip_vs_nonpsip"
    pos_label = "PSIP"
    neg_label = "NONPSIP"

    run_dir = OUT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    X_pos = load_npy(DATA_DIR / POS_FILE)  # psip
    X_neg = load_npy(DATA_DIR / NEG_FILE)  # non-psip

    plot_overlays(
        X_pos, X_neg, FEATURE_NAMES,
        pos_label=pos_label,
        neg_label=neg_label,
        run_name=run_name,
        out_dir=run_dir,
        bins=100,
        density=True,
        fontsize=18,
        show_stats=True,
        legend_all=False,
        save=True,
        show=False,
    )

    out = train_binary_task(X_pos, X_neg, CFG, run_dir, run_name)
    y_prob = predict_prob(out["model"], out["X_test"], CFG.device)
    np.savez_compressed(
        run_dir / f"{run_name}.test_bundle.npz",
        X_test=out["X_test"].astype(np.float32),
        y_test=np.asarray(out["y_test"]).astype(np.int64),
        y_prob=np.asarray(y_prob).astype(np.float32),
        pos_label=pos_label,
        neg_label=neg_label,
        run_name=run_name,
        threshold=float(THR),
    )
    print(f"[INFO] saved test bundle: {run_dir / f'{run_name}.test_bundle.npz'}")

    plot_confusion_binary(
        out["y_test"], y_prob,
        threshold=THR,
        run_name=run_name,
        pos_label=pos_label,
        neg_label=neg_label,
        out_dir=run_dir,
        save=True,
        show=False,
    )

    plot_prob_hists(
        out["y_test"], y_prob,
        run_name=run_name,
        pos_label=pos_label,
        neg_label=neg_label,
        out_dir=run_dir,
        save=True,
        show=False,
    )


if __name__ == "__main__":
    main()

