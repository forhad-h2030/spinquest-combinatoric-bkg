#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.core_train_multiclass_transformer import (
    TransformerTrainConfig,
    train_multiclass_transformer,
)
from utils.core_train_multiclass import (
    set_seed,
    load_npy,
    predict_proba,
    predict_class,
)
from utils.features import FEATURE_NAMES

# Input files — identical to train_multiclass.py
JPSI_FILE = "features_mc_jpsi_tuned1.npy"
PSIP_FILE = "features_mc_psip_tuned1.npy"
DY_FILE   = "features_mc_dy_target_pythia8_clsDNNmulti_v1.npy"
COMB_FILE = "features_mc_comb_target_gun_clsDNNmulti_v1.npy"

DATA_DIR = REPO_ROOT / "data" / "ml_input"
OUT_DIR  = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "models")))

CFG = TransformerTrainConfig(
    epochs          = int(os.environ.get("EPOCHS",            "300")),
    lr              = float(os.environ.get("LR",              "5e-4")),
    lr_min          = float(os.environ.get("LR_MIN",          "1e-6")),
    batch_size      = int(os.environ.get("BATCH_SIZE",        "1024")),
    seed            = int(os.environ.get("BOOT_SEED", os.environ.get("SPLIT_SEED", "42"))),
    standardize     = bool(int(os.environ.get("STANDARDIZE",  "1"))),
    d_model         = int(os.environ.get("D_MODEL",           "128")),
    n_heads         = int(os.environ.get("N_HEADS",           "4")),
    n_encoder_layers= int(os.environ.get("N_ENCODER_LAYERS",  "4")),
    dim_feedforward = int(os.environ.get("DIM_FF",            "512")),
    dropout_rate    = float(os.environ.get("DROPOUT",         "0.1")),
    loss_type       = os.environ.get("LOSS_TYPE",             "ce_ls"),
    focal_gamma     = float(os.environ.get("FOCAL_GAMMA",     "2.0")),
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", "0.05")),
)


def main():
    set_seed(CFG.seed)

    run_name    = "transformer_jpsi_psip_dy_comb"
    class_names = ["J/psi", "psi(2S)", "DY", "Combinatoric"]

    run_dir = OUT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {DATA_DIR}")
    X_jpsi = load_npy(DATA_DIR / JPSI_FILE)
    X_psip = load_npy(DATA_DIR / PSIP_FILE)
    X_dy   = load_npy(DATA_DIR / DY_FILE)
    X_comb = load_npy(DATA_DIR / COMB_FILE)

    print(f"[INFO] Data shapes:")
    print(f"  J/psi:        {X_jpsi.shape}")
    print(f"  psi(2S):      {X_psip.shape}")
    print(f"  DY:           {X_dy.shape}")
    print(f"  Combinatoric: {X_comb.shape}")

    Xs = [X_jpsi, X_psip, X_dy, X_comb]

    print(f"\n[INFO] Training Transformer multiclass classifier...")
    out = train_multiclass_transformer(
        Xs=Xs,
        cfg=CFG,
        out_dir=run_dir,
        run_name=run_name,
        class_names=class_names,
        use_class_weights=True,
    )

    y_proba = predict_proba(out["model"], out["X_test"], CFG.device)
    y_pred  = predict_class(out["model"], out["X_test"], CFG.device)

    import numpy as np
    np.savez_compressed(
        run_dir / f"{run_name}.test_bundle.npz",
        X_test=out["X_test"].astype(np.float32),
        y_test=np.asarray(out["y_test"]).astype(np.int64),
        y_proba=np.asarray(y_proba).astype(np.float32),
        y_pred=np.asarray(y_pred).astype(np.int64),
        class_names=class_names,
        run_name=run_name,
    )

    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Results saved to: {run_dir}")
    print(f"[INFO] Best checkpoint: {out['summary']['best_ckpt']}")
    print(f"[INFO] Test accuracy:   {out['summary']['test_metrics']['acc']:.4f}")
    print(f"[INFO] Test macro F1:   {out['summary']['test_metrics']['macro_f1']:.4f}")

    print(f"\n[INFO] Test Confusion Matrix:")
    cm = np.array(out["summary"]["test_metrics"]["confusion_matrix"])
    header = "True | " + " | ".join([f"{name:>12s}" for name in class_names])
    print("Predicted ->")
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:>4s} | " + " | ".join([f"{cm[i, j]:>12d}" for j in range(len(class_names))])
        print(row)


if __name__ == "__main__":
    main()
