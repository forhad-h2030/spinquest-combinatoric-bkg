#!/usr/bin/env python3
"""
Inference script for the multiclass Transformer checkpoint.

Loads a .best.pth checkpoint, reproduces the train/val/test split using the
saved seed, runs inference on the test set, and prints + saves metrics.

Usage (env vars):
  CKPT_PATH   path to .best.pth file  (required)
  DATA_DIR    directory with .npy data files  (default: repo/data/ml_input)
  OUT_DIR     directory to write results      (default: same dir as CKPT_PATH)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.core_train_multiclass_transformer import ParticleTransformerMulticlass
from utils.core_train_multiclass import (
    load_npy,
    equalize_classes,
    split_balanced_per_class_multiclass,
    standardize_fit_transform,
    eval_multiclass,
    predict_proba,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

JPSI_FILE = "features_mc_jpsi_tuned1.npy"
PSIP_FILE = "features_mc_psip_tuned1.npy"
DY_FILE   = "features_mc_dy_target_pythia8_clsDNNmulti_v1.npy"
COMB_FILE = "features_mc_comb_target_gun_clsDNNmulti_v1.npy"


def main():
    ckpt_path = Path(os.environ["CKPT_PATH"])
    data_dir  = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data" / "ml_input")))
    out_dir   = Path(os.environ.get("OUT_DIR",  str(ckpt_path.parent)))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] checkpoint : {ckpt_path}", flush=True)
    print(f"[INFO] data_dir   : {data_dir}",  flush=True)
    print(f"[INFO] out_dir    : {out_dir}",   flush=True)

    # ── load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict    = ckpt["cfg"]
    class_names = ckpt["class_names"]
    scaler      = ckpt.get("scaler")
    best_epoch  = ckpt.get("best_epoch", "?")
    K           = ckpt["num_classes"]

    print(f"[INFO] class_names : {class_names}", flush=True)
    print(f"[INFO] best_epoch  : {best_epoch}",  flush=True)
    print(f"[INFO] best_val    : {ckpt['best_val']:.6f}", flush=True)
    print(f"[INFO] cfg         : d_model={cfg_dict['d_model']} n_heads={cfg_dict['n_heads']} "
          f"n_layers={cfg_dict['n_encoder_layers']} dim_ff={cfg_dict['dim_feedforward']} "
          f"standardize={cfg_dict['standardize']}", flush=True)

    # ── rebuild model ─────────────────────────────────────────────────────────
    model = ParticleTransformerMulticlass(
        input_dim       = ckpt["input_dim"],
        num_classes     = K,
        d_model         = cfg_dict["d_model"],
        n_heads         = cfg_dict["n_heads"],
        n_encoder_layers= cfg_dict["n_encoder_layers"],
        dim_feedforward = cfg_dict["dim_feedforward"],
        dropout_rate    = cfg_dict["dropout_rate"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params: {n_params:,}", flush=True)

    # ── load & split data (reproduce exact test split) ────────────────────────
    print(f"[INFO] loading data ...", flush=True)
    X_jpsi = load_npy(data_dir / JPSI_FILE)
    X_psip = load_npy(data_dir / PSIP_FILE)
    X_dy   = load_npy(data_dir / DY_FILE)
    X_comb = load_npy(data_dir / COMB_FILE)
    Xs = [X_jpsi, X_psip, X_dy, X_comb]

    seed        = cfg_dict["seed"]
    train_frac  = cfg_dict["train_frac"]
    val_frac    = cfg_dict["val_frac"]
    test_frac   = cfg_dict["test_frac"]
    standardize = cfg_dict["standardize"]

    Xs_eq = equalize_classes(Xs, seed=seed)
    X_train, _, X_val, _, X_test, y_test = split_balanced_per_class_multiclass(
        Xs_eq, train_frac, val_frac, test_frac, seed
    )
    print(f"[INFO] test set: {len(X_test):,} samples", flush=True)

    if standardize:
        if scaler is not None:
            mu  = scaler["mean"]
            sig = scaler["std"]
            X_test = (X_test - mu) / sig
            print("[INFO] applied saved scaler", flush=True)
        else:
            # scaler not in checkpoint (old format) — refit on train split
            _, _, X_test, _ = standardize_fit_transform(X_train, X_val, X_test)
            print("[INFO] scaler not in checkpoint — refit on train split", flush=True)

    # ── inference ─────────────────────────────────────────────────────────────
    print("[INFO] running inference ...", flush=True)
    test_ds     = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    metrics = eval_multiclass(model, test_loader, "cpu", num_classes=K)

    # per-class accuracy from confusion matrix
    cm      = np.array(metrics["confusion_matrix"])
    per_acc = cm.diagonal() / cm.sum(axis=1)

    # macro one-vs-rest AUC
    y_proba = predict_proba(model, X_test, "cpu")
    try:
        auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    except Exception as e:
        auc = float("nan")
        print(f"[WARN] AUC failed: {e}", flush=True)

    # ── print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}", flush=True)
    print(f"  Test accuracy  : {metrics['acc']:.4f}", flush=True)
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}", flush=True)
    print(f"  Macro AUC (OvR): {auc:.4f}", flush=True)
    print(f"  Per-class accuracy:", flush=True)
    for name, acc in zip(class_names, per_acc):
        print(f"    {name:<14}: {acc:.4f}", flush=True)
    print(f"{'='*55}\n", flush=True)

    print("  Confusion matrix (rows=true, cols=pred):", flush=True)
    header = "              " + "  ".join(f"{n:>12s}" for n in class_names)
    print(header, flush=True)
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>12s}  " + "  ".join(f"{v:>12d}" for v in row), flush=True)

    # ── save results ──────────────────────────────────────────────────────────
    run_name = ckpt.get("run_name", ckpt_path.stem.replace(".best", ""))
    summary = {
        "run_name"    : run_name,
        "ckpt_path"   : str(ckpt_path),
        "best_epoch"  : best_epoch,
        "best_val_loss": float(ckpt["best_val"]),
        "test_acc"    : float(metrics["acc"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "test_macro_auc": auc,
        "per_class_acc": {n: float(a) for n, a in zip(class_names, per_acc)},
        "confusion_matrix": metrics["confusion_matrix"],
        "cfg"         : cfg_dict,
        "class_names" : class_names,
        "n_test"      : int(len(X_test)),
    }
    out_json = out_dir / f"{run_name}.infer_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[INFO] metrics saved → {out_json}", flush=True)

    np.savez_compressed(
        out_dir / f"{run_name}.test_bundle.npz",
        X_test     = X_test.astype(np.float32),
        y_test     = y_test.astype(np.int64),
        y_proba    = y_proba.astype(np.float32),
        y_pred     = y_proba.argmax(axis=1).astype(np.int64),
        class_names= np.array(class_names),
    )
    print(f"[INFO] test bundle saved → {out_dir / f'{run_name}.test_bundle.npz'}", flush=True)


if __name__ == "__main__":
    main()
