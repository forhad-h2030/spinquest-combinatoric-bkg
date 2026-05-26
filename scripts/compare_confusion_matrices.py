#!/usr/bin/env python3
"""
Compare confusion matrices across multiple trained models.

Usage:
    python3 scripts/compare_confusion_matrices.py

Env vars:
    OUT_DIR     output directory for plots (default: plots/cm_compare)
    BUNDLES     colon-separated list of "label:path_to_npz"  (optional override)

If BUNDLES is not set, the script uses the hardcoded MODEL_BUNDLES list below.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "plots" / "cm_compare")))

CLASS_NAMES = ["J/psi", "psi(2S)", "DY", "Comb"]

# ── models to compare ────────────────────────────────────────────────────────
# Edit this list to add/remove models. Label is used in plot titles.
BASE = REPO_ROOT

MODEL_BUNDLES = [
    ("DNN ce_base 300ep",
     BASE / "outputs_followup_13480731/ce_base/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"),
    ("DNN ce_ls 300ep",
     BASE / "outputs_followup_13480731/ce_ls/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"),
    ("ResNet d512 b4",
     BASE / "outputs_resnet_13480717/resnet_d512_b4/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"),
    ("ResNet d512 b6",
     BASE / "outputs_resnet_13480717/resnet_d512_b6/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"),
    ("ResNet d1024 b4",
     BASE / "outputs_resnet_13480717/resnet_d1024_b4/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"),
    ("AdamW+OneCycle DNN",
     BASE / "outputs_adamw_13481852/adamw_onecycle_dnn/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"),
    ("AdamW+OneCycle ResNet",
     BASE / "outputs_adamw_13481852/adamw_onecycle_resnet/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"),
]

# Override via env var: "Label1:path1:Label2:path2" (colons separate alternating label/path)
if "BUNDLES" in os.environ:
    parts = os.environ["BUNDLES"].split(":")
    MODEL_BUNDLES = [(parts[i], Path(parts[i+1])) for i in range(0, len(parts)-1, 2)]

# ── load and compute ──────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

results = []
for label, path in MODEL_BUNDLES:
    if not Path(path).exists():
        print(f"[WARN] skipping {label}: {path} not found")
        continue
    bundle  = np.load(path, allow_pickle=True)
    y_test  = bundle["y_test"].astype(np.int64)
    y_pred  = bundle["y_pred"].astype(np.int64)
    y_proba = bundle["y_proba"].astype(np.float64)
    K = y_proba.shape[1]
    names = CLASS_NAMES[:K]

    cm = confusion_matrix(y_test, y_pred, labels=list(range(K)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acc = float((y_pred == y_test).mean())
    results.append({"label": label, "cm": cm, "cm_norm": cm_norm, "acc": acc,
                    "y_test": y_test, "y_pred": y_pred, "names": names})
    print(f"[INFO] {label:30s}  acc={acc:.4f}")

if not results:
    print("[ERROR] No bundles found. Check MODEL_BUNDLES paths.")
    sys.exit(1)

# ── print summary table ───────────────────────────────────────────────────────
print("\n── Per-class efficiency (diagonal of row-normalised CM) ──")
header = f"{'Model':30s}" + "".join(f"  {n:>10s}" for n in CLASS_NAMES) + "  {'Overall':>8s}"
print(header)
print("─" * len(header))
for r in results:
    diag = [r["cm_norm"][i, i] for i in range(len(r["names"]))]
    row  = f"{r['label']:30s}" + "".join(f"  {v:>10.3f}" for v in diag) + f"  {r['acc']:>8.4f}"
    print(row)

# ── confusion matrix heatmap grid ────────────────────────────────────────────
n = len(results)
ncols = min(n, 4)
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
axes = np.array(axes).flatten()

for ax, r in zip(axes, results):
    cm_n = r["cm_norm"]
    K    = len(r["names"])
    im   = ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cm_n[i,j] > 0.6 else "black")
    ax.set_xticks(range(K)); ax.set_xticklabels(r["names"], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(r["names"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(f"{r['label']}\nacc={r['acc']:.4f}", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes[len(results):]:
    ax.set_visible(False)

fig.suptitle("Confusion matrices — row normalised (balanced test set)", fontsize=11, fontweight="bold")
plt.tight_layout()
out_path = OUT_DIR / "confusion_matrix_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[INFO] grid plot saved → {out_path}")

# ── per-class precision / recall table ───────────────────────────────────────
print("\n── Classification report (best model: AdamW+OneCycle DNN) ──")
best = next((r for r in results if "AdamW+OneCycle DNN" in r["label"]), results[0])
print(classification_report(best["y_test"], best["y_pred"],
                             target_names=best["names"], digits=3))

# ── diagonal comparison bar chart ────────────────────────────────────────────
K      = len(CLASS_NAMES)
labels = [r["label"] for r in results]
diags  = np.array([[r["cm_norm"][i, i] for i in range(K)] for r in results])

fig2, axes2 = plt.subplots(1, K, figsize=(4 * K, 4.5), sharey=True)
colors = ["#2471A3", "#E74C3C", "#27AE60", "#8E44AD"]
for k, (ax, name) in enumerate(zip(axes2, CLASS_NAMES)):
    bars = ax.barh(range(len(labels)), diags[:, k], color=colors[k], alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("Efficiency", fontsize=9)
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.axvline(diags[:, k].max(), color="black", lw=0.8, ls="--", alpha=0.5)
    for bar, val in zip(bars, diags[:, k]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7)

fig2.suptitle("Per-class efficiency by model", fontsize=11, fontweight="bold")
plt.tight_layout()
out_path2 = OUT_DIR / "efficiency_by_class.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[INFO] efficiency chart saved → {out_path2}")
