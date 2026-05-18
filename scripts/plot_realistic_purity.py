#!/usr/bin/env python3
"""
Realistic confusion matrix and purity/F1 vs probability threshold plot.

Resamples the balanced MC test set to experimental proportions, then:
  1. Builds a confusion matrix at argmax threshold
  2. Plots Purity and F1 vs min-probability threshold for J/psi and psi(2S)
     with bootstrap uncertainty bands (matching the slide style)

Env vars:
  BUNDLE_PATH   path to test_bundle.npz            (required)
  OUT_DIR       output directory                   (default: same dir as bundle)
  N_BOOT        bootstrap iterations               (default: 100)
  N_JPSI        experimental J/psi count           (default: 500)
  N_PSIP        experimental psi(2S) count         (default: 150)
  N_DY          experimental DY count              (default: 500)
  N_COMB        experimental combinatoric count    (default: 2000)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── config ────────────────────────────────────────────────────────────────────
BUNDLE_PATH = Path(os.environ["BUNDLE_PATH"])
OUT_DIR     = Path(os.environ.get("OUT_DIR", str(BUNDLE_PATH.parent)))
N_BOOT      = int(os.environ.get("N_BOOT",  "100"))
N_JPSI      = int(os.environ.get("N_JPSI",  "500"))
N_PSIP      = int(os.environ.get("N_PSIP",  "150"))
N_DY        = int(os.environ.get("N_DY",    "500"))
N_COMB      = int(os.environ.get("N_COMB",  "2000"))

EXP_COUNTS  = [N_JPSI, N_PSIP, N_DY, N_COMB]
THRESHOLDS  = np.linspace(0.40, 0.95, 60)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load bundle ───────────────────────────────────────────────────────────────
print(f"[INFO] bundle : {BUNDLE_PATH}", flush=True)
bundle      = np.load(BUNDLE_PATH, allow_pickle=True)
y_proba     = bundle["y_proba"].astype(np.float64)   # (N, 4)
y_test      = bundle["y_test"].astype(np.int64)       # (N,)
class_names = list(bundle["class_names"])

K = len(class_names)
rng = np.random.default_rng(42)

# Indices for each true class in the balanced test set
class_idx = [np.where(y_test == c)[0] for c in range(K)]
print(f"[INFO] test set counts (balanced MC): {[len(i) for i in class_idx]}", flush=True)
print(f"[INFO] experimental counts: J/psi={N_JPSI} psi(2S)={N_PSIP} DY={N_DY} Comb={N_COMB}", flush=True)
print(f"[INFO] bootstrap N={N_BOOT}", flush=True)

# ── helper: sample one experimental realisation ───────────────────────────────
def sample_exp(rng):
    idxs, labels = [], []
    for c, (cidx, n) in enumerate(zip(class_idx, EXP_COUNTS)):
        chosen = rng.choice(cidx, size=n, replace=True)
        idxs.append(chosen)
        labels.append(np.full(n, c, dtype=np.int64))
    return np.concatenate(idxs), np.concatenate(labels)

# ── 1. Realistic confusion matrix (argmax, averaged over bootstraps) ──────────
print("[INFO] computing realistic confusion matrix ...", flush=True)
cm_acc = np.zeros((K, K), dtype=np.float64)

for _ in range(N_BOOT):
    sidx, slabels = sample_exp(rng)
    proba_s = y_proba[sidx]
    y_pred_s = proba_s.argmax(axis=1)
    for true_c in range(K):
        mask = slabels == true_c
        for pred_c in range(K):
            cm_acc[true_c, pred_c] += (y_pred_s[mask] == pred_c).sum()

cm_acc /= N_BOOT   # average counts per bootstrap
cm_norm = cm_acc / cm_acc.sum(axis=1, keepdims=True)  # row-normalised

# Print
print("\n  Realistic confusion matrix (row=true, col=pred) — row-normalised:", flush=True)
header = "              " + "  ".join(f"{n:>13s}" for n in class_names)
print(header, flush=True)
for i, row in enumerate(cm_norm):
    print(f"  {class_names[i]:>12s}  " + "  ".join(f"{v:>13.4f}" for v in row), flush=True)
print(flush=True)

# Save CM
np.savetxt(
    OUT_DIR / "realistic_confusion_matrix_norm.csv",
    cm_norm, delimiter=",", header=",".join(class_names), comments="",
    fmt="%.6f",
)
np.savetxt(
    OUT_DIR / "realistic_confusion_matrix_counts.csv",
    cm_acc, delimiter=",", header=",".join(class_names), comments="",
    fmt="%.2f",
)
print(f"[INFO] saved confusion matrices to {OUT_DIR}", flush=True)

# ── 2. Purity & F1 vs threshold — bootstrap over experimental proportions ─────
print("[INFO] computing purity/F1 curves ...", flush=True)

# We plot for J/psi (cls=0) and psi(2S) (cls=1)
signal_classes = [0, 1]
purity_boots = {c: np.zeros((N_BOOT, len(THRESHOLDS))) for c in signal_classes}
f1_boots     = {c: np.zeros((N_BOOT, len(THRESHOLDS))) for c in signal_classes}

for b in range(N_BOOT):
    sidx, slabels = sample_exp(rng)
    proba_s = y_proba[sidx]

    for c in signal_classes:
        p_signal = proba_s[:, c]
        is_signal = slabels == c

        for t, thresh in enumerate(THRESHOLDS):
            accepted  = p_signal > thresh
            tp = float(( accepted &  is_signal).sum())
            fp = float(( accepted & ~is_signal).sum())
            fn = float((~accepted &  is_signal).sum())

            purity_boots[c][b, t] = tp / (tp + fp + 1e-12)
            f1_boots[c][b, t]     = (2 * tp) / (2 * tp + fp + fn + 1e-12)

    if (b + 1) % 20 == 0:
        print(f"  bootstrap {b+1}/{N_BOOT}", flush=True)

# ── 3. Plot ───────────────────────────────────────────────────────────────────
exp_label = (f"J/ψ={N_JPSI}, ψ(2S)={N_PSIP}, DY={N_DY}, "
             f"Comb={N_COMB} (Comb = {N_COMB//N_DY}×DY)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig.suptitle(
    f"Purity and F1 vs min probability threshold\n"
    f"Bootstrap N = {N_BOOT}  |  {exp_label}",
    fontsize=12, fontweight="bold",
)

color_purity = "#2471A3"
color_f1     = "#E74C3C"

for ax, c in zip(axes, signal_classes):
    pur_mean = purity_boots[c].mean(axis=0)
    pur_std  = purity_boots[c].std(axis=0)
    f1_mean  = f1_boots[c].mean(axis=0)
    f1_std   = f1_boots[c].std(axis=0)

    # best threshold = max mean F1
    best_t = int(f1_mean.argmax())
    best_p = float(THRESHOLDS[best_t])

    ax.plot(THRESHOLDS, pur_mean, color=color_purity, lw=2, label="Purity (mean ± 1σ)")
    ax.fill_between(THRESHOLDS, pur_mean - pur_std, pur_mean + pur_std,
                    color=color_purity, alpha=0.2)

    ax.plot(THRESHOLDS, f1_mean, color=color_f1, lw=2, ls="--", label="F1    (mean ± 1σ)")
    ax.fill_between(THRESHOLDS, f1_mean - f1_std, f1_mean + f1_std,
                    color=color_f1, alpha=0.2)

    ax.axvline(best_p, color="black", lw=1, ls=":")

    # annotation box
    ann_text = (f"optimum\np = {best_p:.2f}\n"
                f"F1 = {f1_mean[best_t]:.3f}\n"
                f"Pur = {pur_mean[best_t]:.3f}")
    ax.annotate(
        ann_text,
        xy=(best_p, max(pur_mean[best_t], f1_mean[best_t])),
        xytext=(best_p + 0.03, 0.60),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFACD",
                  edgecolor="black", linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    ax.set_xlim(0.40, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Min softmax probability threshold", fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title(
        f"{class_names[c]}\n{exp_label}",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=9,
              title=f"Optimum (p = {best_p:.2f})",
              title_fontsize=8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(axis="y", ls="--", alpha=0.4)

    print(f"[INFO] {class_names[c]}  optimum p={best_p:.2f}  "
          f"F1={f1_mean[best_t]:.3f}  Purity={pur_mean[best_t]:.3f}", flush=True)

plt.tight_layout()
out_plot = OUT_DIR / "purity_f1_vs_threshold.png"
fig.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] plot saved → {out_plot}", flush=True)

# ── 4. Also plot realistic CM as heatmap ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
for i in range(K):
    for j in range(K):
        ax2.text(j, i, f"{cm_norm[i,j]:.3f}", ha="center", va="center",
                 fontsize=10, color="white" if cm_norm[i,j] > 0.6 else "black")
ax2.set_xticks(range(K)); ax2.set_xticklabels(class_names, rotation=20, ha="right")
ax2.set_yticks(range(K)); ax2.set_yticklabels(class_names)
ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
ax2.set_title(f"Realistic confusion matrix\n{exp_label}", fontsize=10)
plt.tight_layout()
out_cm = OUT_DIR / "realistic_confusion_matrix.png"
fig2.savefig(out_cm, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[INFO] CM plot saved → {out_cm}", flush=True)
