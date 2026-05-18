#!/usr/bin/env python3
"""
Mass-window scan: accuracy, macro F1, and J/psi purity vs dimuon mass.

Selects events in [MASS_LO, MASS_HI] GeV, bins them, and for each bin
computes classifier metrics using experimental proportions (bootstrap).

Env vars:
  BUNDLE_PATH   path to test_bundle.npz              (required)
  CKPT_PATH     path to .best.pth (needed for scaler) (required)
  OUT_DIR       output directory  (default: same dir as bundle)
  N_BOOT        bootstrap iterations (default: 200)
  N_JPSI        experimental J/psi total  (default: 500)
  N_PSIP        experimental psi(2S) total (default: 150)
  N_DY          experimental DY total      (default: 500)
  N_COMB        experimental comb total    (default: 2000)
  MASS_LO       lower mass edge GeV (default: 2.5)
  MASS_HI       upper mass edge GeV (default: 3.5)
  N_BINS        number of mass bins (default: 20)
  JPSI_THRESH   J/psi probability threshold for purity (default: 0.56)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── config ────────────────────────────────────────────────────────────────────
BUNDLE_PATH  = Path(os.environ["BUNDLE_PATH"])
CKPT_PATH    = Path(os.environ["CKPT_PATH"])
OUT_DIR      = Path(os.environ.get("OUT_DIR",      str(BUNDLE_PATH.parent)))
N_BOOT       = int(os.environ.get("N_BOOT",        "200"))
N_JPSI       = int(os.environ.get("N_JPSI",        "500"))
N_PSIP       = int(os.environ.get("N_PSIP",        "150"))
N_DY         = int(os.environ.get("N_DY",          "500"))
N_COMB       = int(os.environ.get("N_COMB",        "2000"))
MASS_LO      = float(os.environ.get("MASS_LO",     "2.5"))
MASS_HI      = float(os.environ.get("MASS_HI",     "3.5"))
N_BINS       = int(os.environ.get("N_BINS",        "20"))
JPSI_THRESH  = float(os.environ.get("JPSI_THRESH", "0.56"))

EXP_COUNTS   = np.array([N_JPSI, N_PSIP, N_DY, N_COMB], dtype=float)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
MASS_IDX     = 4   # rec_dimu_M index in feature vector

# ── load data ─────────────────────────────────────────────────────────────────
print(f"[INFO] bundle : {BUNDLE_PATH}", flush=True)
bundle      = np.load(BUNDLE_PATH, allow_pickle=True)
y_proba     = bundle["y_proba"].astype(np.float64)
y_test      = bundle["y_test"].astype(np.int64)
X_std       = bundle["X_test"].astype(np.float64)   # standardized features
class_names = list(bundle["class_names"])
K           = len(class_names)

# un-standardize mass
ckpt   = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
scaler = ckpt["scaler"]
mass_mean = float(scaler["mean"].flatten()[MASS_IDX])
mass_std  = float(scaler["std"].flatten()[MASS_IDX])
mass_gev  = X_std[:, MASS_IDX] * mass_std + mass_mean

print(f"[INFO] mass range in data: [{mass_gev.min():.2f}, {mass_gev.max():.2f}] GeV", flush=True)
print(f"[INFO] scan window: [{MASS_LO}, {MASS_HI}] GeV  bins={N_BINS}", flush=True)

# ── mass bins ─────────────────────────────────────────────────────────────────
edges      = np.linspace(MASS_LO, MASS_HI, N_BINS + 1)
bin_centers= 0.5 * (edges[:-1] + edges[1:])
bin_width  = edges[1] - edges[0]

rng = np.random.default_rng(42)

# per-class indices in the full test set
class_idx_full = [np.where(y_test == c)[0] for c in range(K)]

# ── per-bin bootstrap metrics ─────────────────────────────────────────────────
acc_all    = np.full((N_BINS, N_BOOT), np.nan)
f1_all     = np.full((N_BINS, N_BOOT), np.nan)
purity_all = np.full((N_BINS, N_BOOT), np.nan)   # J/psi purity at JPSI_THRESH
n_pass_all = np.zeros((N_BINS,), dtype=float)     # mean events passing threshold

warnings.filterwarnings("ignore", category=RuntimeWarning)
print("[INFO] scanning mass bins ...", flush=True)
for bi in range(N_BINS):
    lo, hi = edges[bi], edges[bi + 1]

    # indices per class within this mass bin
    cidx_bin = [
        class_idx_full[c][
            (mass_gev[class_idx_full[c]] >= lo) & (mass_gev[class_idx_full[c]] < hi)
        ]
        for c in range(K)
    ]
    counts_in_bin = np.array([len(ci) for ci in cidx_bin], dtype=float)

    if counts_in_bin.min() < 2:
        continue   # skip bins with essentially no events in some class

    # scale experimental counts proportionally to MC class fractions in this bin
    mc_total_in_bin = counts_in_bin.sum()
    class_fracs     = counts_in_bin / counts_in_bin.sum()
    # experimental counts per class scaled to total_exp * class_frac
    total_exp       = EXP_COUNTS.sum()
    exp_in_bin      = np.maximum((class_fracs * total_exp).astype(int), 1)

    for b in range(N_BOOT):
        # sample experimental proportions within this bin
        sampled_idx, sampled_labels = [], []
        for c in range(K):
            n_draw = exp_in_bin[c]
            if len(cidx_bin[c]) == 0:
                continue
            chosen = rng.choice(cidx_bin[c], size=n_draw, replace=True)
            sampled_idx.append(chosen)
            sampled_labels.append(np.full(n_draw, c, dtype=np.int64))

        if not sampled_idx:
            continue

        idx_arr    = np.concatenate(sampled_idx)
        labels_arr = np.concatenate(sampled_labels)
        proba_s    = y_proba[idx_arr]
        pred_s     = proba_s.argmax(axis=1)

        # accuracy
        acc_all[bi, b] = (pred_s == labels_arr).mean()

        # macro F1
        f1s = []
        for c in range(K):
            tp = ((pred_s == c) & (labels_arr == c)).sum()
            fp = ((pred_s == c) & (labels_arr != c)).sum()
            fn = ((pred_s != c) & (labels_arr == c)).sum()
            f1s.append(2 * tp / (2 * tp + fp + fn + 1e-12))
        f1_all[bi, b] = np.mean(f1s)

        # J/psi purity at threshold
        p_jpsi    = proba_s[:, 0]
        accepted  = p_jpsi > JPSI_THRESH
        tp_j      = float((accepted & (labels_arr == 0)).sum())
        fp_j      = float((accepted & (labels_arr != 0)).sum())
        purity_all[bi, b] = tp_j / (tp_j + fp_j + 1e-12)
        n_pass_all[bi]   += accepted.sum()

    n_pass_all[bi] /= N_BOOT

    if (bi + 1) % 5 == 0:
        print(f"  bin {bi+1}/{N_BINS}  M=[{lo:.2f},{hi:.2f}]  "
              f"acc={np.nanmean(acc_all[bi]):.3f}  "
              f"f1={np.nanmean(f1_all[bi]):.3f}  "
              f"purity(J/ψ)={np.nanmean(purity_all[bi]):.3f}", flush=True)

# ── compute mass histogram per class (MC, unnormalised) ───────────────────────
hist_mc = np.zeros((K, N_BINS))
for c in range(K):
    hist_mc[c], _ = np.histogram(mass_gev[y_test == c], bins=edges)

# scale to experimental proportions for display
scale = EXP_COUNTS / hist_mc.sum(axis=1).clip(1)
hist_exp = hist_mc * scale[:, None]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True,
                         gridspec_kw={"height_ratios": [1.6, 1, 1]})
fig.suptitle(
    f"Classifier performance vs dimuon mass  [{MASS_LO}–{MASS_HI} GeV]\n"
    f"J/ψ={N_JPSI}  ψ(2S)={N_PSIP}  DY={N_DY}  Comb={N_COMB}  "
    f"(bootstrap N={N_BOOT})",
    fontsize=11, fontweight="bold",
)

# ── panel 1: mass histogram with experimental proportions ────────────────────
ax0 = axes[0]
bottom = np.zeros(N_BINS)
for c in range(K):
    ax0.bar(bin_centers, hist_exp[c], width=bin_width * 0.9,
            bottom=bottom, color=CLASS_COLORS[c], label=class_names[c], alpha=0.85)
    bottom += hist_exp[c]
ax0.set_ylabel("Events (scaled to exp.)", fontsize=10)
ax0.legend(fontsize=9, loc="upper right")
ax0.set_title("Mass distribution (experimental proportions)", fontsize=10)
ax0.yaxis.set_major_locator(ticker.MaxNLocator(5))

# vertical line at J/psi mass
ax0.axvline(3.097, color="black", lw=1, ls="--", alpha=0.6, label="J/ψ mass")
ax0.axvline(3.686, color="gray",  lw=1, ls=":",  alpha=0.6, label="ψ' mass")

# ── panel 2: accuracy and macro F1 ───────────────────────────────────────────
ax1 = axes[1]
valid = ~np.all(np.isnan(acc_all), axis=1)

acc_mean = np.nanmean(acc_all, axis=1)
acc_std  = np.nanstd(acc_all,  axis=1)
f1_mean  = np.nanmean(f1_all,  axis=1)
f1_std   = np.nanstd(f1_all,   axis=1)

ax1.plot(bin_centers[valid], acc_mean[valid], color="#2471A3", lw=2, label="Accuracy")
ax1.fill_between(bin_centers[valid],
                 acc_mean[valid] - acc_std[valid],
                 acc_mean[valid] + acc_std[valid],
                 color="#2471A3", alpha=0.2)

ax1.plot(bin_centers[valid], f1_mean[valid], color="#E74C3C", lw=2, ls="--", label="Macro F1")
ax1.fill_between(bin_centers[valid],
                 f1_mean[valid] - f1_std[valid],
                 f1_mean[valid] + f1_std[valid],
                 color="#E74C3C", alpha=0.2)

ax1.axvline(3.097, color="black", lw=1, ls="--", alpha=0.5)
ax1.set_ylabel("Metric value", fontsize=10)
ax1.set_ylim(0.4, 1.05)
ax1.legend(fontsize=9, loc="lower right")
ax1.set_title(f"Accuracy and Macro F1 (mean ± 1σ)", fontsize=10)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax1.grid(axis="y", ls="--", alpha=0.3)

# ── panel 3: J/psi purity at threshold ───────────────────────────────────────
ax2 = axes[2]
pur_mean = np.nanmean(purity_all, axis=1)
pur_std  = np.nanstd(purity_all,  axis=1)

ax2.plot(bin_centers[valid], pur_mean[valid], color="#27AE60", lw=2,
         label=f"J/ψ purity (p>{JPSI_THRESH})")
ax2.fill_between(bin_centers[valid],
                 pur_mean[valid] - pur_std[valid],
                 pur_mean[valid] + pur_std[valid],
                 color="#27AE60", alpha=0.2)

ax2.axvline(3.097, color="black", lw=1, ls="--", alpha=0.5)
ax2.set_ylabel("J/ψ purity", fontsize=10)
ax2.set_xlabel("Dimuon invariant mass M [GeV]", fontsize=10)
ax2.set_ylim(0.0, 1.05)
ax2.legend(fontsize=9, loc="lower right")
ax2.set_title(f"J/ψ purity at threshold p > {JPSI_THRESH} (mean ± 1σ)", fontsize=10)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.grid(axis="y", ls="--", alpha=0.3)

for ax in axes:
    ax.set_xlim(MASS_LO, MASS_HI)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

plt.tight_layout()
out_path = OUT_DIR / f"mass_scan_{MASS_LO:.1f}_{MASS_HI:.1f}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[INFO] plot saved → {out_path}", flush=True)
