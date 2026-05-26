#!/usr/bin/env python3
"""
Purity and Efficiency vs softmax threshold with bootstrap uncertainty bands.

For each signal class (J/psi, psi(2S)), plots:
  - Purity    = TP / (TP + FP)   [blue solid]
  - Efficiency = TP / (TP + FN)  [red dashed]
  - Crossover point where Purity = Efficiency
  - ±1σ bootstrap bands

Env vars:
  BUNDLE_PATH   path to test_bundle.npz                    (required)
  OUT_DIR       output directory        (default: same dir as bundle)
  N_BOOT        bootstrap iterations   (default: 500)
  N_JPSI        experimental J/psi count   (default: 438)
  N_PSIP        experimental psi(2S) count (default: 20)
  N_DY          experimental DY count      (default: 266)
  N_COMB        experimental Comb count    (default: 267)
  MASS_WIN_LO   mass window lower edge GeV (default: 2.779)
  MASS_WIN_HI   mass window upper edge GeV (default: 3.814)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── config ────────────────────────────────────────────────────────────────────
BUNDLE_PATH  = Path(os.environ["BUNDLE_PATH"])
OUT_DIR      = Path(os.environ.get("OUT_DIR", str(BUNDLE_PATH.parent)))
N_BOOT       = int(os.environ.get("N_BOOT",  "500"))
N_JPSI       = int(os.environ.get("N_JPSI",  "438"))
N_PSIP       = int(os.environ.get("N_PSIP",  "20"))
N_DY         = int(os.environ.get("N_DY",    "266"))
N_COMB       = int(os.environ.get("N_COMB",  "267"))
MASS_WIN_LO  = float(os.environ.get("MASS_WIN_LO", "2.779"))
MASS_WIN_HI  = float(os.environ.get("MASS_WIN_HI", "3.814"))

EXP_COUNTS   = [N_JPSI, N_PSIP, N_DY, N_COMB]
THRESHOLDS   = np.linspace(0.0, 0.99, 100)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load bundle ───────────────────────────────────────────────────────────────
print(f"[INFO] bundle : {BUNDLE_PATH}")
bundle      = np.load(BUNDLE_PATH, allow_pickle=True)
y_proba     = bundle["y_proba"].astype(np.float64)   # (N, 4)
y_test      = bundle["y_test"].astype(np.int64)       # (N,)
class_names = list(bundle["class_names"])
K           = len(class_names)

rng         = np.random.default_rng(42)
class_idx   = [np.where(y_test == c)[0] for c in range(K)]

print(f"[INFO] test set (balanced MC): {[len(i) for i in class_idx]}")
print(f"[INFO] experimental counts: {dict(zip(class_names, EXP_COUNTS))}")
print(f"[INFO] N_BOOT={N_BOOT}  thresholds={len(THRESHOLDS)}")


def sample_exp(rng):
    idxs, labels = [], []
    for c, (cidx, n) in enumerate(zip(class_idx, EXP_COUNTS)):
        chosen = rng.choice(cidx, size=n, replace=True)
        idxs.append(chosen)
        labels.append(np.full(n, c, dtype=np.int64))
    return np.concatenate(idxs), np.concatenate(labels)


# ── bootstrap purity + efficiency ─────────────────────────────────────────────
signal_classes = [0, 1]   # J/psi, psi(2S)
purity_boots   = {c: np.zeros((N_BOOT, len(THRESHOLDS))) for c in signal_classes}
effic_boots    = {c: np.zeros((N_BOOT, len(THRESHOLDS))) for c in signal_classes}

print("[INFO] running bootstrap ...")
for b in range(N_BOOT):
    sidx, slabels = sample_exp(rng)
    proba_s = y_proba[sidx]

    for c in signal_classes:
        p_signal  = proba_s[:, c]
        is_signal = slabels == c

        for t, thresh in enumerate(THRESHOLDS):
            accepted = p_signal > thresh
            tp = float(( accepted &  is_signal).sum())
            fp = float(( accepted & ~is_signal).sum())
            fn = float((~accepted &  is_signal).sum())

            purity_boots[c][b, t] = tp / (tp + fp + 1e-12)
            effic_boots [c][b, t] = tp / (tp + fn + 1e-12)

    if (b + 1) % 100 == 0:
        print(f"  bootstrap {b+1}/{N_BOOT}")


# ── plot one figure per signal class ─────────────────────────────────────────
exp_label = (f"J/ψ={N_JPSI},  ψ(2S)={N_PSIP},  DY={N_DY},  Comb={N_COMB} "
             f"(DY+Comb={N_DY+N_COMB})")
mass_label = f"mass window [{MASS_WIN_LO:.3f}, {MASS_WIN_HI:.3f}] GeV"

COLOR_PUR  = "#2471A3"   # blue
COLOR_EFF  = "#C0392B"   # red

for c in signal_classes:
    cname = class_names[c]

    pur_mean = purity_boots[c].mean(axis=0)
    pur_std  = purity_boots[c].std(axis=0)
    eff_mean = effic_boots[c].mean(axis=0)
    eff_std  = effic_boots[c].std(axis=0)

    # Crossover: first threshold where purity crosses above efficiency.
    # Restrict to the region where both curves are non-trivial (> 1%) to
    # avoid picking the collapsed high-threshold region where both → 0.
    valid = (pur_mean > 0.01) | (eff_mean > 0.01)
    signed = pur_mean - eff_mean          # negative when eff > pur, positive after crossing
    sign_changes = np.where(np.diff(np.sign(signed[valid])))[0]
    valid_idx = np.where(valid)[0]
    if len(sign_changes) > 0:
        cross_t = int(valid_idx[sign_changes[0]])
    else:
        # No clean crossing; pick the closest approach in the valid region
        abs_diff = np.where(valid, np.abs(signed), 1e9)
        cross_t = int(abs_diff.argmin())
    cross_p   = float(THRESHOLDS[cross_t])
    cross_pur = float(pur_mean[cross_t])
    cross_eff = float(eff_mean[cross_t])

    fig, ax = plt.subplots(figsize=(8, 6))

    # purity
    ax.plot(THRESHOLDS, pur_mean, color=COLOR_PUR, lw=2.0,
            label="Purity  (mean ± 1σ)")
    ax.fill_between(THRESHOLDS, pur_mean - pur_std, pur_mean + pur_std,
                    color=COLOR_PUR, alpha=0.18)

    # efficiency
    ax.plot(THRESHOLDS, eff_mean, color=COLOR_EFF, lw=2.0, ls="--",
            label="Efficiency  (mean ± 1σ)")
    ax.fill_between(THRESHOLDS, eff_mean - eff_std, eff_mean + eff_std,
                    color=COLOR_EFF, alpha=0.18)

    # crossover vertical line
    ax.axvline(cross_p, color="black", lw=0.9, ls=":", label=f"Crossover  (p = {cross_p:.2f})")

    # crossover annotation box
    ax.annotate(
        f"crossover\np = {cross_p:.2f}\nPur = {cross_pur:.3f}\nEff = {cross_eff:.3f}",
        xy=(cross_p, (cross_pur + cross_eff) / 2),
        xytext=(cross_p + 0.04, (cross_pur + cross_eff) / 2 - 0.05),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFACD",
                  edgecolor="black", linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    # dual y-axis (mirror)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_ylabel("Metric value", fontsize=11)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Min softmax probability threshold", fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.legend(loc="center left", fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.3)

    fig.suptitle(
        f"{cname}\n{exp_label}  |  {mass_label}  |  {N_BOOT} resamples",
        fontsize=10, fontweight="bold",
    )

    plt.tight_layout()
    slug = cname.replace("/", "").replace("(", "").replace(")", "").replace(" ", "_")
    out_path = OUT_DIR / f"purity_efficiency_{slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] {cname:10s}  crossover p={cross_p:.2f}  "
          f"Pur={cross_pur:.3f}  Eff={cross_eff:.3f}  → {out_path}")
