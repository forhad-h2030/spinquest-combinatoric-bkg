"""
utils/plots_multiclass.py
─────────────────────────────────────────────────────────────────────────────
Plotting utilities for the multiclass (J/ψ, ψ(2S), DY, Combinatoric)
particle-ID classifier.

Public API
----------
plot_overlays_multiclass(...)   – per-feature histogram overlays for all classes
plot_confusion_matrix(...)      – annotated confusion-matrix heat-map
plot_roc_multiclass(...)        – one-vs-rest ROC curves for all classes
plot_score_distributions(...)   – classifier output score distributions
plot_training_history(...)      – train / val loss & accuracy curves
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")          # must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

# ── colour palette (colour-blind friendly) ───────────────────────────────────
_DEFAULT_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
_DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _save_or_show(fig: plt.Figure, path: Optional[Path], show: bool) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _stat_text(arr: np.ndarray) -> str:
    """Return a compact mean ± std string."""
    return f"μ={arr.mean():.3g}\nσ={arr.std():.3g}\nn={len(arr):,}"


def _robust_range(arrays: List[np.ndarray], q: float = 1.0) -> tuple[float, float]:
    """
    Compute a shared axis range clipped to the [q, 100-q] percentile
    across all arrays, ignoring NaN / Inf.
    """
    combined = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if len(combined) == 0:
        return 0.0, 1.0
    lo = float(np.percentile(combined, q))
    hi = float(np.percentile(combined, 100.0 - q))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi


# ═════════════════════════════════════════════════════════════════════════════
# plot_overlays_multiclass
# ═════════════════════════════════════════════════════════════════════════════

def plot_overlays_multiclass(
    Xs: List[np.ndarray],
    class_names: List[str],
    feature_names: List[str],
    run_name: str,
    out_dir: Union[str, Path],
    *,
    bins: int = 100,
    density: bool = True,
    fontsize: int = 14,
    show_stats: bool = True,
    legend_all: bool = False,
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    clip_percentile: float = 1.0,
    feature_ranges: Optional[dict] = None,
    ncols: int = 4,
    fig_width_per_col: float = 4.5,
    fig_height_per_row: float = 3.5,
    save: bool = True,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot per-feature histogram overlays for every class.

    Parameters
    ----------
    Xs            : list of 2-D arrays (n_samples, n_features), one per class
    class_names   : class labels, same order as Xs
    feature_names : column names, length == Xs[0].shape[1]
    run_name      : used as figure title and output filename stem
    out_dir       : directory to write PNG files
    bins          : number of histogram bins
    density       : normalise histograms to unit area
    fontsize      : base font size
    show_stats    : overlay μ / σ / n text box per class
    legend_all    : show legend on every subplot (else only first)
    colors        : per-class colours (defaults to colour-blind palette)
    linestyles    : per-class line styles
    clip_percentile : percentile to clip axis range (0 = no clip)
    feature_ranges  : optional dict mapping feature name -> (lo, hi) to
                      override the auto-computed axis range for specific
                      features.  Example: {"rec_dimu_M": (2.0, 6.0)}
    ncols         : subplot columns per figure page
    save          : save PNG to out_dir
    show          : call plt.show() (requires display)

    Returns
    -------
    The last matplotlib Figure created (useful for testing).
    """
    out_dir = Path(out_dir)
    K = len(Xs)
    n_features = len(feature_names)

    if len(class_names) != K:
        raise ValueError(f"len(class_names)={len(class_names)} != len(Xs)={K}")
    for k, X in enumerate(Xs):
        if X.ndim != 2:
            raise ValueError(f"Xs[{k}] must be 2-D, got shape {X.shape}")
        if X.shape[1] != n_features:
            raise ValueError(
                f"Xs[{k}].shape[1]={X.shape[1]} != len(feature_names)={n_features}"
            )

    colors     = (colors or _DEFAULT_COLORS)[:K]
    linestyles = (linestyles or _DEFAULT_LINESTYLES)[:K]

    matplotlib.rcParams.update({"font.size": fontsize})

    nrows = math.ceil(n_features / ncols)
    fig_w = fig_width_per_col * ncols
    fig_h = fig_height_per_row * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).ravel()

    for fi, feat in enumerate(feature_names):
        ax = axes_flat[fi]
        cols = [X[:, fi] for X in Xs]

        # Use caller-supplied range if provided, else auto-compute
        if feature_ranges and feat in feature_ranges:
            lo, hi = feature_ranges[feat]
        else:
            lo, hi = _robust_range(cols, q=clip_percentile)
        bin_edges = np.linspace(lo, hi, bins + 1)

        for k, (col, cname) in enumerate(zip(cols, class_names)):
            finite = col[np.isfinite(col)]
            counts, edges = np.histogram(finite, bins=bin_edges, density=density)
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(
                centres, counts,
                where="mid",
                color=colors[k],
                linestyle=linestyles[k],
                linewidth=1.6,
                label=cname,
            )

            if show_stats:
                # place stat boxes at the top of the panel, stacked per class
                txt = _stat_text(finite)
                ax.text(
                    0.97, 0.97 - k * 0.22,
                    txt,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=max(fontsize - 5, 7),
                    color=colors[k],
                    linespacing=1.3,
                )

        ax.set_xlabel(feat, fontsize=fontsize)
        ax.set_ylabel("Density" if density else "Counts", fontsize=fontsize)
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
        ax.tick_params(labelsize=max(fontsize - 3, 8))

        if fi == 0 or legend_all:
            ax.legend(fontsize=max(fontsize - 3, 8), loc="upper left",
                      framealpha=0.7, handlelength=1.8)

    # Hide unused subplots
    for ax in axes_flat[n_features:]:
        ax.set_visible(False)

    fig.suptitle(f"Feature overlays — {run_name}", fontsize=fontsize + 2, y=1.01)

    last_fig = fig
    if save:
        out_path = out_dir / f"{run_name}.feature_overlays.png"
        _save_or_show(fig, out_path, show)
        print(f"[plots] saved → {out_path}")
    else:
        _save_or_show(fig, None, show)

    return last_fig


# ═════════════════════════════════════════════════════════════════════════════
# plot_confusion_matrix
# ═════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    run_name: str,
    out_dir: Union[str, Path],
    *,
    normalize: bool = True,
    fontsize: int = 14,
    cmap: str = "Blues",
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Plot an annotated confusion matrix.

    Parameters
    ----------
    cm          : (K, K) integer confusion matrix (rows=true, cols=pred)
    normalize   : if True, normalise each row to sum to 1
    """
    out_dir = Path(out_dir)
    K = len(class_names)
    cm = np.array(cm, dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
        label = "Fraction"
    else:
        cm_plot = cm
        fmt = "d"
        label = "Count"

    fig, ax = plt.subplots(figsize=(K * 1.6 + 1.5, K * 1.4 + 1.5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap, vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, label=label)

    thresh = cm_plot.max() / 2.0
    for i in range(K):
        for j in range(K):
            val = cm_plot[i, j]
            txt = f"{val:{fmt}}" if fmt == "d" else f"{val:.2f}"
            ax.text(j, i, txt,
                    ha="center", va="center", fontsize=fontsize,
                    color="white" if val > thresh else "black")

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("True", fontsize=fontsize)
    ax.set_title(f"Confusion Matrix — {run_name}", fontsize=fontsize + 2)
    fig.tight_layout()

    out_path = out_dir / f"{run_name}.confusion_matrix.png" if save else None
    _save_or_show(fig, out_path, show)
    if save:
        print(f"[plots] saved → {out_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# plot_roc_multiclass
# ═════════════════════════════════════════════════════════════════════════════

def plot_roc_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    run_name: str,
    out_dir: Union[str, Path],
    *,
    colors: Optional[List[str]] = None,
    fontsize: int = 14,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    One-vs-rest ROC curves for each class.

    Parameters
    ----------
    y_true  : (N,) integer class labels
    y_proba : (N, K) predicted probabilities
    """
    from sklearn.metrics import roc_curve, auc

    out_dir = Path(out_dir)
    K = len(class_names)
    colors = (colors or _DEFAULT_COLORS)[:K]

    fig, ax = plt.subplots(figsize=(7, 6))

    for k, cname in enumerate(class_names):
        y_bin = (y_true == k).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, k])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[k], lw=2,
                label=f"{cname}  (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize)
    ax.set_title(f"ROC (one-vs-rest) — {run_name}", fontsize=fontsize + 2)
    ax.legend(fontsize=fontsize - 1, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=fontsize - 2)
    fig.tight_layout()

    out_path = out_dir / f"{run_name}.roc.png" if save else None
    _save_or_show(fig, out_path, show)
    if save:
        print(f"[plots] saved → {out_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# plot_score_distributions
# ═════════════════════════════════════════════════════════════════════════════

def plot_score_distributions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    run_name: str,
    out_dir: Union[str, Path],
    *,
    bins: int = 60,
    colors: Optional[List[str]] = None,
    fontsize: int = 14,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    For each class node k, plot the output score distribution split by
    true class label (signal vs background).
    """
    out_dir = Path(out_dir)
    K = len(class_names)
    colors = (colors or _DEFAULT_COLORS)[:K]
    bin_edges = np.linspace(0, 1, bins + 1)

    fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 4.5), constrained_layout=True)
    if K == 1:
        axes = [axes]

    for k, (ax, cname) in enumerate(zip(axes, class_names)):
        for j, jname in enumerate(class_names):
            mask = y_true == j
            scores = y_proba[mask, k]
            ax.hist(scores, bins=bin_edges, density=True,
                    histtype="step", lw=1.8,
                    color=colors[j],
                    linestyle=_DEFAULT_LINESTYLES[j % len(_DEFAULT_LINESTYLES)],
                    label=jname)
        ax.set_xlabel(f"Score for {cname}", fontsize=fontsize)
        ax.set_ylabel("Density", fontsize=fontsize)
        ax.set_title(cname, fontsize=fontsize)
        ax.legend(fontsize=fontsize - 3, loc="upper center", framealpha=0.7)
        ax.tick_params(labelsize=fontsize - 3)

    fig.suptitle(f"Score distributions — {run_name}", fontsize=fontsize + 2)

    out_path = out_dir / f"{run_name}.score_distributions.png" if save else None
    _save_or_show(fig, out_path, show)
    if save:
        print(f"[plots] saved → {out_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# plot_training_history
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_history(
    history: dict,
    run_name: str,
    out_dir: Union[str, Path],
    *,
    fontsize: int = 13,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Plot training and validation loss (and accuracy if available).

    Parameters
    ----------
    history : dict with keys "train" and "val", each a list of dicts
              containing at minimum {"epoch": int, "loss": float}.
              Val dicts may also contain {"acc": float, "macro_f1": float}.
    """
    out_dir = Path(out_dir)

    tr  = history.get("train", [])
    val = history.get("val", [])

    epochs_tr  = [d["epoch"] for d in tr]
    loss_tr    = [d["loss"]  for d in tr]
    epochs_val = [d["epoch"] for d in val]
    loss_val   = [d["loss"]  for d in val]

    has_acc = val and "acc" in val[0]
    ncols   = 2 if has_acc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    # Loss
    axes[0].plot(epochs_tr, loss_tr, label="Train", color="#0072B2", lw=1.8)
    axes[0].plot(epochs_val, loss_val, label="Val",   color="#D55E00", lw=1.8, linestyle="--")
    axes[0].set_xlabel("Epoch", fontsize=fontsize)
    axes[0].set_ylabel("Loss",  fontsize=fontsize)
    axes[0].set_title("Loss",   fontsize=fontsize + 1)
    axes[0].legend(fontsize=fontsize - 1)
    axes[0].tick_params(labelsize=fontsize - 2)

    # Accuracy & macro F1
    if has_acc:
        acc_val = [d["acc"] for d in val]
        axes[1].plot(epochs_val, acc_val, label="Val Acc", color="#009E73", lw=1.8)
        if "macro_f1" in val[0]:
            f1_val = [d["macro_f1"] for d in val]
            axes[1].plot(epochs_val, f1_val, label="Val F1 (macro)",
                         color="#CC79A7", lw=1.8, linestyle="--")
        axes[1].set_xlabel("Epoch",    fontsize=fontsize)
        axes[1].set_ylabel("Metric",   fontsize=fontsize)
        axes[1].set_title("Val metrics", fontsize=fontsize + 1)
        axes[1].legend(fontsize=fontsize - 1)
        axes[1].set_ylim(0, 1.02)
        axes[1].tick_params(labelsize=fontsize - 2)

    fig.suptitle(f"Training history — {run_name}", fontsize=fontsize + 2)
    fig.tight_layout()

    out_path = out_dir / f"{run_name}.training_history.png" if save else None
    _save_or_show(fig, out_path, show)
    if save:
        print(f"[plots] saved → {out_path}")
    return fig
