# utils/plots_binary.py
from __future__ import annotations

from pathlib import Path
import datetime
import os
import numpy as np
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in s.strip())


def _timestamp(add_timestamp: bool) -> str:
    return datetime.datetime.now().strftime("_%Y%m%d_%H%M%S") if add_timestamp else ""


def _ensure_dir(out_dir: str | Path | None) -> Path:
    d = Path(".") if out_dir is None else Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d

def plot_overlays(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    feature_names: list[str],
    *,
    pos_label: str = "POS",
    neg_label: str = "NEG",
    run_name: str = "run",
    out_dir: str | Path | None = None,
    bins: int = 50,
    max_features: int | None = None,
    density: bool = True,
    show_stats: bool = True,
    legend_all: bool = False,   
    save: bool = True,
    show: bool = False,
    fmt: str = "png",
    add_timestamp: bool = False,
    fontsize: int = 16,
):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "legend.fontsize": max(12, fontsize - 5),
        "xtick.labelsize": max(12, fontsize - 6),
        "ytick.labelsize": max(12, fontsize - 6),
    })

    n_feat = int(X_pos.shape[1])
    if max_features is not None:
        n_feat = min(n_feat, int(max_features))

    d = _ensure_dir(out_dir) if save else None
    ts = _timestamp(add_timestamp)

    n_cols, n_rows = 3, 3
    per_page = n_cols * n_rows  # 9
    n_pages = int(np.ceil(n_feat / per_page))

    saved_paths: list[Path] = []

    def _stats(arr: np.ndarray) -> tuple[int, float, float]:
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0, float("nan"), float("nan")
        return int(arr.size), float(arr.mean()), float(arr.std())

    for page in range(n_pages):
        start = page * per_page
        stop = min(start + per_page, n_feat)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(7.0 * n_cols, 5.2 * n_rows),
            squeeze=False
        )
        axes_flat = axes.reshape(-1)

        for k, i in enumerate(range(start, stop)):
            ax = axes_flat[k]

            if show_stats:
                nN, muN, sdN = _stats(X_neg[:, i])
                nP, muP, sdP = _stats(X_pos[:, i])
                lab_neg = f"{neg_label}\nN={nN:,}\nMean={muN:.3g}\nSigma={sdN:.3g}"
                lab_pos = f"{pos_label}\nN={nP:,}\nMean={muP:.3g}\nSigma={sdP:.3g}"
            else:
                lab_neg = neg_label
                lab_pos = pos_label

            ax.hist(
                X_neg[:, i],
                bins=bins,
                density=density,
                histtype="step",
                linewidth=2,
                label=lab_neg,
            )
            ax.hist(
                X_pos[:, i],
                bins=bins,
                density=density,
                histtype="step",
                linewidth=2,
                label=lab_pos,
            )

            ax.set_xlabel(feature_names[i] if i < len(feature_names) else f"f{i}")
            ax.set_ylabel("Norm. count" if density else "Counts")

            # legend behavior
            if legend_all:
                ax.legend(loc="best", frameon=True)
            else:
                # only top-left panel per page
                if k == 0:
                    ax.legend(loc="best", frameon=True)

        for k in range(stop - start, per_page):
            axes_flat[k].axis("off")

        fig.suptitle(
            f"Feature overlays: {pos_label} vs {neg_label}  [{run_name}]  (page {page+1}/{n_pages})",
            y=1.02
        )
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.30, hspace=0.35)

        if save:
            fname = f"overlay_{_safe(run_name)}_{_safe(pos_label)}_vs_{_safe(neg_label)}_p{page+1}{ts}.{fmt}"
            out_path = d / fname
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] saved overlay figure: {out_path}")
            saved_paths.append(out_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_paths

def plot_confusion_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    pos_label: str = "POS",
    neg_label: str = "NEG",
    run_name: str = "run",
    out_dir: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    fmt: str = "png",
    add_timestamp: bool = False,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = int(cm.sum())
    TN, FP, FN, TP = cm.ravel()
    print(f"[INFO] {run_name} thr={threshold:.3f} TN={TN} FP={FP} FN={FN} TP={TP}")

    cm_pct = cm / max(total, 1) * 100.0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of all test events")

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[neg_label, pos_label],
        yticklabels=[neg_label, pos_label],
        xlabel="Predicted",
        ylabel="True",
    )
    ax.set_title(f"Confusion matrix: {pos_label} vs {neg_label}\n[{run_name}]  thr={threshold:.2f}")

    thresh_val = cm_pct.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%",
                ha="center", va="center",
                color="white" if cm_pct[i, j] > thresh_val else "black",
            )

    fig.tight_layout()

    saved_path = None
    if save:
        d = _ensure_dir(out_dir)
        ts = _timestamp(add_timestamp)
        thr_tag = f"thr{threshold:.2f}".replace(".", "p")
        fname = f"cm_{_safe(run_name)}_{_safe(pos_label)}_vs_{_safe(neg_label)}_{thr_tag}{ts}.{fmt}"
        saved_path = d / fname
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] saved confusion matrix: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path, cm, cm_pct


def plot_prob_hists(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bins: int = 50,
    pos_label: str = "POS",
    neg_label: str = "NEG",
    run_name: str = "run",
    out_dir: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    fmt: str = "png",
    add_timestamp: bool = False,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    m0 = (y_true == 0)
    m1 = (y_true == 1)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(
        y_prob[m0],
        bins=bins, range=(0, 1), density=True,
        histtype="step", linewidth=2,
        label=f"True {neg_label} (N={int(m0.sum())})",
    )
    plt.hist(
        y_prob[m1],
        bins=bins, range=(0, 1), density=True,
        histtype="step", linewidth=2,
        label=f"True {pos_label} (N={int(m1.sum())})",
    )
    plt.xlabel(f"Predicted p({pos_label})")
    plt.ylabel("Norm. count")
    plt.title(f"Predicted probability distribution: {pos_label} vs {neg_label}\n[{run_name}]")
    plt.legend()
    plt.tight_layout()

    saved_path = None
    if save:
        d = _ensure_dir(out_dir)
        ts = _timestamp(add_timestamp)
        fname = f"prob_{_safe(run_name)}_{_safe(pos_label)}_vs_{_safe(neg_label)}{ts}.{fmt}"
        saved_path = d / fname
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] saved prob hist figure: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path

