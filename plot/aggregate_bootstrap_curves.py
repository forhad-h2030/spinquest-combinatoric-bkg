#!/usr/bin/env python3
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def history_to_df(hist, key):
    """key: 'train' or 'val' -> DataFrame with epoch + metrics"""
    if key not in hist:
        return pd.DataFrame()
    df = pd.DataFrame(hist[key])
    if df.empty:
        return df
    return df.sort_values("epoch")


def collect_histories(out_root, run_name):
    pattern = str(Path(out_root) / run_name / "boot_*" / f"{run_name}.history.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[ERROR] No files matched: {pattern}")
    return [Path(x) for x in files]


def agg_metric(dfs, metric):
    """
    dfs: list of DataFrames containing 'epoch' and metric
    returns: (epochs, mean, std)
    """
    if not dfs:
        return None

    # outer-join on epoch
    merged = None
    for i, df in enumerate(dfs):
        if df.empty or ("epoch" not in df.columns) or (metric not in df.columns):
            continue
        tmp = df[["epoch", metric]].copy()
        tmp = tmp.rename(columns={metric: f"{metric}_{i}"})
        merged = tmp if merged is None else pd.merge(merged, tmp, on="epoch", how="outer")

    if merged is None or merged.empty:
        return None

    merged = merged.sort_values("epoch")
    cols = [c for c in merged.columns if c.startswith(metric + "_")]
    vals = merged[cols].to_numpy(dtype=float)

    mean = np.nanmean(vals, axis=1)
    std  = np.nanstd(vals, axis=1)
    epochs = merged["epoch"].to_numpy(dtype=int)
    return epochs, mean, std


def plot_mean_std(epochs, mean, std, title, ylabel, save_path=None):
    plt.figure()
    plt.plot(epochs, mean, marker="o", label="mean")
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.25, label="±1σ")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[OK] wrote {save_path}")
    else:
        plt.show()
    plt.close()


def main(out_root, run_name, save=False):
    files = collect_histories(out_root, run_name)
    print(f"[INFO] found {len(files)} bootstrap histories")

    train_dfs = []
    val_dfs = []
    for f in files:
        h = load_json(f)
        train_dfs.append(history_to_df(h, "train"))
        val_dfs.append(history_to_df(h, "val"))

    out_dir = Path(out_root) / run_name

    # Train/Val loss on one figure
    tr = agg_metric(train_dfs, "loss")
    va = agg_metric(val_dfs, "loss")

    if tr is not None or va is not None:
        plt.figure()
        if tr is not None:
            e, m, s = tr
            plt.plot(e, m, marker="o", label="Train loss (mean)")
            plt.fill_between(e, m - s, m + s, alpha=0.20)
        if va is not None:
            e, m, s = va
            plt.plot(e, m, marker="o", label="Val loss (mean)")
            plt.fill_between(e, m - s, m + s, alpha=0.20)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{run_name}: Loss (bootstrap mean ± 1σ)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            p = out_dir / f"{run_name}.loss_bootstrap_mean_std.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            print(f"[OK] wrote {p}")
        else:
            plt.show()
        plt.close()

    # Val acc@0.5
    acc = agg_metric(val_dfs, "acc@0.5")
    if acc is not None:
        e, m, s = acc
        plot_mean_std(
            e, m, s,
            title=f"{run_name}: Val Acc@0.5 (bootstrap mean ± 1σ)",
            ylabel="Accuracy",
            save_path=(out_dir / f"{run_name}.val_acc_bootstrap_mean_std.png") if save else None
        )

    # Val AUC
    auc = agg_metric(val_dfs, "auc")
    if auc is not None:
        e, m, s = auc
        plot_mean_std(
            e, m, s,
            title=f"{run_name}: Val AUC (bootstrap mean ± 1σ)",
            ylabel="AUC",
            save_path=(out_dir / f"{run_name}.val_auc_bootstrap_mean_std.png") if save else None
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", help="e.g. outputs_array_6876064")
    ap.add_argument("--run", required=True, help="e.g. dy_comb_raw")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    main(args.out_root, args.run, save=args.save)

