#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def to_df(d, key):
    if key not in d:
        return pd.DataFrame()
    return pd.DataFrame(d[key]).sort_values("epoch")

def main(train_val_json_path):
    d = load_json(train_val_json_path)

    train = to_df(d, "train")
    val   = to_df(d, "val")

    plt.figure()
    if not train.empty:
        plt.plot(train["epoch"], train["loss"], marker="o", label="Train Loss")
    if not val.empty:
        plt.plot(val["epoch"], val["loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if (not val.empty) and ("acc@0.5" in val.columns):
        plt.figure()
        plt.plot(val["epoch"], val["acc@0.5"], marker="o", label="Val Acc@0.5")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if (not val.empty) and ("auc" in val.columns):
        plt.figure()
        plt.plot(val["epoch"], val["auc"], marker="o", label="Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("Validation AUC")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # python3 plot_curves.py metrics.json
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 plot_curves.py <metrics.json>")
    main(sys.argv[1])

