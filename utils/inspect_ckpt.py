#!/usr/bin/env python3
import sys
import torch
import numpy as np

ckpt_path = sys.argv[1]
p = torch.load(ckpt_path, map_location="cpu")

print("=== keys ===")
print(list(p.keys()))

print("\n=== run_name ===")
print(p.get("run_name"))

print("\n=== cfg ===")
cfg = p.get("cfg", {})
for k in sorted(cfg.keys()):
    print(f"{k:>15s} : {cfg[k]}")

print("\n=== input_dim ===")
print(p.get("input_dim"))

print("\n=== best_val ===")
print(p.get("best_val"))

print("\n=== val_metrics ===")
print(p.get("val_metrics"))

print("\n=== scaler ===")
scaler = p.get("scaler")
if scaler is None:
    print("None (STANDARDIZE was off)")
else:
    mu = scaler["mean"]  # shape (1, F)
    sd = scaler["std"]   # shape (1, F)
    print("mean shape:", mu.shape, "std shape:", sd.shape)
    print("mean[0,:5]:", np.array(mu[0,:5]))
    print("std [0,:5]:", np.array(sd[0,:5]))

