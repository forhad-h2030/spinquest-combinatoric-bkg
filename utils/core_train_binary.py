# utils/core_train_binary.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score


# Config
@dataclass
class TrainConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    epochs: int = 200
    lr: float = 5e-4
    batch_size: int = 1024
    seed: int = 42
    standardize: bool = False  # keep False for your current style
    hidden_dim: int = 512
    num_layers: int = 4
    dropout_rate: float = 0.3
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset
class NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y_float01: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_float01, dtype=torch.float32)  # [N], 0/1

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model
class ParticleClassifierBinary(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=512, num_layers=4, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        h = hidden_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
            h = max(h // 2, 8)

        layers.append(nn.Linear(in_dim, 1))  # single logit
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)  # [B]


# Utilities
def equalize_pos_neg(X_pos: np.ndarray, X_neg: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    n = min(len(X_pos), len(X_neg))

    ip = rng.choice(len(X_pos), n, replace=False)
    ineg = rng.choice(len(X_neg), n, replace=False)

    Xp = X_pos[ip]
    Xn = X_neg[ineg]
    return Xp, Xn

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 2D (N, F). Got shape={arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr.astype(np.float32)


def split_three_way(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    N = len(X)
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    return X_train, y_train, X_val, y_val, X_test, y_test


def standardize_fit_transform(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0, keepdims=True)
    sig = X_train.std(axis=0, keepdims=True)
    sig = np.where(sig == 0, 1.0, sig)
    X_train2 = (X_train - mu) / sig
    X_val2 = (X_val - mu) / sig
    X_test2 = (X_test - mu) / sig
    scaler = {"mean": mu.astype(np.float32), "std": sig.astype(np.float32)}
    return X_train2, X_val2, X_test2, scaler


@torch.no_grad()
def predict_prob(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(x)
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


@torch.no_grad()
def eval_binary(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_prob = []
    loss_sum = 0.0
    n = 0
    criterion = nn.BCEWithLogitsLoss()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += float(loss.item()) * xb.size(0)
        n += xb.size(0)

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.append(prob)
        y_true.append(yb.detach().cpu().numpy())

    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)

    # accuracy at 0.5 (for monitoring)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0

    # AUC can fail if one class missing (rare but possible on tiny val)
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    return {"loss": loss_sum / max(n, 1), "acc@0.5": acc, "auc": auc}


def train_binary_task(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    cfg: TrainConfig,
    out_dir: Path,
    run_name: str,
) -> Dict[str, object]:
    """
    Trains: positive vs negative (pos label=1, neg label=0).
    Saves best checkpoint by val_loss with metrics JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    X_pos, X_neg = equalize_pos_neg(X_pos, X_neg, cfg.seed)

    assert X_pos.shape[1] == X_neg.shape[1], "POS/NEG feature dims differ!"
    print(f"[INFO] equalized: N_pos={len(X_pos)} N_neg={len(X_neg)} (each)")

    # build labels
    y_pos = np.ones(len(X_pos), dtype=np.int64)
    y_neg = np.zeros(len(X_neg), dtype=np.int64)
    X_all = np.concatenate([X_pos, X_neg], axis=0)
    y_all = np.concatenate([y_pos, y_neg], axis=0)

    # split
    X_train, y_train, X_val, y_val, X_test, y_test = split_three_way(
        X_all, y_all, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed
    )

    scaler = None
    if cfg.standardize:
        X_train, X_val, X_test, scaler = standardize_fit_transform(X_train, X_val, X_test)

    # loaders
    train_ds = NpyDataset(X_train, y_train.astype(np.float32))
    val_ds   = NpyDataset(X_val,   y_val.astype(np.float32))
    test_ds  = NpyDataset(X_test,  y_test.astype(np.float32))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # model
    input_dim = X_train.shape[1]
    model = ParticleClassifierBinary(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    best_ckpt = out_dir / f"{run_name}.best.pth"

    history = {"train": [], "val": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0

        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss_sum += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        tr_loss = tr_loss_sum / max(tr_n, 1)
        val_metrics = eval_binary(model, val_loader, cfg.device)

        history["train"].append({"epoch": epoch, "loss": tr_loss})
        history["val"].append({"epoch": epoch, **val_metrics})

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]

            payload = {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "cfg": asdict(cfg),
                "scaler": scaler,               # None or {mean,std}
                "best_val": best_val,
                "val_metrics": val_metrics,
                "run_name": run_name,
            }
            torch.save(payload, best_ckpt)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"[{run_name}] epoch {epoch:03d}/{cfg.epochs} "
                f"train_loss={tr_loss:.6f}  val_loss={val_metrics['loss']:.6f} "
                f"val_acc@0.5={val_metrics['acc@0.5']:.3f} val_auc={val_metrics['auc']:.3f}"
            )

    # load best and evaluate test
    best = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(best["state_dict"])
    model.to(cfg.device).eval()

    test_metrics = eval_binary(model, test_loader, cfg.device)

    # save metrics summary JSON
    summary = {
        "run_name": run_name,
        "best_ckpt": str(best_ckpt),
        "best_val_loss": float(best["best_val"]),
        "val_metrics_at_best": best["val_metrics"],
        "test_metrics": test_metrics,
        "cfg": best["cfg"],
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
    }
    (out_dir / f"{run_name}.metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{run_name}.history.json").write_text(json.dumps(history, indent=2))

    return {
        "summary": summary,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "model": model,  # trained best-loaded model (on device)
    }

