# utils/core_train_multiclass.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    epochs: int = 200
    lr: float = 5e-4
    batch_size: int = 1024
    seed: int = 42
    standardize: bool = False
    hidden_dim: int = 512
    num_layers: int = 4
    dropout_rate: float = 0.3
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Dataset
# -------------------------
class NpyDatasetMulticlass(Dataset):
    def __init__(self, X: np.ndarray, y_int: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_int, dtype=torch.long)  # [N], 0..K-1

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# Splits / balancing
# -------------------------
def split_three_way_stratified(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    val_size = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=seed, stratify=y_tmp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def equalize_classes(
    Xs: List[np.ndarray],
    seed: int,
) -> List[np.ndarray]:
    """
    Downsample each class array to the minimum class count.
    """
    rng = np.random.default_rng(seed)
    nmin = min(len(X) for X in Xs)
    out = []
    for X in Xs:
        if len(X) == nmin:
            out.append(X)
        else:
            idx = rng.choice(len(X), nmin, replace=False)
            out.append(X[idx])
    return out


def split_balanced_per_class_multiclass(
    Xs: List[np.ndarray],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Xs: list of arrays, one per class, already equalized length.
    Returns balanced splits across classes.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    rng = np.random.default_rng(seed)
    K = len(Xs)
    n = len(Xs[0])
    for k in range(K):
        if len(Xs[k]) != n:
            raise ValueError("All class arrays must have same length after equalization.")

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list = [], []

    for cls, Xc in enumerate(Xs):
        Xc2 = Xc.copy()
        rng.shuffle(Xc2)

        tr = Xc2[:n_train]
        va = Xc2[n_train:n_train + n_val]
        te = Xc2[n_train + n_val:]

        X_tr_list.append(tr)
        y_tr_list.append(np.full(len(tr), cls, dtype=np.int64))

        X_va_list.append(va)
        y_va_list.append(np.full(len(va), cls, dtype=np.int64))

        X_te_list.append(te)
        y_te_list.append(np.full(len(te), cls, dtype=np.int64))

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)

    X_val = np.concatenate(X_va_list, axis=0)
    y_val = np.concatenate(y_va_list, axis=0)

    X_test = np.concatenate(X_te_list, axis=0)
    y_test = np.concatenate(y_te_list, axis=0)

    def shuffle_xy(X, y):
        idx = rng.permutation(len(X))
        return X[idx], y[idx]

    X_train, y_train = shuffle_xy(X_train, y_train)
    X_val, y_val = shuffle_xy(X_val, y_val)
    X_test, y_test = shuffle_xy(X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_with_fixed_test_counts(
    Xs: List[np.ndarray],
    test_counts: List[int],
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data with fixed test counts per class.
    Remaining data split into train/val based on val_frac.
    
    Args:
        Xs: List of feature arrays, one per class
        test_counts: List of exact test sample counts per class
        val_frac: Fraction of remaining data to use for validation
        seed: Random seed
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    rng = np.random.default_rng(seed)
    K = len(Xs)

    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list = [], []

    for cls, (Xc, n_test) in enumerate(zip(Xs, test_counts)):
        n_total = len(Xc)
        if n_test > n_total:
            raise ValueError(
                f"Class {cls}: requested {n_test} test samples "
                f"but only {n_total} available"
            )

        # Shuffle
        idx = rng.permutation(n_total)
        Xc_shuffled = Xc[idx]

        # Reserve test
        X_test = Xc_shuffled[:n_test]
        X_remaining = Xc_shuffled[n_test:]

        # Split remaining into train/val
        n_remaining = len(X_remaining)
        n_val = int(val_frac * n_remaining)

        X_val = X_remaining[:n_val]
        X_train = X_remaining[n_val:]

        X_tr_list.append(X_train)
        y_tr_list.append(np.full(len(X_train), cls, dtype=np.int64))

        X_va_list.append(X_val)
        y_va_list.append(np.full(len(X_val), cls, dtype=np.int64))

        X_te_list.append(X_test)
        y_te_list.append(np.full(len(X_test), cls, dtype=np.int64))

    # Concatenate all classes
    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    X_val = np.concatenate(X_va_list, axis=0)
    y_val = np.concatenate(y_va_list, axis=0)
    X_test = np.concatenate(X_te_list, axis=0)
    y_test = np.concatenate(y_te_list, axis=0)

    # Shuffle each split
    def shuffle_xy(X, y):
        idx = rng.permutation(len(X))
        return X[idx], y[idx]

    X_train, y_train = shuffle_xy(X_train, y_train)
    X_val, y_val = shuffle_xy(X_val, y_val)
    X_test, y_test = shuffle_xy(X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------
# Model
# -------------------------
class ParticleClassifierMulticlass(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout_rate: float = 0.3,
    ):
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

        layers.append(nn.Linear(in_dim, num_classes))  # logits for K classes
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # [B, K]


# -------------------------
# Utilities
# -------------------------
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


def standardize_fit_transform(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0, keepdims=True)
    sig = X_train.std(axis=0, keepdims=True)
    sig = np.where(sig == 0, 1.0, sig)
    X_train2 = (X_train - mu) / sig
    X_val2 = (X_val - mu) / sig
    X_test2 = (X_test - mu) / sig
    scaler = {"mean": mu.astype(np.float32), "std": sig.astype(np.float32)}
    return X_train2, X_val2, X_test2, scaler


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse frequency weights for balanced loss.
    
    Args:
        y_train: Training labels
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights normalized to sum to num_classes
    """
    counts = np.bincount(y_train, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


@torch.no_grad()
def predict_class(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    probs = predict_proba(model, X, device)
    return probs.argmax(axis=1).astype(np.int64)


@torch.no_grad()
def eval_multiclass(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    loss_sum = 0.0
    n = 0

    y_true_list = []
    y_pred_list = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)                 # [B,K]
        loss = criterion(logits, yb)       # scalar

        loss_sum += float(loss.item()) * xb.size(0)
        n += xb.size(0)

        pred = torch.argmax(logits, dim=1)
        y_true_list.append(yb.detach().cpu().numpy())
        y_pred_list.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([], dtype=np.int64)

    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))) if len(y_true) else np.zeros((num_classes, num_classes), dtype=int)

    return {
        "loss": loss_sum / max(n, 1),
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }


# -------------------------
# Training
# -------------------------
def train_multiclass_task(
    Xs: List[np.ndarray],
    cfg: TrainConfig,
    out_dir: Path,
    run_name: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Trains K-class classifier from per-class feature arrays Xs.
    Saves best checkpoint by val_loss with metrics JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    K = len(Xs)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(K)]
    if len(class_names) != K:
        raise ValueError("class_names length must match number of classes in Xs")

    # equalize class counts
    Xs_eq = equalize_classes(Xs, seed=cfg.seed)
    n_each = len(Xs_eq[0])
    for i, Xc in enumerate(Xs_eq):
        if Xc.ndim != 2:
            raise ValueError(f"Class {i} array must be 2D (N,F). Got {Xc.shape}")

    # split balanced
    X_train, y_train, X_val, y_val, X_test, y_test = split_balanced_per_class_multiclass(
        Xs_eq, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed
    )

    print(f"[INFO] classes={K}  each_class_n={n_each}")
    print(f"[INFO] splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    # quick sanity class fractions
    def frac_by_class(y):
        y = np.asarray(y).astype(int)
        out = {}
        for i, name in enumerate(class_names):
            out[name] = float((y == i).mean()) if len(y) else float("nan")
        return out
    print(f"[INFO] train frac by class: {frac_by_class(y_train)}")
    print(f"[INFO] val   frac by class: {frac_by_class(y_val)}")
    print(f"[INFO] test  frac by class: {frac_by_class(y_test)}")

    scaler = None
    if cfg.standardize:
        X_train, X_val, X_test, scaler = standardize_fit_transform(X_train, X_val, X_test)

    # loaders
    train_ds = NpyDatasetMulticlass(X_train, y_train)
    val_ds   = NpyDatasetMulticlass(X_val,   y_val)
    test_ds  = NpyDatasetMulticlass(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # model
    input_dim = X_train.shape[1]
    model = ParticleClassifierMulticlass(
        input_dim=input_dim,
        num_classes=K,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

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
        val_metrics = eval_multiclass(model, val_loader, cfg.device, num_classes=K)

        history["train"].append({"epoch": epoch, "loss": tr_loss})
        history["val"].append({"epoch": epoch, **{k: v for k, v in val_metrics.items() if k != "confusion_matrix"}})

        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            payload = {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": K,
                "class_names": class_names,
                "cfg": asdict(cfg),
                "scaler": scaler,
                "best_val": best_val,
                "val_metrics": val_metrics,
                "run_name": run_name,
            }
            torch.save(payload, best_ckpt)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"[{run_name}] epoch {epoch:03d}/{cfg.epochs} "
                f"train_loss={tr_loss:.6f}  val_loss={val_metrics['loss']:.6f} "
                f"val_acc={val_metrics['acc']:.3f} val_macroF1={val_metrics['macro_f1']:.3f}"
            )

    # load best and evaluate test
    best = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    model.to(cfg.device).eval()

    test_metrics = eval_multiclass(model, test_loader, cfg.device, num_classes=K)

    summary = {
        "run_name": run_name,
        "best_ckpt": str(best_ckpt),
        "best_val_loss": float(best["best_val"]),
        "val_metrics_at_best": best["val_metrics"],
        "test_metrics": test_metrics,
        "cfg": best["cfg"],
        "class_names": best["class_names"],
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
        "model": model,
        "class_names": class_names,
    }


def train_multiclass_weighted(
    Xs: List[np.ndarray],
    test_counts: List[int],
    cfg: TrainConfig,
    out_dir: Path,
    run_name: str,
    class_names: List[str],
) -> Dict[str, object]:
    """
    Train multiclass classifier with class weights and fixed test counts.
    
    Args:
        Xs: List of feature arrays, one per class
        test_counts: List of exact test sample counts per class
        cfg: Training configuration
        out_dir: Output directory for checkpoints
        run_name: Name for this run
        class_names: Names of classes
        
    Returns:
        Dictionary with summary, test data, model, etc.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(Xs)

    # Split with fixed test counts
    X_train, y_train, X_val, y_val, X_test, y_test = split_with_fixed_test_counts(
        Xs, test_counts, val_frac=cfg.val_frac, seed=cfg.seed
    )

    print(f"\n[INFO] Classes: {class_names}")
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Standardize
    scaler = None
    if cfg.standardize:
        X_train, X_val, X_test, scaler = standardize_fit_transform(X_train, X_val, X_test)

    # Compute class weights
    class_weights = compute_class_weights(y_train, K)
    print(f"\n[INFO] Class weights: {dict(zip(class_names, class_weights.tolist()))}")

    # Datasets and loaders
    train_ds = NpyDatasetMulticlass(X_train, y_train)
    val_ds = NpyDatasetMulticlass(X_val, y_val)
    test_ds = NpyDatasetMulticlass(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    input_dim = X_train.shape[1]
    model = ParticleClassifierMulticlass(
        input_dim=input_dim,
        num_classes=K,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)

    # Weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_ckpt = out_dir / f"{run_name}.best.pth"
    history = {"train": [], "val": []}

    print(f"\n[INFO] Training for {cfg.epochs} epochs...")
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
        val_metrics = eval_multiclass(model, val_loader, cfg.device, K)

        history["train"].append({"epoch": epoch, "loss": tr_loss})
        history["val"].append({"epoch": epoch, **{k: v for k, v in val_metrics.items() if k != "confusion_matrix"}})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            payload = {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": K,
                "class_names": class_names,
                "class_weights": class_weights.tolist(),
                "test_counts": test_counts,
                "cfg": asdict(cfg),
                "scaler": scaler,
                "best_val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "run_name": run_name,
            }
            torch.save(payload, best_ckpt)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"train_loss={tr_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['acc']:.3f} | "
                f"val_f1={val_metrics['macro_f1']:.3f}"
            )

    # Load best and test
    best = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    model.to(cfg.device).eval()

    test_metrics = eval_multiclass(model, test_loader, cfg.device, K)

    summary = {
        "run_name": run_name,
        "best_ckpt": str(best_ckpt),
        "best_val_loss": float(best_val_loss),
        "class_names": class_names,
        "class_weights": class_weights.tolist(),
        "test_counts": test_counts,
        "val_metrics_at_best": best["val_metrics"],
        "test_metrics": test_metrics,
        "cfg": asdict(cfg),
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
        "model": model,
        "class_names": class_names,
    }
