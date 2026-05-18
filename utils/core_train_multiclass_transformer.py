# utils/core_train_multiclass_transformer.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Re-use all shared data utilities from the DNN core
from utils.core_train_multiclass import (
    NpyDatasetMulticlass,
    equalize_classes,
    split_balanced_per_class_multiclass,
    split_with_fixed_test_counts,
    standardize_fit_transform,
    compute_class_weights,
    eval_multiclass,
    predict_proba,
    predict_class,
    set_seed,
    load_npy,
)


# -------------------------
# Config
# -------------------------
@dataclass
class TransformerTrainConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    epochs: int = 600
    lr: float = 5e-4
    lr_min: float = 1e-6        # cosine annealing floor
    batch_size: int = 1024
    seed: int = 42
    standardize: bool = True
    # Transformer-specific
    d_model: int = 128          # token embedding dimension (head_dim = d_model / n_heads)
    n_heads: int = 4            # attention heads → head_dim = 32
    n_encoder_layers: int = 4   # number of TransformerEncoderLayer blocks
    dim_feedforward: int = 512  # inner FFN dimension (keep 4× d_model)
    dropout_rate: float = 0.1
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Model
# -------------------------

class _SafeEncoderLayer(nn.TransformerEncoderLayer):
    """
    Bypasses all fused CUDA fast-paths in PyTorch >= 2.1 that fail with
    small head_dim (e.g. d_model=64, n_heads=4 → head_dim=16).

    PyTorch 2.x has two independent fast-paths that both call broken kernels:
      1. TransformerEncoderLayer.forward → torch._transformer_encoder_layer_fwd
         Fix: override forward() to call _sa_block/_ff_block directly.
      2. MultiheadAttention.forward → torch._native_multi_head_attention
         Triggered when need_weights=False (the default in _sa_block).
         Fix: override _sa_block() to pass need_weights=True, which forces
         the standard scaled-dot-product path. Weights are discarded.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # need_weights=True bypasses torch._native_multi_head_attention.
        # The returned weight tensor is discarded; only the output is used.
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )
        return self.dropout1(x)


class ParticleTransformerMulticlass(nn.Module):
    """
    Feature-tokenization Transformer for tabular particle physics data.

    Each of the `input_dim` scalar features is treated as a separate token:
      - Feature-specific linear projection:  scalar → d_model
      - Learned positional embedding per feature index
      - Transformer encoder (batch_first, pre-norm for stability)
      - Mean pooling over the feature-token sequence
      - Linear classification head

    Input  : (B, input_dim)  float32
    Output : (B, num_classes) logits
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.input_dim = input_dim

        # One linear projection per feature (heterogeneous features deserve
        # independent weights rather than a shared embedding matrix).
        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(input_dim)]
        )

        # Learned positional embedding: each feature index gets its own vector.
        self.pos_embed = nn.Embedding(input_dim, d_model)

        # Transformer encoder with pre-LayerNorm (norm_first) for stability.
        # Use _SafeEncoderLayer to bypass torch._transformer_encoder_layer_fwd,
        # a fused CUDA kernel that fails with small head_dim (< 32) on some GPUs.
        encoder_layer = _SafeEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        # Buffer for feature position indices (avoids re-creating each forward pass).
        self.register_buffer("_positions", torch.arange(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        # Project each feature independently → stack into token sequence
        tokens = torch.stack(
            [self.feature_projections[i](x[:, i : i + 1]) for i in range(self.input_dim)],
            dim=1,
        )  # (B, input_dim, d_model)

        # Add positional embeddings (broadcast over batch)
        tokens = tokens + self.pos_embed(self._positions).unsqueeze(0)  # (B, F, d_model)

        # Transformer encoder
        out = self.transformer(tokens)      # (B, F, d_model)
        out = self.norm(out)

        # Mean pooling over the feature-token sequence
        out = out.mean(dim=1)               # (B, d_model)

        return self.classifier(out)         # (B, num_classes)


# -------------------------
# Training
# -------------------------
def train_multiclass_transformer(
    Xs: List[np.ndarray],
    cfg: TransformerTrainConfig,
    out_dir: Path,
    run_name: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Train K-class Transformer classifier from per-class feature arrays Xs.
    API is identical to core_train_multiclass.train_multiclass_task.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    K = len(Xs)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(K)]
    if len(class_names) != K:
        raise ValueError("class_names length must match number of classes in Xs")

    # Equalize class counts then split
    Xs_eq = equalize_classes(Xs, seed=cfg.seed)
    n_each = len(Xs_eq[0])
    for i, Xc in enumerate(Xs_eq):
        if Xc.ndim != 2:
            raise ValueError(f"Class {i} array must be 2D (N,F). Got {Xc.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_balanced_per_class_multiclass(
        Xs_eq, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed
    )

    print(f"[INFO] classes={K}  each_class_n={n_each}")
    print(f"[INFO] splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    def frac_by_class(y):
        y = np.asarray(y).astype(int)
        return {name: float((y == i).mean()) if len(y) else float("nan")
                for i, name in enumerate(class_names)}

    print(f"[INFO] train frac by class: {frac_by_class(y_train)}")
    print(f"[INFO] val   frac by class: {frac_by_class(y_val)}")
    print(f"[INFO] test  frac by class: {frac_by_class(y_test)}")

    scaler = None
    if cfg.standardize:
        X_train, X_val, X_test, scaler = standardize_fit_transform(X_train, X_val, X_test)

    train_loader = DataLoader(
        NpyDatasetMulticlass(X_train, y_train),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        NpyDatasetMulticlass(X_val, y_val),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        NpyDatasetMulticlass(X_test, y_test),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    input_dim = X_train.shape[1]
    model = ParticleTransformerMulticlass(
        input_dim=input_dim,
        num_classes=K,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Transformer params: {n_params:,}")
    print(f"[INFO] d_model={cfg.d_model} n_heads={cfg.n_heads} "
          f"n_encoder_layers={cfg.n_encoder_layers} dim_ff={cfg.dim_feedforward}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
    )

    best_val = float("inf")
    best_epoch = 0
    best_ckpt = out_dir / f"{run_name}.best.pth"
    progress_file = out_dir / f"{run_name}.progress.json"
    history: Dict[str, list] = {"train": [], "val": []}

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
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train"].append({"epoch": epoch, "loss": tr_loss, "lr": current_lr})
        history["val"].append({"epoch": epoch,
                                **{k: v for k, v in val_metrics.items()
                                   if k != "confusion_matrix"}})

        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            payload = {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": K,
                "class_names": class_names,
                "cfg": asdict(cfg),
                "scaler": scaler,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "val_metrics": val_metrics,
                "run_name": run_name,
                "model_type": "transformer",
            }
            torch.save(payload, best_ckpt)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"[{run_name}] epoch {epoch:03d}/{cfg.epochs} "
                f"train_loss={tr_loss:.6f}  val_loss={val_metrics['loss']:.6f} "
                f"val_acc={val_metrics['acc']:.3f} val_macroF1={val_metrics['macro_f1']:.3f} "
                f"lr={current_lr:.2e}  best_epoch={best_epoch}",
                flush=True,
            )
            progress_file.write_text(json.dumps({
                "epoch": epoch,
                "total_epochs": cfg.epochs,
                "train_loss": tr_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": current_lr,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
            }, indent=2))

    # Load best checkpoint and evaluate on test set
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
        "n_train": int(len(train_loader.dataset)),
        "n_val": int(len(val_loader.dataset)),
        "n_test": int(len(test_loader.dataset)),
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
