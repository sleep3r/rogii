"""PathFormer training loop.

Usage:
    python -m pathformer.train --config configs/pathformer_v0.yml
    python -m pathformer.train --config configs/pathformer_v0.yml --epochs 1  # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from .config import PathFormerConfig, load_config
from .dataset import (
    WellSample,
    WellDataset,
    load_all_wells,
    make_tail_balanced_indices,
)
from .evaluate import evaluate_samples, log_metrics, predict_row_predictions, save_metrics
from .model import PathFormer, pathformer_loss, count_parameters


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Train/valid split
# ---------------------------------------------------------------------------

def split_wells(
    samples: list[WellSample],
    valid_fraction: float,
    seed: int,
) -> tuple[list[WellSample], list[WellSample]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_valid = max(1, int(round(len(indices) * valid_fraction)))
    valid_idx = indices[:n_valid].tolist()
    train_idx = indices[n_valid:].tolist()
    return [samples[i] for i in train_idx], [samples[i] for i in valid_idx]


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: PathFormer,
    dataset: WellDataset,
    sample_indices: list[int],
    optimizer: torch.optim.Optimizer,
    cfg: PathFormerConfig,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    model.train() if train else model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=SubsetRandomSampler(sample_indices) if train else SubsetRandomSampler(sample_indices),
        num_workers=0,
        collate_fn=_collate,
    )

    total_loss = 0.0
    total_huber = 0.0
    total_smooth = 0.0
    n_batches = 0

    ctx = torch.no_grad() if not train else _noop_ctx()

    with ctx:
        for batch in loader:
            features = batch["features"].to(device)      # (B, L, F)
            target = batch["target_delta"].to(device)    # (B, L)
            hidden_mask = batch["hidden_mask"].to(device) # (B, L)
            pad_mask = batch["pad_mask"].to(device)       # (B, L)

            if train:
                optimizer.zero_grad()

            pred_delta = model(features, pad_mask)       # (B, L)

            loss, info = pathformer_loss(
                pred_delta, target, hidden_mask, pad_mask,
                huber_delta=cfg.train.huber_delta,
                smooth_lambda=cfg.train.smooth_lambda,
            )

            if train and torch.isfinite(loss) and loss > 0:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                optimizer.step()

            total_loss += info["total"]
            total_huber += info["huber"]
            total_smooth += info["smooth"]
            n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "huber": total_huber / denom,
        "smooth": total_smooth / denom,
    }


class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _collate(batch: list[dict]) -> dict:
    """Collate a list of well dicts; pad to the longest sequence in batch."""
    max_len = max(int(item["seq_len"].item()) for item in batch)
    # Re-pad each item to max_len (they were padded to max_seq_len in dataset,
    # but here we collate with the actual max in this batch for memory efficiency)
    keys_to_stack = ["features", "target_delta", "hidden_mask", "pad_mask"]
    out: dict = {}

    for key in keys_to_stack:
        tensors = []
        for item in batch:
            t = item[key]
            seq = item["seq_len"].item()
            # slice to actual seq_len, then re-pad to max_len
            t_trimmed = t[:seq]
            if key == "features":
                pad_shape = (max_len - seq, t.shape[-1])
                padding = torch.zeros(pad_shape, dtype=t.dtype)
                t_padded = torch.cat([t_trimmed, padding], dim=0)
            else:
                pad_shape = (max_len - seq,)
                if key == "hidden_mask" or key == "pad_mask":
                    padding = torch.ones(pad_shape, dtype=t.dtype) if key == "pad_mask" else torch.zeros(pad_shape, dtype=t.dtype)
                else:
                    padding = torch.zeros(pad_shape, dtype=t.dtype)
                t_padded = torch.cat([t_trimmed, padding], dim=0)
            tensors.append(t_padded)
        out[key] = torch.stack(tensors, dim=0)

    out["seq_len"] = torch.stack([item["seq_len"] for item in batch])
    out["well_id"] = [item["well_id"] for item in batch]
    out["last_known_tvt"] = torch.stack([item["last_known_tvt"] for item in batch])
    return out


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: PathFormerConfig, override_epochs: int | None = None) -> None:
    epochs = override_epochs if override_epochs is not None else cfg.train.epochs
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Save resolved config
    with open(out_dir / "config_resolved.json", "w") as f:
        json.dump(_config_to_dict(cfg), f, indent=2, default=str)

    # Seed
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    device = resolve_device(cfg.train.device)
    print(f"[train] device={device}", flush=True)

    # Load data
    t0 = time.time()
    all_samples, _tail_df = load_all_wells(cfg)
    print(f"[train] data loaded in {time.time()-t0:.1f}s  ({len(all_samples)} wells)", flush=True)

    if not all_samples:
        raise RuntimeError("No wells loaded — check data_dir and configuration")

    # Split
    train_samples, valid_samples = split_wells(all_samples, cfg.train.valid_fraction, cfg.train.seed)
    print(f"[train] train={len(train_samples)}  valid={len(valid_samples)}", flush=True)

    # Print tail class distribution
    from collections import Counter
    tc = Counter(s.tail_class for s in train_samples)
    print(f"[train] tail classes (train): {dict(tc)}", flush=True)

    # Datasets
    train_dataset = WellDataset(
        train_samples,
        max_seq_len=cfg.data.max_seq_len,
        augment=True,
        aug_cfg=cfg.augmentation,
        seed=cfg.train.seed,
    )
    valid_dataset = WellDataset(
        valid_samples,
        max_seq_len=cfg.data.max_seq_len,
        augment=False,
    )

    # Model
    model = PathFormer(cfg.model).to(device)
    print(f"[train] model params: {count_parameters(model):,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.train.lr * 0.1
    )

    rng = np.random.default_rng(cfg.train.seed)
    best_valid_rmse = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # Tail-balanced sampling for training
        train_indices = make_tail_balanced_indices(
            train_samples,
            oversample_factor=cfg.train.tail_oversample_factor,
            rng=rng,
        )
        valid_indices = list(range(len(valid_samples)))

        train_stats = run_epoch(model, train_dataset, train_indices, optimizer, cfg, device, train=True)
        valid_stats = run_epoch(model, valid_dataset, valid_indices, optimizer, cfg, device, train=False)
        scheduler.step()

        ep_time = time.time() - t_ep

        eval_metrics = evaluate_samples(
            model, valid_samples, cfg.data.max_seq_len, device
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_huber": train_stats["huber"],
            "valid_loss": valid_stats["loss"],
            "valid_huber": valid_stats["huber"],
            "epoch_seconds": ep_time,
            **{f"eval_{k}": v for k, v in eval_metrics.items() if not isinstance(v, dict)},
        }
        history.append(row)

        print(
            f"[epoch {epoch:3d}/{epochs}] "
            f"train_loss={train_stats['loss']:.4f}  "
            f"valid_loss={valid_stats['loss']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"t={ep_time:.0f}s",
            flush=True,
        )
        if eval_metrics:
            log_metrics(eval_metrics, prefix=f"eval e{epoch}")

        valid_score = float(eval_metrics.get("row_rmse_ft", valid_stats["loss"]))
        if valid_score < best_valid_rmse:
            best_valid_rmse = valid_score
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "valid_row_rmse_ft": valid_score,
                "config": _config_to_dict(cfg),
            }, ckpt_dir / "best.pt")
            print(
                f"  -> saved best checkpoint (row_rmse_ft={valid_score:.4f})",
                flush=True,
            )

        if epoch % cfg.train.save_every_n_epochs == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    # Save training history
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on full valid set with best checkpoint
    print("\n[train] loading best checkpoint for final evaluation...", flush=True)
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    final_metrics = evaluate_samples(model, valid_samples, cfg.data.max_seq_len, device)
    print("\n=== Final evaluation (best checkpoint) ===")
    log_metrics(final_metrics, prefix="final")
    save_metrics(final_metrics, out_dir / "eval_metrics.json")
    row_predictions = predict_row_predictions(
        model,
        valid_samples,
        cfg.data.max_seq_len,
        device,
        candidate="pathformer_direct",
    )
    row_predictions.to_parquet(out_dir / "pathformer_row_predictions.parquet", index=False)

    print(f"\n[train] done. Output: {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Config serialization helper
# ---------------------------------------------------------------------------

def _config_to_dict(cfg: PathFormerConfig) -> dict:
    import dataclasses
    def _convert(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, Path):
            return str(obj)
        return obj
    return _convert(cfg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PathFormer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, override_epochs=args.epochs)


if __name__ == "__main__":
    main()
