"""RAC-Former v1 training script.

5-fold GroupKFold by well_id.
AdamW + cosine LR with warmup.
EMA weights after ema_start_epoch.
Early stopping on pooled row-RMSE.
Pseudo-anchor augmentation + GR/prior dropouts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .clearml_data import resolve_data_dir
from .clearml_tracking import ClearMLTracker, NullTracker
from .config import RACFormerConfig, config_to_dict, load_config
from .dataset import RACDataset, load_all_wells
from .loss import RACLoss, compute_oof_rmse
from .model import RACFormer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(cfg: RACFormerConfig) -> torch.device:
    spec = cfg.train.device
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _device_label(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(device)
        return f"cuda:{torch.cuda.current_device()} {name}"
    return str(device)


def _cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def _dataloader_kwargs(batch_size: int, num_workers: int, device: torch.device) -> dict:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def _format_perf_suffix(
    train_dt: float,
    val_dt: float,
    ema_dt: float,
    train_batches: int,
    val_batches: int,
    cuda_mem_mb: float,
) -> str:
    mem = f" | cuda_mem={cuda_mem_mb:.0f}MB" if cuda_mem_mb > 0 else ""
    return (
        f" | train_dt={train_dt:.1f}s"
        f" | val_dt={val_dt:.1f}s"
        f" | ema_dt={ema_dt:.1f}s"
        f" | batches={train_batches}/{val_batches}"
        f"{mem}"
    )


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.num_updates == 0:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    )
        self.num_updates += 1

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


def _backup_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# GroupKFold split
# ---------------------------------------------------------------------------

def _group_kfold_splits(
    samples: list,
    n_folds: int,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Return list of (train_indices, val_indices) tuples, grouped by well_id."""
    # Collect unique well_ids in order
    well_ids = [s.well_id for s in samples]
    unique_wells = list(dict.fromkeys(well_ids))  # preserves insertion order
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_wells)

    # Assign each well to a fold
    fold_assignments = {wid: i % n_folds for i, wid in enumerate(unique_wells)}

    splits = []
    for fold in range(n_folds):
        train_idx = [i for i, s in enumerate(samples) if fold_assignments[s.well_id] != fold]
        val_idx = [i for i, s in enumerate(samples) if fold_assignments[s.well_id] == fold]
        splits.append((train_idx, val_idx))

    return splits


def _train_val_splits(samples: list, cfg: RACFormerConfig) -> list[tuple[list[int], list[int]]]:
    """Return train/val index splits, or a one-fold same-sample debug split."""
    if cfg.train.debug_overfit_samples > 0:
        n_debug = min(cfg.train.debug_overfit_samples, len(samples))
        debug_idx = list(range(n_debug))
        return [(debug_idx, debug_idx)]
    return _group_kfold_splits(samples, cfg.train.n_folds, cfg.train.seed)


def _optimizer_steps_per_epoch(train_batches: int, grad_accum: int) -> int:
    """Number of optimizer updates in one complete pass over the loader."""
    grad_accum = max(int(grad_accum), 1)
    return max((int(train_batches) + grad_accum - 1) // grad_accum, 1)


def _should_validate_step_mode(
    optimizer_steps_done: int,
    max_optimizer_steps: int,
    validate_every_steps: int,
) -> bool:
    """Whether step-based training should run validation now."""
    if max_optimizer_steps <= 0:
        return True
    if optimizer_steps_done >= max_optimizer_steps:
        return True
    if validate_every_steps <= 0:
        return True
    return optimizer_steps_done > 0 and optimizer_steps_done % validate_every_steps == 0


def _oracle_segment_prediction(model: RACFormer, batch: dict, out) -> torch.Tensor:
    """Materialize TVT using the dataset's segment oracle and zero direct residual."""
    original_s_pred = out.s_pred
    original_direct = out.direct_resid_step
    try:
        out.s_pred = batch["s_star"]
        out.direct_resid_step = torch.zeros_like(original_direct)
        return model.materialize(batch, out)
    finally:
        out.s_pred = original_s_pred
        out.direct_resid_step = original_direct


@torch.no_grad()
def _oracle_segment_rmse(model: RACFormer, batch: dict) -> float:
    """RMSE for s_star through the real materializer; diagnoses segment mismatch."""
    out = model(batch)
    pred = _oracle_segment_prediction(model, batch, out)
    rmse, _ = compute_oof_rmse(pred, batch["tvt_hidden"], batch["n_hidden_rows"])
    return rmse


def _oracle_direct_step_prediction(batch: dict) -> torch.Tensor:
    """Predict base + mean residual per hidden step."""
    base = batch["base_tvt_hidden"]
    true = batch["tvt_hidden"]
    n_hidden = batch["n_hidden_rows"]
    hidden_row_to_step = batch["hidden_row_to_step"]

    pred = base.clone()
    for b in range(base.shape[0]):
        h_b = int(n_hidden[b].item())
        if h_b <= 0:
            continue
        residual = true[b, :h_b] - base[b, :h_b]
        steps = hidden_row_to_step[b, :h_b]
        for step_id in torch.unique_consecutive(steps):
            mask = steps == step_id
            pred[b, :h_b][mask] = base[b, :h_b][mask] + residual[mask].mean()
    return pred


@torch.no_grad()
def _oracle_direct_step_rmse(batch: dict) -> float:
    """RMSE for an oracle step-wise direct residual; diagnoses row/step granularity."""
    pred = _oracle_direct_step_prediction(batch)
    rmse, _ = compute_oof_rmse(pred, batch["tvt_hidden"], batch["n_hidden_rows"])
    return rmse


# ---------------------------------------------------------------------------
# Batch to device
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model: RACFormer,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: RACLoss,
    device: torch.device,
    cfg: RACFormerConfig,
    grad_accum: int,
    scaler,  # GradScaler or None
    optimizer_steps_done: int = 0,
    max_optimizer_steps: int = 0,
    optimizer_steps_to_run: int = 0,
) -> tuple[dict[str, float], int]:
    model.train()
    accum: dict[str, float] = {}
    n_batches = 0
    optimizer_steps_delta = 0
    optimizer.zero_grad()

    while True:
        saw_batch = False
        for step, raw_batch in enumerate(loader):
            if max_optimizer_steps > 0 and optimizer_steps_done >= max_optimizer_steps:
                break
            if optimizer_steps_to_run > 0 and optimizer_steps_delta >= optimizer_steps_to_run:
                break

            saw_batch = True
            batch = _to_device(raw_batch, device)
            use_amp = scaler is not None

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch)
                pred_tvt = model.materialize(batch, out)
                losses = loss_fn(out, batch, pred_tvt)

            total = losses["total"] / grad_accum
            if scaler is not None:
                scaler.scale(total).backward()
            else:
                total.backward()

            # Accumulate metrics
            for k, v in losses.items():
                accum[k] = accum.get(k, 0.0) + float(v.item() if isinstance(v, torch.Tensor) else v)
            n_batches += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                optimizer_steps_done += 1
                optimizer_steps_delta += 1
                if max_optimizer_steps > 0 and optimizer_steps_done >= max_optimizer_steps:
                    break
                if optimizer_steps_to_run > 0 and optimizer_steps_delta >= optimizer_steps_to_run:
                    break

        if optimizer_steps_to_run <= 0:
            break
        if optimizer_steps_delta >= optimizer_steps_to_run:
            break
        if max_optimizer_steps > 0 and optimizer_steps_done >= max_optimizer_steps:
            break
        if not saw_batch:
            break

    metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}
    metrics["optimizer_steps"] = float(optimizer_steps_delta)
    return metrics, optimizer_steps_delta


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: RACFormer,
    loader: DataLoader,
    loss_fn: RACLoss,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Returns (pooled_row_rmse, metrics_dict)."""
    model.eval()
    all_pred, all_base, all_oracle_seg, all_oracle_direct, all_true, all_nhr = [], [], [], [], [], []
    metrics: dict[str, float] = {}
    n_batches = 0

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        out = model(batch)
        pred_tvt = model.materialize(batch, out)
        oracle_seg_tvt = _oracle_segment_prediction(model, batch, out)
        oracle_direct_tvt = _oracle_direct_step_prediction(batch)
        losses = loss_fn(out, batch, pred_tvt)

        all_pred.append(pred_tvt.cpu())
        all_base.append(batch["base_tvt_hidden"].cpu())
        all_oracle_seg.append(oracle_seg_tvt.cpu())
        all_oracle_direct.append(oracle_direct_tvt.cpu())
        all_true.append(batch["tvt_hidden"].cpu())
        all_nhr.append(batch["n_hidden_rows"].cpu())

        for k, v in losses.items():
            metrics[k] = metrics.get(k, 0.0) + float(v.item() if isinstance(v, torch.Tensor) else v)
        n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in metrics.items()}

    # Pooled RMSE (all hidden rows across all validation wells)
    pred_cat = torch.cat(all_pred, dim=0)
    base_cat = torch.cat(all_base, dim=0)
    oracle_seg_cat = torch.cat(all_oracle_seg, dim=0)
    oracle_direct_cat = torch.cat(all_oracle_direct, dim=0)
    true_cat = torch.cat(all_true, dim=0)
    nhr_cat = torch.cat(all_nhr, dim=0)
    pooled_rmse, _ = compute_oof_rmse(pred_cat, true_cat, nhr_cat)
    base_rmse, _ = compute_oof_rmse(base_cat, true_cat, nhr_cat)
    oracle_seg_rmse, _ = compute_oof_rmse(oracle_seg_cat, true_cat, nhr_cat)
    oracle_direct_rmse, _ = compute_oof_rmse(oracle_direct_cat, true_cat, nhr_cat)
    row_idx = torch.arange(pred_cat.shape[1]).unsqueeze(0)
    row_valid = row_idx < nhr_cat.unsqueeze(1)
    pred_delta_abs = (pred_cat - base_cat).abs()[row_valid]
    metrics["pooled_rmse"] = pooled_rmse
    metrics["base_rmse"] = base_rmse
    metrics["gain_vs_base"] = base_rmse - pooled_rmse
    metrics["oracle_seg_rmse"] = oracle_seg_rmse
    metrics["oracle_direct_step_rmse"] = oracle_direct_rmse
    metrics["pred_delta_abs_mean"] = float(pred_delta_abs.mean().item()) if pred_delta_abs.numel() else 0.0
    metrics["pred_delta_abs_max"] = float(pred_delta_abs.max().item()) if pred_delta_abs.numel() else 0.0

    return pooled_rmse, metrics


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def _train_fold(
    fold: int,
    train_samples: list,
    val_samples: list,
    cfg: RACFormerConfig,
    device: torch.device,
    output_dir: Path,
    tracker: ClearMLTracker | NullTracker | None = None,
) -> float:
    """Train one fold.  Returns best val RMSE."""
    if tracker is None:
        tracker = NullTracker()
    print(f"\n{'='*60}")
    print(f"  Fold {fold+1} | train={len(train_samples)} val={len(val_samples)}")
    print(f"{'='*60}")

    # ---- Datasets ----
    train_ds = RACDataset(
        train_samples,
        max_seq_len=cfg.data.max_seq_len,
        augment=cfg.train.debug_overfit_samples <= 0,
        aug_cfg=cfg.augmentation,
        seed=cfg.train.seed + fold * 1000,
        k_seg=cfg.model.k_seg,
        rows_per_step=cfg.data.rows_per_step,
    )
    val_ds = RACDataset(
        val_samples,
        max_seq_len=cfg.data.max_seq_len,
        augment=False,
        seed=cfg.train.seed,
        k_seg=cfg.model.k_seg,
        rows_per_step=cfg.data.rows_per_step,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        **_dataloader_kwargs(cfg.train.batch_size, cfg.train.num_workers, device),
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **_dataloader_kwargs(cfg.train.batch_size, cfg.train.num_workers, device),
    )

    # ---- Model + optimizer ----
    model = RACFormer(cfg.model).to(device)
    print(f"  Model parameters: {model.n_params:,}")
    print(f"  Train device: {_device_label(device)}")
    print(
        f"  Loader: train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={cfg.train.batch_size} num_workers={cfg.train.num_workers} "
        f"persistent_workers={cfg.train.num_workers > 0}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    steps_per_epoch = _optimizer_steps_per_epoch(len(train_loader), cfg.train.grad_accum)
    step_mode = cfg.train.max_optimizer_steps > 0
    step_interval = steps_per_epoch
    if step_mode and cfg.train.validate_every_steps > 0:
        step_interval = cfg.train.validate_every_steps
    epoch_limit = cfg.train.epochs
    if step_mode:
        epoch_limit = (cfg.train.max_optimizer_steps + step_interval - 1) // step_interval
    scheduler_t_max = max(epoch_limit - cfg.train.warmup_epochs, 1)

    # Cosine annealing from lr → lr_min over (epoch_limit - warmup) epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=cfg.train.lr_min,
    )

    loss_fn = RACLoss(cfg.train, cfg.model).to(device)
    ema = EMA(model, cfg.train.ema_decay)

    # Mixed precision (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ---- Training loop ----
    best_rmse = float("inf")
    best_epoch = 0
    best_optimizer_steps = 0
    no_improve = 0
    optimizer_steps_done = 0
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epoch_limit + 1):
        t0 = time.time()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Warmup LR (linear scale-up)
        if epoch <= cfg.train.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.train.lr * epoch / cfg.train.warmup_epochs

        # Train
        t_train0 = time.time()
        optimizer_steps_to_run = 0
        if step_mode:
            optimizer_steps_to_run = min(step_interval, cfg.train.max_optimizer_steps - optimizer_steps_done)
        train_metrics, optimizer_steps_delta = _train_epoch(
            model, train_loader, optimizer, loss_fn, device, cfg,
            cfg.train.grad_accum, scaler,
            optimizer_steps_done=optimizer_steps_done,
            max_optimizer_steps=cfg.train.max_optimizer_steps,
            optimizer_steps_to_run=optimizer_steps_to_run,
        )
        optimizer_steps_done += optimizer_steps_delta
        train_dt = time.time() - t_train0

        if step_mode and optimizer_steps_delta == 0:
            break

        # EMA update
        t_ema0 = time.time()
        if epoch >= cfg.train.ema_start_epoch:
            ema.update(model)
        ema_update_dt = time.time() - t_ema0

        # Step scheduler (skip warmup epochs)
        if epoch > cfg.train.warmup_epochs:
            scheduler.step()

        should_validate = True
        if step_mode:
            should_validate = _should_validate_step_mode(
                optimizer_steps_done=optimizer_steps_done,
                max_optimizer_steps=cfg.train.max_optimizer_steps,
                validate_every_steps=cfg.train.validate_every_steps,
            )
        if not should_validate:
            continue

        # Validation (with EMA weights if active)
        backup = None
        use_ema_for_eval = ema.num_updates > 0
        if use_ema_for_eval:
            backup = _backup_params(model)
            ema.apply_to(model)

        t_val0 = time.time()
        val_rmse, val_metrics = _validate(model, val_loader, loss_fn, device)
        val_dt = time.time() - t_val0

        ema_restore_dt = 0.0
        if backup is not None:
            t_restore0 = time.time()
            ema.restore(model, backup)
            ema_restore_dt = time.time() - t_restore0

        dt = time.time() - t0
        ema_dt = ema_update_dt + ema_restore_dt
        cuda_mem_mb = _cuda_memory_mb(device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Ep {epoch:3d}/{epoch_limit} | "
            f"opt_steps={optimizer_steps_done}"
            f"{('/' + str(cfg.train.max_optimizer_steps)) if step_mode else ''} | "
            f"lr={lr_now:.2e} | "
            f"train_loss={train_metrics.get('total', 0):.4f} | "
            f"train_rmse={train_metrics.get('row_rmse', 0):.2f} | "
            f"val_rmse={val_rmse:.4f} | "
            f"base_rmse={val_metrics.get('base_rmse', 0):.4f} | "
            f"gain={val_metrics.get('gain_vs_base', 0):+.4f} | "
            f"corr_abs={val_metrics.get('pred_delta_abs_mean', 0):.2f} | "
            f"seg_floor={val_metrics.get('oracle_seg_rmse', 0):.4f} | "
            f"direct_floor={val_metrics.get('oracle_direct_step_rmse', 0):.4f} | "
            f"seg_loss={val_metrics.get('seg', 0):.4f} | "
            f"dt={dt:.1f}s"
            f"{_format_perf_suffix(train_dt, val_dt, ema_dt, len(train_loader), len(val_loader), cuda_mem_mb)}"
        )

        # ---- ClearML per-epoch scalars ----
        fold_prefix = f"fold_{fold}"
        scalars: dict[str, float] = {
            f"{fold_prefix}/lr": lr_now,
            f"{fold_prefix}/epoch_seconds": dt,
            f"{fold_prefix}/train_seconds": train_dt,
            f"{fold_prefix}/val_seconds": val_dt,
            f"{fold_prefix}/ema_seconds": ema_dt,
            f"{fold_prefix}/cuda_peak_mem_mb": cuda_mem_mb,
            f"{fold_prefix}/train_batches": float(len(train_loader)),
            f"{fold_prefix}/val_batches": float(len(val_loader)),
            f"{fold_prefix}/val_rmse": val_rmse,
        }
        for k, v in train_metrics.items():
            scalars[f"{fold_prefix}/train_{k}"] = v
        for k, v in val_metrics.items():
            scalars[f"{fold_prefix}/val_{k}"] = v
        tracker.report_scalars(scalars, iteration=epoch)

        # Save best checkpoint
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_optimizer_steps = optimizer_steps_done
            no_improve = 0
            use_ema = ema.num_updates > 0
            ckpt = {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps_done,
                "model_state": model.state_dict(),
                "ema_shadow": ema.shadow if use_ema else {},
                "use_ema": use_ema,
                "ema_num_updates": ema.num_updates,
                "optimizer_state": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "cfg": config_to_dict(cfg),
            }
            torch.save(ckpt, fold_dir / "best.pt")
            tracker.report_scalar(f"{fold_prefix}/best_val_rmse", "value", val_rmse, epoch)
        else:
            no_improve += 1

        # Early stopping
        if step_mode and optimizer_steps_done >= cfg.train.max_optimizer_steps:
            break
        if no_improve >= cfg.train.early_stop_patience:
            print(f"  Early stop at epoch {epoch} (best={best_rmse:.4f} at ep {best_epoch})")
            break

    print(
        f"  Fold {fold+1} best val RMSE: {best_rmse:.4f} "
        f"(epoch {best_epoch}, opt_steps={best_optimizer_steps})"
    )
    return best_rmse


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(cfg: RACFormerConfig) -> None:
    """Full 5-fold training run."""
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Save config (resolved)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2, default=str)

    device = _get_device(cfg)
    print(f"Device: {device}")

    # ---- ClearML: resolve dataset (optional) ----
    try:
        resolved = resolve_data_dir(cfg.data_clearml)
    except Exception as exc:
        print(f"[clearml_data] resolve failed: {exc}", file=sys.stderr)
        resolved = None
    if resolved is not None:
        print(f"ClearML dataset resolved → {resolved}")
        cfg.data_dir = Path(resolved)

    # ---- ClearML: start experiment tracking (optional) ----
    tracker: ClearMLTracker | NullTracker = NullTracker()
    if cfg.tracking.enabled:
        task_name = cfg.tracking.task_name or f"{cfg.run.name}-{run_id}"
        tracker = ClearMLTracker.start(
            project=cfg.tracking.project,
            task_name=task_name,
            output_uri=cfg.tracking.output_uri,
            tags=list(cfg.tracking.tags) + [cfg.run.name, run_id],
            fail_on_error=cfg.tracking.fail_on_error,
            log_artifacts=cfg.tracking.log_artifacts,
            log_model=cfg.tracking.log_model,
        )
        tracker.connect_config(config_to_dict(cfg))

    # Load all training wells
    print("Loading training wells...")
    samples = load_all_wells(cfg, split="train")
    print(f"Loaded {len(samples)} wells")
    tracker.report_scalar("dataset", "n_wells", float(len(samples)), 0)

    if len(samples) == 0:
        print("ERROR: no training wells found", file=sys.stderr)
        tracker.close()
        return

    # 5-fold GroupKFold splits, or same-sample debug overfit mode.
    splits = _train_val_splits(samples, cfg)
    debug_mode = cfg.train.debug_overfit_samples > 0
    if debug_mode:
        n_debug = len(splits[0][0])
        print(f"Debug overfit mode: train=val first {n_debug} wells, augment=False")
        tracker.report_scalar("dataset", "debug_overfit_wells", float(n_debug), 0)

    fold_rmses = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        rmse = _train_fold(fold, train_samples, val_samples, cfg, device, output_dir, tracker)
        fold_rmses.append(rmse)
        tracker.report_scalar("folds", f"fold_{fold}_best_rmse", rmse, fold)

    mean_rmse = float(np.mean(fold_rmses))
    print(f"\n{'='*60}")
    if debug_mode:
        print(f"  Debug Overfit Results: {[f'{r:.4f}' for r in fold_rmses]}")
        print(f"  Mean Debug RMSE: {mean_rmse:.4f}")
    else:
        print(f"  OOF Results: {[f'{r:.4f}' for r in fold_rmses]}")
        print(f"  Mean OOF RMSE: {mean_rmse:.4f}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "fold_rmses": fold_rmses,
        "mean_oof_rmse": mean_rmse,
        "n_wells": len(samples),
        "n_folds": len(splits),
        "debug_overfit_samples": cfg.train.debug_overfit_samples,
        "run_id": run_id,
    }
    with open(output_dir / "oof_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- ClearML: final scalar + artifact upload ----
    tracker.report_scalar("oof", "mean_rmse", mean_rmse, 0)
    tracker.upload_artifacts(output_dir)
    tracker.close()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train RAC-Former v1")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
