"""5-fold cross-validation training loop.

Usage (via train.py entry point):
    python -m bphwt.train configs/bphwt_lite.yml

Key features:
  - GroupKFold by well_id (5 folds)
  - AdamW + cosine LR schedule with linear warmup
  - Mixed-precision (torch.amp)
  - EMA weights (start after warmup)
  - Gradient accumulation
  - Validate every N steps, early stopping
  - Saves best EMA checkpoint per fold
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


LOSS_DISPLAY_KEYS = [
    ("loss_total", "total"),
    ("loss_tvt", "tvt"),
    ("loss_fwd_gr", "fwd_gr"),
    ("loss_anchor", "anchor"),
    ("loss_velocity", "vel"),
    ("loss_dip_sign", "dip"),
    ("loss_seg", "seg"),
    ("loss_smooth", "smooth"),
    ("loss_nll", "nll"),
    ("loss_distill", "distill"),
]


class MetricTracker:
    """Accumulate scalar metrics with per-key counts."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            value_f = float(value)
            if not math.isfinite(value_f):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + value_f * n
            self.counts[key] = self.counts.get(key, 0) + n

    def average(self) -> dict[str, float]:
        return {key: self.sums[key] / max(self.counts[key], 1) for key in self.sums}


def format_loss_metrics(prefix: str, metrics: dict[str, float]) -> str:
    """Compact human-readable loss breakdown for logs."""
    parts = []
    for key, label in LOSS_DISPLAY_KEYS:
        if key in metrics:
            parts.append(f"{label}={metrics[key]:.4f}")
    if not parts and "loss" in metrics:
        parts.append(f"total={metrics['loss']:.4f}")
    return f"{prefix} " + " ".join(parts) if parts else f"{prefix} n/a"


def format_progress_header(fold: int) -> str:
    return (
        f"[fold {fold}] "
        f"{'event':<6} {'ep':>7} {'step':>6} {'lr':>9} "
        f"{'train':>10} {'val':>10} {'rmse':>9} {'base':>9} {'gain':>9} "
        f"{'best':>9} {'hidden':>7} {'sec':>6} note"
    )


def format_progress_rule(fold: int) -> str:
    return f"[fold {fold}] " + "-" * 108


def should_log_train_step(global_step: int, log_every: int, log_first_steps: int = 0) -> bool:
    if global_step <= max(0, log_first_steps):
        return True
    return log_every > 0 and global_step % log_every == 0


def format_cv_summary_table(fold_results: list[dict], mean_rmse: float, std_rmse: float) -> str:
    lines = [
        "",
        "CV SUMMARY",
        "Note: validation uses data.val_max_seq_len (0 = full sequence); run `make error-analysis` for per-well diagnostics.",
        f"{'fold':>4} {'val_rmse':>10} {'base':>10} {'gain':>10} {'val_loss':>10} {'best':>10} metrics",
        "-" * 96,
    ]
    for result in fold_results:
        lines.append(
            f"{int(result['fold']):>4} "
            f"{_fmt_float(result.get('val_rmse')):>10} "
            f"{_fmt_float(result.get('val_base_rmse')):>10} "
            f"{_fmt_gain(result.get('val_rmse_gain')):>10} "
            f"{_fmt_float(result.get('val_loss')):>10} "
            f"{_fmt_float(result.get('best_val_rmse')):>10} "
            f"{result.get('metrics_path', '-')}"
        )
    lines.extend(
        [
            "-" * 96,
            f"{'CV':>4} {_fmt_float(mean_rmse):>10} +/- {_fmt_float(std_rmse):<10}",
        ]
    )
    return "\n".join(lines)


def format_progress_row(
    *,
    fold: int,
    event: str,
    epoch: int,
    epochs: int,
    step: int,
    lr: float | None = None,
    train_losses: dict[str, float] | None = None,
    val_losses: dict[str, float] | None = None,
    rmse: float | None = None,
    base_rmse: float | None = None,
    rmse_gain: float | None = None,
    best: float | None = None,
    hidden_rows: int | None = None,
    elapsed_s: float | None = None,
    note: str = "",
) -> str:
    train_losses = train_losses or {}
    val_losses = val_losses or {}
    return (
        f"[fold {fold}] "
        f"{event:<6} "
        f"{epoch:03d}/{epochs:03d} "
        f"{step:06d} "
        f"{_fmt_sci(lr):>9} "
        f"{_fmt_float(train_losses.get('loss_total')):>10} "
        f"{_fmt_float(val_losses.get('loss_total')):>10} "
        f"{_fmt_float(rmse):>9} "
        f"{_fmt_float(base_rmse):>9} "
        f"{_fmt_gain(rmse_gain):>9} "
        f"{_fmt_float(best):>9} "
        f"{_fmt_int(hidden_rows):>7} "
        f"{_fmt_float(elapsed_s, width=6, precision=1):>6} "
        f"{note}"
    ).rstrip()


def _fmt_float(value: float | None, *, width: int = 10, precision: int = 4) -> str:
    if value is None:
        return "-"
    value_f = float(value)
    if not math.isfinite(value_f):
        return str(value_f)
    return f"{value_f:{width}.{precision}f}".strip()


def _fmt_gain(value: float | None, *, width: int = 10, precision: int = 4) -> str:
    if value is None:
        return "-"
    value_f = float(value)
    if not math.isfinite(value_f):
        return str(value_f)
    return f"{value_f:+{width}.{precision}f}".strip()


def _fmt_sci(value: float | None) -> str:
    if value is None:
        return "-"
    value_f = float(value)
    if not math.isfinite(value_f):
        return str(value_f)
    return f"{value_f:.2e}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(int(value))


def append_metric_row(path: Path, row: dict) -> None:
    """Append one metrics row, expanding the CSV header if new columns appear."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    fieldnames = list(row.keys())

    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow(row)


def _flatten_metric_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key.removeprefix('loss_')}": float(value) for key, value in metrics.items()}


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Load EMA weights into model."""
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()})

    def restore(self, model: nn.Module, original_sd: dict) -> None:
        model.load_state_dict(original_sd)


# ---------------------------------------------------------------------------
# LR Schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, total_steps: int, warmup_steps: int, lr: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr - lr_min) * cosine


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    amp_dtype,
) -> dict:
    """Run validation and return loss breakdown + hidden-row RMSE."""
    model.eval()
    tracker = MetricTracker()
    preds_all, base_all, truths_all = [], [], []
    n_hidden_rows = 0

    for batch in loader:
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            preds = _forward(model, batch)
            loss, log_dict = loss_fn(preds, batch)

        tracker.update(log_dict or {"loss_total": loss.item()})

        # Collect hidden-row predictions for RMSE
        hm = batch["hidden_mask"].bool() & (batch["pad_mask"].bool())
        tvt_p = preds["tvt_pred"]
        tvt_b = batch["tvt_base"]
        tvt_t = batch["tvt_true"]
        if hm.any():
            n_hidden_rows += int(hm.sum().item())
            preds_all.append(tvt_p[hm].cpu().float().numpy())
            base_all.append(tvt_b[hm].cpu().float().numpy())
            truths_all.append(tvt_t[hm].cpu().float().numpy())

    model.train()

    rmse = float("inf")
    base_rmse = float("inf")
    rmse_gain = float("nan")
    if preds_all:
        p = np.concatenate(preds_all)
        b = np.concatenate(base_all)
        t = np.concatenate(truths_all)
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        base_rmse = float(np.sqrt(np.mean((b - t) ** 2)))
        rmse_gain = base_rmse - rmse

    losses = tracker.average()
    return {
        "loss": float(losses.get("loss_total", float("nan"))),
        "rmse": rmse,
        "base_rmse": base_rmse,
        "rmse_gain": rmse_gain,
        "losses": losses,
        "n_batches": len(loader),
        "n_hidden_rows": n_hidden_rows,
    }


# ---------------------------------------------------------------------------
# Forward helper (unpacks batch → model call)
# ---------------------------------------------------------------------------


def _forward(model: nn.Module, batch: dict) -> dict:
    X = batch["X"]  # [B, C, L]
    tvt_base = batch["tvt_base"]  # [B, L]
    tvt_input_filled = batch["tvt_input_filled"]
    known_mask = batch["known_mask"]  # [B, L]
    md = batch["md"]  # [B, L]

    return model(
        x=X,
        tvt_base=tvt_base,
        tvt_input=tvt_input_filled,
        known_mask=known_mask,
        md=md,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [x.to(device, non_blocking=True) for x in v]
        else:
            out[k] = v
    return out


def make_validation_config(cfg):
    """Return a validation-only config with full sequences by default."""
    val_cfg = copy.deepcopy(cfg)
    val_max_seq_len = getattr(cfg.data, "val_max_seq_len", 0)
    if val_max_seq_len is not None:
        val_cfg.data.max_seq_len = int(val_max_seq_len)
    val_cfg.data.random_train_crop = False
    return val_cfg


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------


def train_fold(
    cfg,
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    cache_dir: Path,
    output_dir: Path,
) -> dict:
    """
    Train one fold. Returns dict with val_rmse, val_loss, checkpoint_path.
    """
    from bphwt.data.well_dataset import WellDataset, collate_wells
    from bphwt.models.bphwt import BPHWT
    from bphwt.models.losses import BPHWTLoss

    seed_everything(cfg.train.seed + fold)
    device = resolve_device(cfg.train.device)
    amp_dtype = torch.float16 if cfg.train.mixed_precision and device.type == "cuda" else None
    logger.info(f"Fold {fold}: device={device}, amp={amp_dtype}, train={len(train_ids)}, val={len(val_ids)}")

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = fold_dir / "metrics.csv"

    # ---- Datasets & loaders -------------------------------------------
    train_ds = WellDataset(cache_dir, train_ids, cfg, augment=True)
    val_cfg = make_validation_config(cfg)
    val_ds = WellDataset(cache_dir, val_ids, val_cfg, augment=False)
    logger.info(
        "[fold %s] data: train_wells=%s val_wells=%s batch=%s grad_accum=%s train_max_seq_len=%s val_max_seq_len=%s",
        fold,
        len(train_ds),
        len(val_ds),
        cfg.train.batch_size,
        cfg.train.grad_accum,
        cfg.data.max_seq_len,
        val_cfg.data.max_seq_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_wells,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, cfg.train.num_workers),
        collate_fn=collate_wells,
    )

    # ---- Infer in_channels from first sample ---------------------------
    sample_path = cache_dir / f"{train_ids[0]}.npz"
    d = np.load(sample_path, allow_pickle=True)
    in_channels = d["X"].shape[1]
    logger.info(f"in_channels = {in_channels}")

    # ---- Model ---------------------------------------------------------
    mc = cfg.model
    model = BPHWT(
        in_channels=in_channels,
        stage_channels=mc.stage_channels,
        stage_strides=mc.stage_strides,
        decoder_channels=mc.decoder_channels,
        n_blocks=mc.n_blocks,
        use_bottleneck_attn=mc.use_bottleneck_attn,
        attn_heads=mc.attn_heads,
        dropout=mc.dropout,
        stoch_depth=mc.stoch_depth,
        predict_velocity=mc.predict_velocity,
        predict_dip_sign=mc.predict_dip_sign,
        predict_seg_boundary=mc.predict_seg_boundary,
    ).to(device)

    n_params = model.param_count()
    logger.info(f"Model params: {n_params:,}")
    if mc.max_params and n_params > mc.max_params:
        raise ValueError(f"Model has {n_params:,} parameters, above configured max_params={mc.max_params:,}")
    logger.info(format_progress_header(fold))
    logger.info(format_progress_rule(fold))

    # ---- Loss ----------------------------------------------------------
    loss_fn = BPHWTLoss(cfg.loss)

    # ---- Optimizer & schedule ------------------------------------------
    tc = cfg.train
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    steps_per_epoch = len(train_loader)
    total_steps = tc.epochs * steps_per_epoch // tc.grad_accum
    warmup_steps = tc.warmup_epochs * steps_per_epoch // tc.grad_accum
    logger.info(
        "[fold %s] train loop: batches_per_epoch=%s optimizer_steps_per_epoch=%s first_step_logs=%s "
        "log_every=%s validate_every_steps=%s",
        fold,
        steps_per_epoch,
        steps_per_epoch // max(tc.grad_accum, 1),
        cfg.run.log_first_steps,
        cfg.run.log_every,
        tc.validate_every_steps,
    )

    # ---- EMA -----------------------------------------------------------
    ema = EMA(model, decay=tc.ema_decay)
    ema_active = False

    # ---- Training loop -------------------------------------------------
    best_val_rmse = float("inf")
    best_ckpt_path = fold_dir / "best_ema.pt"
    patience_count = 0
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(tc.epochs):
        model.train()
        epoch_tracker = MetricTracker()
        t0 = time.time()
        logger.info(
            "[fold %s] epoch %03d/%03d start: waiting for first batch...",
            fold,
            epoch + 1,
            tc.epochs,
        )

        for step_in_epoch, batch in enumerate(train_loader):
            first_batch = epoch == 0 and step_in_epoch == 0
            first_batch_t0 = time.time()
            if first_batch:
                logger.info(
                    "[fold %s] first batch loaded: X=%s hidden_rows=%s gr_valid_rows=%s",
                    fold,
                    tuple(batch["X"].shape),
                    int(batch["hidden_mask"].sum().item()),
                    int(batch["gr_valid"].sum().item()),
                )
            batch = _to_device(batch, device)

            # LR update
            lr_now = get_lr(global_step, total_steps, warmup_steps, tc.lr, tc.lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                preds = _forward(model, batch)
                loss, log_dict = loss_fn(preds, batch)
                epoch_tracker.update(log_dict or {"loss_total": loss.item()})
                loss = loss / tc.grad_accum
            if first_batch:
                logger.info(
                    "[fold %s] first batch forward/loss ok: loss=%.4f elapsed=%.1fs",
                    fold,
                    float(loss.detach().item() * tc.grad_accum),
                    time.time() - first_batch_t0,
                )

            scaler.scale(loss).backward()
            if first_batch:
                logger.info(
                    "[fold %s] first batch backward ok: elapsed=%.1fs",
                    fold,
                    time.time() - first_batch_t0,
                )

            if (step_in_epoch + 1) % tc.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # EMA update
                if tc.use_ema and epoch >= tc.ema_start_epoch:
                    ema_active = True
                    ema.update(model)

                # Periodic validation
                if global_step % tc.validate_every_steps == 0:
                    # Apply EMA for validation
                    if ema_active:
                        orig_sd = copy.deepcopy(model.state_dict())
                        ema.apply(model)

                    val_metrics = validate(model, val_loader, loss_fn, device, amp_dtype)
                    val_rmse = val_metrics["rmse"]

                    if ema_active:
                        ema.restore(model, orig_sd)

                    train_losses = epoch_tracker.average()
                    val_losses = val_metrics["losses"]
                    is_new_best = val_rmse < best_val_rmse
                    best_after = min(best_val_rmse, val_rmse)

                    append_metric_row(
                        metrics_path,
                        {
                            "event": "val",
                            "fold": fold,
                            "epoch": epoch + 1,
                            "step": global_step,
                            "lr": lr_now,
                            "elapsed_s": time.time() - t0,
                            "val_rmse": val_rmse,
                            "val_base_rmse": val_metrics["base_rmse"],
                            "val_rmse_gain": val_metrics["rmse_gain"],
                            "best_val_rmse": best_after,
                            "val_hidden_rows": val_metrics["n_hidden_rows"],
                            **_flatten_metric_prefix("train", train_losses),
                            **_flatten_metric_prefix("val", val_losses),
                        },
                    )

                    if is_new_best:
                        best_val_rmse = val_rmse
                        patience_count = 0
                        # Save EMA weights
                        if ema_active:
                            orig_sd2 = copy.deepcopy(model.state_dict())
                            ema.apply(model)
                            torch.save(
                                {
                                    "model_state": model.state_dict(),
                                    "fold": fold,
                                    "epoch": epoch,
                                    "val_rmse": best_val_rmse,
                                    "in_channels": in_channels,
                                    "cfg": cfg.model_dump(),
                                },
                                best_ckpt_path,
                            )
                            ema.restore(model, orig_sd2)
                        else:
                            torch.save(
                                {
                                    "model_state": model.state_dict(),
                                    "fold": fold,
                                    "epoch": epoch,
                                    "val_rmse": best_val_rmse,
                                    "in_channels": in_channels,
                                    "cfg": cfg.model_dump(),
                                },
                                best_ckpt_path,
                            )
                    logger.info(
                        format_progress_row(
                            fold=fold,
                            event="VAL*" if is_new_best else "VAL",
                            epoch=epoch + 1,
                            epochs=tc.epochs,
                            step=global_step,
                            lr=lr_now,
                            train_losses=train_losses,
                            val_losses=val_losses,
                            rmse=val_rmse,
                            base_rmse=val_metrics["base_rmse"],
                            rmse_gain=val_metrics["rmse_gain"],
                            best=best_after,
                            hidden_rows=val_metrics["n_hidden_rows"],
                            elapsed_s=time.time() - t0,
                            note=f"saved {best_ckpt_path.name}" if is_new_best else "",
                        )
                    )
                    if not is_new_best:
                        patience_count += 1
                        if patience_count >= tc.early_stop_patience:
                            logger.info(f"Early stopping at step {global_step}")
                            break
                else:
                    if should_log_train_step(global_step, cfg.run.log_every, cfg.run.log_first_steps):
                        train_losses = epoch_tracker.average()
                        logger.info(
                            format_progress_row(
                                fold=fold,
                                event="TRAIN",
                                epoch=epoch + 1,
                                epochs=tc.epochs,
                                step=global_step,
                                lr=lr_now,
                                train_losses=train_losses,
                                best=best_val_rmse,
                                elapsed_s=time.time() - t0,
                            )
                        )
                        append_metric_row(
                            metrics_path,
                            {
                                "event": "train",
                                "fold": fold,
                                "epoch": epoch + 1,
                                "step": global_step,
                                "lr": lr_now,
                                "elapsed_s": time.time() - t0,
                                **_flatten_metric_prefix("train", train_losses),
                            },
                        )

        # Log epoch summary
        elapsed = time.time() - t0
        train_losses = epoch_tracker.average()
        logger.info(
            format_progress_row(
                fold=fold,
                event="EPOCH",
                epoch=epoch + 1,
                epochs=tc.epochs,
                step=global_step,
                lr=optimizer.param_groups[0]["lr"],
                train_losses=train_losses,
                best=best_val_rmse,
                elapsed_s=elapsed,
            )
        )
        append_metric_row(
            metrics_path,
            {
                "event": "epoch",
                "fold": fold,
                "epoch": epoch + 1,
                "step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": elapsed,
                "best_val_rmse": best_val_rmse,
                **_flatten_metric_prefix("train", train_losses),
            },
        )

        # Check early stopping from inner loop
        if patience_count >= tc.early_stop_patience:
            break

    # ---- Final validation with best checkpoint -------------------------
    if not best_ckpt_path.exists():
        final_val_metrics = validate(model, val_loader, loss_fn, device, amp_dtype)
        val_rmse = final_val_metrics["rmse"]
        best_val_rmse = val_rmse
        torch.save(
            {
                "model_state": model.state_dict(),
                "fold": fold,
                "epoch": tc.epochs,
                "val_rmse": best_val_rmse,
                "in_channels": in_channels,
                "cfg": cfg.model_dump(),
            },
            best_ckpt_path,
        )
        logger.info(
            format_progress_row(
                fold=fold,
                event="SAVE",
                epoch=tc.epochs,
                epochs=tc.epochs,
                step=global_step,
                val_losses=final_val_metrics["losses"],
                rmse=val_rmse,
                base_rmse=final_val_metrics["base_rmse"],
                rmse_gain=final_val_metrics["rmse_gain"],
                best=best_val_rmse,
                hidden_rows=final_val_metrics["n_hidden_rows"],
                note=f"saved {best_ckpt_path.name}",
            )
        )

    final_val_loss = float("nan")
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        final_val_metrics = validate(model, val_loader, loss_fn, device, amp_dtype)
        val_loss = final_val_metrics["loss"]
        val_rmse = final_val_metrics["rmse"]
        final_val_loss = val_loss
        logger.info(
            format_progress_row(
                fold=fold,
                event="FINAL",
                epoch=tc.epochs,
                epochs=tc.epochs,
                step=global_step,
                val_losses=final_val_metrics["losses"],
                rmse=val_rmse,
                base_rmse=final_val_metrics["base_rmse"],
                rmse_gain=final_val_metrics["rmse_gain"],
                best=best_val_rmse,
                hidden_rows=final_val_metrics["n_hidden_rows"],
                note=str(best_ckpt_path),
            )
        )
        append_metric_row(
            metrics_path,
            {
                "event": "final",
                "fold": fold,
                "epoch": tc.epochs,
                "step": global_step,
                "val_rmse": val_rmse,
                "val_base_rmse": final_val_metrics["base_rmse"],
                "val_rmse_gain": final_val_metrics["rmse_gain"],
                "best_val_rmse": best_val_rmse,
                "val_hidden_rows": final_val_metrics["n_hidden_rows"],
                **_flatten_metric_prefix("val", final_val_metrics["losses"]),
            },
        )
    else:
        val_rmse = best_val_rmse
        final_val_metrics = {
            "base_rmse": float("nan"),
            "rmse_gain": float("nan"),
        }

    return {
        "fold": fold,
        "val_rmse": float(val_rmse),
        "val_loss": float(final_val_loss),
        "val_base_rmse": float(final_val_metrics.get("base_rmse", float("nan"))),
        "val_rmse_gain": float(final_val_metrics.get("rmse_gain", float("nan"))),
        "best_val_rmse": float(best_val_rmse),
        "checkpoint": str(best_ckpt_path),
        "metrics_path": str(metrics_path),
        "val_ids": val_ids,
    }


# ---------------------------------------------------------------------------
# Run full CV
# ---------------------------------------------------------------------------


def run_cv(cfg) -> dict:
    """Run all folds and collect OOF RMSE."""
    from sklearn.model_selection import GroupKFold

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = cfg.resolved_cache_dir() / "train"

    # Load meta to get well IDs
    meta_path = cfg.resolved_cache_dir() / "meta_train.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Train metadata not found at {meta_path}. Run build_cache first.")
    meta_df = pd.read_csv(meta_path)
    well_ids = meta_df["well_id"].tolist()

    # Check that cache files exist
    missing = [w for w in well_ids if not (cache_dir / f"{w}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} wells have no cache file, e.g. {missing[:3]}")

    logger.info(f"Running {cfg.train.n_folds}-fold CV over {len(well_ids)} wells")

    # Main CV is grouped by well_id. typewell_id is the stress CV mode.
    cv_group = getattr(cfg.train, "cv_group", "well_id")
    if cv_group == "typewell_id" and "typewell_id" in meta_df.columns:
        groups = meta_df["typewell_id"].tolist()
    else:
        groups = well_ids

    gkf = GroupKFold(n_splits=cfg.train.n_folds)
    X_dummy = np.zeros(len(well_ids))

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
        train_ids = [well_ids[i] for i in train_idx]
        val_ids = [well_ids[i] for i in val_idx]

        result = train_fold(
            cfg=cfg,
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )
        fold_results.append(result)

    # Summary
    rmses = [r["val_rmse"] for r in fold_results]
    mean_rmse = float(np.mean(rmses))
    std_rmse = float(np.std(rmses))
    logger.info(format_cv_summary_table(fold_results, mean_rmse, std_rmse))

    summary_rows = [
        {key: value for key, value in result.items() if key not in {"val_ids"}} for result in fold_results
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "fold_summary.csv", index=False)

    # Save fold summary
    summary = {
        "fold_results": fold_results,
        "oof_rmse_mean": mean_rmse,
        "oof_rmse_std": std_rmse,
    }
    import json

    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
