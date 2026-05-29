"""Tiny one-well overfit harness for RAC-Former debugging.

Usage from the repository parent:
    python -m racformer.overfit_one_well racformer/configs/racformer_sanity.yml --data-dir MTPNet/data

Usage from this package directory:
    PYTHONPATH=.. python -m racformer.overfit_one_well configs/racformer_sanity.yml --data-dir ../MTPNet/data
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import RACFormerConfig, config_to_dict, load_config
from .dataset import RACDataset, WellSample, _build_well_sample, _resolve_split_dir, discover_wells
from .loss import RACLoss
from .model import RACFormer
from .train import _get_device, _to_device


def _select_sample(
    samples: Sequence[WellSample | SimpleNamespace],
    well_id: str | None,
    min_hidden_rows: int,
) -> WellSample | SimpleNamespace:
    """Pick the explicit well, or the first sample with enough hidden rows."""
    if well_id:
        for sample in samples:
            if sample.well_id == well_id:
                return sample
        raise ValueError(f"No sample with well_id={well_id!r}")

    for sample in samples:
        if int(sample.n_hidden_rows) >= min_hidden_rows:
            return sample
    raise ValueError(f"No sample with at least {min_hidden_rows} hidden rows")


def _resolve_local_data_dir(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser()
    candidates = [data_dir]
    if not data_dir.is_absolute():
        candidates.extend([Path.cwd() / data_dir, Path.cwd().parent / data_dir])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return data_dir


def _load_candidate_samples(
    cfg: RACFormerConfig,
    split: str,
    well_id: str | None,
    max_candidates: int,
) -> list[WellSample]:
    well_dir = _resolve_split_dir(_resolve_local_data_dir(Path(cfg.data_dir)), split)
    wells = discover_wells(well_dir, k_wells=-1)
    if well_id:
        wells = [item for item in wells if item[0] == well_id]
    else:
        wells = wells[:max_candidates]

    samples: list[WellSample] = []
    for candidate_well_id, horizontal_path, _typewell_path in wells:
        horizontal = pd.read_csv(horizontal_path)
        sample = _build_well_sample(
            well_id=candidate_well_id,
            horizontal=horizontal,
            rows_per_step=cfg.data.rows_per_step,
            max_seq_len=cfg.data.max_seq_len,
            last_known_window=cfg.data.last_known_window,
            tail_class="overfit",
            k_seg=cfg.model.k_seg,
            bin_shift=0,
            use_c0_drift=cfg.data.use_c0_drift,
            top_teacher_eps=cfg.data.top_teacher_eps,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _disable_augmentations(cfg: RACFormerConfig) -> None:
    cfg.augmentation.pseudo_anchor_prob = 0.0
    cfg.augmentation.gr_dropout_prob = 0.0
    cfg.augmentation.feature_noise_std = 0.0


def _disable_regularization(cfg: RACFormerConfig) -> None:
    cfg.model.dropout = 0.0
    cfg.model.stoch_depth = 0.0
    cfg.train.weight_decay = 0.0


def _use_tvt_only_loss(cfg: RACFormerConfig) -> None:
    cfg.train.w_tvt_mse = 1.0
    cfg.train.w_tvt_huber = 0.0
    cfg.train.w_endpoint = 0.0
    cfg.train.w_seg = 0.0
    cfg.train.w_local = 0.0
    cfg.train.w_smooth = 0.0
    cfg.train.w_direct_reg = 0.0
    cfg.train.w_event = 0.0
    cfg.train.w_bucket = 0.0
    cfg.train.w_top_event = 0.0
    cfg.train.w_top_dir = 0.0


def _rmse(pred: torch.Tensor, target: torch.Tensor, n_hidden_rows: torch.Tensor) -> float:
    parts = []
    for b in range(pred.shape[0]):
        h = int(n_hidden_rows[b].item())
        if h > 0:
            parts.append((pred[b, :h] - target[b, :h]) ** 2)
    if not parts:
        return 0.0
    return float(torch.cat(parts).mean().sqrt().item())


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5


def run_overfit(cfg: RACFormerConfig, args: argparse.Namespace) -> dict:
    cfg.tracking.enabled = False
    cfg.data_clearml.enabled = False
    cfg.train.device = args.device
    cfg.train.seed = args.seed
    cfg.train.batch_size = 1
    cfg.train.grad_accum = 1
    _disable_augmentations(cfg)
    _disable_regularization(cfg)
    if args.loss_mode == "tvt":
        _use_tvt_only_loss(cfg)

    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples = _load_candidate_samples(
        cfg,
        split=args.split,
        well_id=args.well_id,
        max_candidates=args.max_candidates,
    )
    sample = _select_sample(samples, args.well_id, args.min_hidden_rows)

    ds = RACDataset(
        [sample],
        max_seq_len=cfg.data.max_seq_len,
        augment=False,
        seed=args.seed,
        k_seg=cfg.model.k_seg,
        rows_per_step=cfg.data.rows_per_step,
    )
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))

    device = _get_device(cfg)
    batch = _to_device(batch, device)
    model = RACFormer(cfg.model).to(device)
    loss_fn = RACLoss(cfg.train, cfg.model).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.train.weight_decay)

    base_rmse = _rmse(batch["base_tvt_hidden"], batch["tvt_hidden"], batch["n_hidden_rows"])
    out = model(batch)
    pred = model.materialize(batch, out)
    initial_rmse = _rmse(pred, batch["tvt_hidden"], batch["n_hidden_rows"])

    output_dir = Path(args.output_dir) / f"{sample.well_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Overfit well={sample.well_id} seq_len={sample.seq_len} "
        f"hidden_rows={sample.n_hidden_rows} device={device} params={model.n_params:,}"
    )
    print(
        f"loss_mode={args.loss_mode} lr={args.lr:.2e} steps={args.steps} "
        f"base_rmse={base_rmse:.4f} initial_rmse={initial_rmse:.4f}"
    )

    history = []
    best_rmse = initial_rmse
    best_step = 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        pred = model.materialize(batch, out)
        losses = loss_fn(out, batch, pred)
        losses["total"].backward()
        grad_norm = _grad_norm(model.parameters())
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        if should_log:
            model.eval()
            with torch.no_grad():
                out_eval = model(batch)
                pred_eval = model.materialize(batch, out_eval)
                rmse = _rmse(pred_eval, batch["tvt_hidden"], batch["n_hidden_rows"])
                max_abs_s = float(out_eval.s_pred.abs().max().item())
                max_abs_direct = float(out_eval.direct_resid_step.abs().max().item())
            if rmse < best_rmse:
                best_rmse = rmse
                best_step = step
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "rmse": rmse,
                        "cfg": config_to_dict(cfg),
                        "well_id": sample.well_id,
                    },
                    output_dir / "best.pt",
                )

            record = {
                "step": step,
                "loss": float(losses["total"].detach().item()),
                "rmse": rmse,
                "grad_norm": grad_norm,
                "max_abs_s": max_abs_s,
                "max_abs_direct": max_abs_direct,
                "seconds": time.time() - t0,
                "loss_components": {
                    k: float(v.detach().item())
                    for k, v in losses.items()
                    if k != "total" and isinstance(v, torch.Tensor)
                },
            }
            history.append(record)
            print(
                f"  step {step:4d}/{args.steps} | "
                f"loss={record['loss']:.6f} | rmse={rmse:.4f} | "
                f"grad={grad_norm:.3e} | |s|={max_abs_s:.4f} | "
                f"|direct|={max_abs_direct:.4f} | dt={record['seconds']:.1f}s"
            )

    summary = {
        "well_id": sample.well_id,
        "seq_len": sample.seq_len,
        "n_hidden_rows": sample.n_hidden_rows,
        "base_rmse": base_rmse,
        "initial_rmse": initial_rmse,
        "best_rmse": best_rmse,
        "best_step": best_step,
        "loss_mode": args.loss_mode,
        "lr": args.lr,
        "steps": args.steps,
        "device": str(device),
        "history": history,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved overfit summary: {output_dir / 'summary.json'}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit RAC-Former on one training well.")
    parser.add_argument("config", help="Path to RAC-Former YAML config")
    parser.add_argument("--data-dir", default="", help="Override data_dir, e.g. ../MTPNet/data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--well-id", default=None)
    parser.add_argument("--min-hidden-rows", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss-mode", choices=["tvt", "full"], default="tvt")
    parser.add_argument("--output-dir", default="artifacts/overfit_one_well")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_overfit(cfg, args)


if __name__ == "__main__":
    main()
