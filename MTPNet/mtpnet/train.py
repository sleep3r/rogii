from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .config import MTPConfig, load_config
from .io import discover_wells, load_well
from .loss import mtp_loss
from .model import MTPNet
from .windows import (
    WindowDataset,
    WindowSample,
    build_windows_for_well,
    split_wells,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_all_windows(cfg: MTPConfig) -> list[WindowSample]:
    samples: list[WindowSample] = []
    skipped: list[str] = []
    for well in discover_wells(cfg.data):
        horizontal, typewell = load_well(well)
        well_samples = build_windows_for_well(well.well_id, horizontal, typewell, cfg.window)
        if not well_samples:
            skipped.append(well.well_id)
            continue
        samples.extend(well_samples)
    if not samples:
        raise RuntimeError(f"No windows were built. Skipped wells: {skipped[:10]}")
    return samples


def split_samples(
    samples: list[WindowSample], cfg: MTPConfig
) -> tuple[list[WindowSample], list[WindowSample]]:
    wells = sorted({sample.well_id for sample in samples})
    _, valid_wells = split_wells(wells, cfg.validation.valid_fraction, cfg.validation.seed)
    valid_set = set(valid_wells)
    train = [sample for sample in samples if sample.well_id not in valid_set]
    valid = [sample for sample in samples if sample.well_id in valid_set]
    if not train or not valid:
        raise RuntimeError(
            f"Invalid split: train_windows={len(train)} valid_windows={len(valid)}"
        )
    return train, valid


def _loader(samples: list[WindowSample], cfg: MTPConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        WindowDataset(samples),
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
    )


def _evaluate(
    model: MTPNet, samples: list[WindowSample], cfg: MTPConfig, device: torch.device
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    errors_top1: list[float] = []
    errors_weighted: list[float] = []
    errors_oracle: list[float] = []
    errors_top3: list[float] = []
    errors_best_mae: list[float] = []
    classification_correct: list[float] = []
    best_modes: list[int] = []
    with torch.no_grad():
        for batch in _loader(samples, cfg, shuffle=False):
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            paths, logits = model(x)
            prob = F.softmax(logits, dim=1)
            err = torch.sqrt(((paths - target[:, None, :]) ** 2).mean(dim=-1))
            mae = torch.abs(paths - target[:, None, :]).mean(dim=-1)
            top1 = prob.argmax(dim=1)
            batch_idx = torch.arange(paths.shape[0], device=device)
            weighted = (paths * prob[:, :, None]).sum(dim=1)
            top3_idx = torch.topk(prob, k=min(3, prob.shape[1]), dim=1).indices
            top3_err = torch.gather(err, 1, top3_idx).min(dim=1).values
            oracle_err, best_k = err.min(dim=1)
            errors_top1.extend(err[batch_idx, top1].cpu().tolist())
            errors_weighted.extend(
                torch.sqrt(((weighted - target) ** 2).mean(dim=-1)).cpu().tolist()
            )
            errors_oracle.extend(oracle_err.cpu().tolist())
            errors_top3.extend(top3_err.cpu().tolist())
            errors_best_mae.extend(mae[batch_idx, best_k].cpu().tolist())
            classification_correct.extend((top1 == best_k).float().cpu().tolist())
            best_modes.extend(best_k.cpu().tolist())
            for i in range(paths.shape[0]):
                rows.append(
                    {
                        "well_id": batch["well_id"][i],
                        "start_step": int(batch["start_step"][i]),
                        "top1_mode": int(top1[i].cpu()),
                        "best_mode": int(best_k[i].cpu()),
                        "top1_rmse_bins": float(err[i, top1[i]].cpu()),
                        "oracle_rmse_bins": float(oracle_err[i].cpu()),
                        "best_mode_mae_bins": float(mae[i, best_k[i]].cpu()),
                    }
                )
    unique_modes, mode_counts = np.unique(np.array(best_modes), return_counts=True)
    mode_hist = {
        str(k): int(v) for k, v in zip(unique_modes, mode_counts, strict=False)
    }
    metrics = {
        "num_windows": len(samples),
        "top1_rmse_bins": float(np.mean(errors_top1)),
        "weighted_mean_rmse_bins": float(np.mean(errors_weighted)),
        "oracle_topk_rmse_bins": float(np.mean(errors_oracle)),
        "oracle_top3_rmse_bins": float(np.mean(errors_top3)),
        "best_mode_mae_bins": float(np.mean(errors_best_mae)),
        "classification_accuracy_best_mode": float(np.mean(classification_correct)),
        "mode_usage_histogram": mode_hist,
    }
    return metrics, pd.DataFrame(rows)


def _json_safe_config(cfg: MTPConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(asdict(cfg))


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    output_dir = cfg.run.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    samples = build_all_windows(cfg)
    train_samples, valid_samples = split_samples(samples, cfg)
    first = samples[0]
    model = MTPNet(
        in_channels=first.x.shape[0],
        height=first.x.shape[1],
        width=first.x.shape[2],
        future_steps=cfg.window.future_steps,
        cfg=cfg.model,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    best_valid = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in _loader(train_samples, cfg, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            paths, logits = model(x)
            loss, _ = mtp_loss(paths, logits, target, cfg.loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_metrics, _ = _evaluate(model, valid_samples, cfg, device)
        valid_score = valid_metrics["oracle_topk_rmse_bins"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "valid_oracle_topk_rmse_bins": valid_score,
            }
        )
        if valid_score < best_valid:
            best_valid = valid_score
            torch.save(
                {"model": model.state_dict(), "config_path": str(config_path)},
                checkpoint_dir / "best.pt",
            )

    valid_metrics, pred_frame = _evaluate(model, valid_samples, cfg, device)
    train_metrics, _ = _evaluate(model, train_samples, cfg, device)
    summary = {
        "train": train_metrics,
        "valid": valid_metrics,
        "history": history,
        "num_train_wells": len({sample.well_id for sample in train_samples}),
        "num_valid_wells": len({sample.well_id for sample in valid_samples}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yml").write_text(
        yaml.safe_dump(_json_safe_config(cfg), sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pred_frame.to_parquet(output_dir / "window_predictions.parquet", index=False)
    print(json.dumps(summary["valid"], indent=2), flush=True)
    return summary
