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


def build_windows_for_wells(
    wells: list[Any],
    cfg: MTPConfig,
    *,
    history_mode: str,
    center_source: str,
) -> list[WindowSample]:
    samples: list[WindowSample] = []
    skipped: list[str] = []
    for well in wells:
        horizontal, typewell = load_well(well)
        well_samples = build_windows_for_well(
            well.well_id,
            horizontal,
            typewell,
            cfg.window,
            history_mode=history_mode,
            center_source=center_source,
        )
        if not well_samples:
            skipped.append(well.well_id)
            continue
        samples.extend(well_samples)
    if not samples:
        raise RuntimeError(
            f"No windows were built for history_mode={history_mode}. "
            f"Skipped wells: {skipped[:10]}"
        )
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


def prepare_train_valid_samples(cfg: MTPConfig) -> tuple[list[WindowSample], list[WindowSample]]:
    wells = discover_wells(cfg.data)
    well_ids = [well.well_id for well in wells]
    train_ids, valid_ids = split_wells(
        well_ids, cfg.validation.valid_fraction, cfg.validation.seed
    )
    by_id = {well.well_id: well for well in wells}
    train_wells = [by_id[well_id] for well_id in train_ids]
    valid_wells = [by_id[well_id] for well_id in valid_ids]
    train_samples = build_windows_for_wells(
        train_wells,
        cfg,
        history_mode="teacher_forcing",
        center_source="true_tvt",
    )
    valid_samples = build_windows_for_wells(
        valid_wells,
        cfg,
        history_mode="known_tail_start",
        center_source="tvt_input_tail",
    )
    return train_samples, valid_samples


def _loader(samples: list[WindowSample], cfg: MTPConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        WindowDataset(samples),
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
    )


def _selection_score(metrics: dict[str, Any]) -> float:
    return float(
        0.5 * metrics["oracle_topk_rmse_bins"]
        + 0.3 * metrics["weighted_mean_rmse_bins"]
        + 0.2 * metrics["top1_rmse_bins"]
    )


def _bins_to_tvt(paths: np.ndarray, crop_tvt: np.ndarray) -> np.ndarray:
    paths_np = np.asarray(paths, dtype=np.float32)
    crops_np = np.asarray(crop_tvt, dtype=np.float32)
    grid = np.arange(crops_np.shape[1], dtype=np.float32)
    out = np.empty_like(paths_np, dtype=np.float32)
    for batch_index in range(paths_np.shape[0]):
        flat = paths_np[batch_index].reshape(-1)
        out[batch_index] = np.interp(flat, grid, crops_np[batch_index]).reshape(
            paths_np.shape[1:]
        )
    return out


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
    errors_top1_ft: list[float] = []
    errors_weighted_ft: list[float] = []
    errors_oracle_ft: list[float] = []
    errors_top3_ft: list[float] = []
    target_in_crop: list[float] = []
    target_at_edge: list[float] = []
    classification_correct: list[float] = []
    best_modes: list[int] = []
    with torch.no_grad():
        for batch in _loader(samples, cfg, shuffle=False):
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            target_tvt = batch["target_tvt"].to(device)
            crop_tvt = batch["crop_tvt"].to(device)
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
            paths_np = paths.cpu().numpy()
            target_tvt_np = target_tvt.cpu().numpy()
            crop_tvt_np = crop_tvt.cpu().numpy()
            path_tvt = _bins_to_tvt(paths_np, crop_tvt_np)
            weighted_bins_np = weighted.cpu().numpy()
            weighted_tvt = _bins_to_tvt(weighted_bins_np[:, None, :], crop_tvt_np)[:, 0, :]
            ft_err = np.sqrt(((path_tvt - target_tvt_np[:, None, :]) ** 2).mean(axis=-1))
            top1_np = top1.cpu().numpy()
            best_k_np = best_k.cpu().numpy()
            top3_idx_np = top3_idx.cpu().numpy()
            batch_indices_np = np.arange(paths_np.shape[0])
            errors_top1_ft.extend(ft_err[batch_indices_np, top1_np].tolist())
            errors_weighted_ft.extend(
                np.sqrt(((weighted_tvt - target_tvt_np) ** 2).mean(axis=-1)).tolist()
            )
            errors_oracle_ft.extend(ft_err.min(axis=1).tolist())
            errors_top3_ft.extend(
                np.take_along_axis(ft_err, top3_idx_np, axis=1).min(axis=1).tolist()
            )
            crop_lo = crop_tvt_np[:, :1]
            crop_hi = crop_tvt_np[:, -1:]
            target_in_crop.extend(
                ((target_tvt_np >= crop_lo) & (target_tvt_np <= crop_hi))
                .astype(np.float32)
                .mean(axis=1)
                .tolist()
            )
            target_bins_np = target.cpu().numpy()
            target_at_edge.extend(
                ((target_bins_np <= 1.0) | (target_bins_np >= crop_tvt_np.shape[1] - 2))
                .astype(np.float32)
                .mean(axis=1)
                .tolist()
            )
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
                        "top1_rmse_ft": float(ft_err[i, int(top1_np[i])]),
                        "oracle_rmse_ft": float(ft_err[i, int(best_k_np[i])]),
                        "top1_pred_bins": paths_np[i, int(top1_np[i])].tolist(),
                        "best_pred_bins": paths_np[i, int(best_k_np[i])].tolist(),
                        "weighted_pred_bins": weighted_bins_np[i].tolist(),
                        "top1_pred_tvt": path_tvt[i, int(top1_np[i])].tolist(),
                        "best_pred_tvt": path_tvt[i, int(best_k_np[i])].tolist(),
                        "weighted_pred_tvt": weighted_tvt[i].tolist(),
                        "target_tvt": target_tvt_np[i].tolist(),
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
        "top1_rmse_ft": float(np.mean(errors_top1_ft)),
        "weighted_mean_rmse_ft": float(np.mean(errors_weighted_ft)),
        "oracle_topk_rmse_ft": float(np.mean(errors_oracle_ft)),
        "oracle_top3_rmse_ft": float(np.mean(errors_top3_ft)),
        "target_in_crop_rate": float(np.mean(target_in_crop)),
        "target_at_crop_edge_frac": float(np.mean(target_at_edge)),
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

    train_samples, valid_samples = prepare_train_valid_samples(cfg)
    first = train_samples[0]
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
        valid_score = _selection_score(valid_metrics)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "valid_score": valid_score,
                "valid_oracle_topk_rmse_bins": valid_metrics["oracle_topk_rmse_bins"],
                "valid_weighted_mean_rmse_bins": valid_metrics["weighted_mean_rmse_bins"],
                "valid_top1_rmse_bins": valid_metrics["top1_rmse_bins"],
            }
        )
        if valid_score < best_valid:
            best_valid = valid_score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config_path": str(config_path),
                    "best_epoch": best_epoch,
                    "best_valid_score": best_valid,
                },
                checkpoint_dir / "best.pt",
            )

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    valid_metrics, pred_frame = _evaluate(model, valid_samples, cfg, device)
    train_metrics, _ = _evaluate(model, train_samples, cfg, device)
    valid_metrics["checkpoint_epoch"] = int(checkpoint["best_epoch"])
    valid_metrics["checkpoint_score"] = float(checkpoint["best_valid_score"])
    summary = {
        "train": train_metrics,
        "valid": valid_metrics,
        "history": history,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_valid_score": float(checkpoint["best_valid_score"]),
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
