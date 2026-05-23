from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from importlib.util import find_spec
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
from .priors import PriorTables, load_prior_tables
from .windows import (
    WindowDataset,
    WindowSample,
    build_windows_for_well,
    split_wells,
)


SAMPLE_TYPE_TO_WINDOW_ARGS = {
    "teacher_forcing_hidden": ("teacher_forcing", "true_tvt"),
    "known_tail_start": ("known_tail_start", "tvt_input_tail"),
    "base_center_hidden": ("base_path", "base_path"),
}

VALID_SET_NAMES = {
    "known_tail_start": "valid_first_chunk_known_tail",
    "base_center_hidden": "valid_base_center_all_hidden",
    "teacher_forcing_hidden": "valid_teacher_forcing_hidden",
}


@dataclass(frozen=True)
class SampleSplits:
    train_samples: list[WindowSample]
    valid_samples: list[WindowSample]
    valid_sets: dict[str, list[WindowSample]]
    train_buckets: dict[str, list[WindowSample]]
    train_mix_counts: dict[str, int]
    primary_valid_name: str


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
    prior_tables = load_prior_tables(cfg.priors)
    for well in discover_wells(cfg.data):
        horizontal, typewell = load_well(well)
        well_samples = build_windows_for_well(
            well.well_id,
            horizontal,
            typewell,
            cfg.window,
            prior_tables=prior_tables,
        )
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
    prior_tables: PriorTables | None = None,
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
            prior_tables=prior_tables,
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


def _build_sample_type_windows(
    wells: list[Any],
    cfg: MTPConfig,
    sample_type: str,
    *,
    prior_tables: PriorTables | None = None,
) -> list[WindowSample]:
    if sample_type not in SAMPLE_TYPE_TO_WINDOW_ARGS:
        raise ValueError(f"Unsupported sample_type: {sample_type}")
    history_mode, center_source = SAMPLE_TYPE_TO_WINDOW_ARGS[sample_type]
    return build_windows_for_wells(
        wells,
        cfg,
        history_mode=history_mode,
        center_source=center_source,
        prior_tables=prior_tables,
    )


def _mix_train_samples(
    buckets: dict[str, list[WindowSample]],
    weights: dict[str, float],
    seed: int,
) -> tuple[list[WindowSample], dict[str, int]]:
    if not buckets:
        return [], {}
    if any(weight <= 0.0 for weight in weights.values()):
        raise ValueError("window.train_sample_mix weights must be positive")
    total_weight = float(sum(weights.values()))
    normalized = {key: float(value) / total_weight for key, value in weights.items()}
    total = max(len(samples) for samples in buckets.values())
    rng = np.random.default_rng(seed)
    mixed: list[WindowSample] = []
    counts: dict[str, int] = {}
    items = list(normalized.items())
    for index, (sample_type, weight) in enumerate(items):
        if index == len(items) - 1:
            count = total - len(mixed)
        else:
            count = int(round(total * weight))
        source = buckets[sample_type]
        replace_items = count > len(source)
        selected = rng.choice(np.arange(len(source)), size=count, replace=replace_items)
        mixed.extend(source[int(item)] for item in selected)
        counts[sample_type] = count
    rng.shuffle(mixed)
    return mixed, counts


def resolve_train_valid_well_ids(
    well_ids: list[str], cfg: MTPConfig
) -> tuple[list[str], list[str]]:
    known = set(well_ids)
    if cfg.validation.valid_wells:
        valid_ids = [str(well_id) for well_id in cfg.validation.valid_wells]
        train_ids = (
            [str(well_id) for well_id in cfg.validation.train_wells]
            if cfg.validation.train_wells
            else [well_id for well_id in well_ids if well_id not in set(valid_ids)]
        )
        missing = sorted((set(train_ids) | set(valid_ids)).difference(known))
        if missing:
            raise ValueError(f"Explicit validation wells not found: {missing[:10]}")
        if set(train_ids).intersection(valid_ids):
            raise ValueError("Explicit train_wells and valid_wells must be disjoint")
        if not train_ids or not valid_ids:
            raise ValueError("Explicit train_wells and valid_wells must be non-empty")
        return sorted(train_ids), sorted(valid_ids)
    return split_wells(well_ids, cfg.validation.valid_fraction, cfg.validation.seed)


def split_samples(
    samples: list[WindowSample], cfg: MTPConfig
) -> tuple[list[WindowSample], list[WindowSample]]:
    wells = sorted({sample.well_id for sample in samples})
    _, valid_wells = resolve_train_valid_well_ids(wells, cfg)
    valid_set = set(valid_wells)
    train = [sample for sample in samples if sample.well_id not in valid_set]
    valid = [sample for sample in samples if sample.well_id in valid_set]
    if not train or not valid:
        raise RuntimeError(
            f"Invalid split: train_windows={len(train)} valid_windows={len(valid)}"
        )
    return train, valid


def prepare_sample_splits(cfg: MTPConfig) -> SampleSplits:
    wells = discover_wells(cfg.data)
    prior_tables = load_prior_tables(cfg.priors)
    well_ids = [well.well_id for well in wells]
    train_ids, valid_ids = resolve_train_valid_well_ids(well_ids, cfg)
    by_id = {well.well_id: well for well in wells}
    train_wells = [by_id[well_id] for well_id in train_ids]
    valid_wells = [by_id[well_id] for well_id in valid_ids]
    if cfg.window.train_sample_mix:
        train_buckets = {
            sample_type: _build_sample_type_windows(
                train_wells, cfg, sample_type, prior_tables=prior_tables
            )
            for sample_type in cfg.window.train_sample_mix
        }
        train_samples, train_mix_counts = _mix_train_samples(
            train_buckets, cfg.window.train_sample_mix, cfg.train.seed
        )
    else:
        train_samples = build_windows_for_wells(
            train_wells,
            cfg,
            history_mode=cfg.window.train_history_mode,
            center_source=cfg.window.train_center_source,
            prior_tables=prior_tables,
        )
        train_buckets = {"legacy_train": train_samples}
        train_mix_counts = {"legacy_train": len(train_samples)}
    train_samples = _augment_prior_conditioning_samples(train_samples, cfg)

    if cfg.window.valid_sample_types:
        valid_sets = {
            VALID_SET_NAMES.get(sample_type, f"valid_{sample_type}"): _build_sample_type_windows(
                valid_wells, cfg, sample_type, prior_tables=prior_tables
            )
            for sample_type in cfg.window.valid_sample_types
        }
        primary_valid_name = (
            "valid_base_center_all_hidden"
            if "valid_base_center_all_hidden" in valid_sets
            else next(iter(valid_sets))
        )
        valid_samples = valid_sets[primary_valid_name]
    else:
        valid_samples = build_windows_for_wells(
            valid_wells,
            cfg,
            history_mode=cfg.window.valid_history_mode,
            center_source=cfg.window.valid_center_source,
            prior_tables=prior_tables,
        )
        primary_valid_name = "valid"
        valid_sets = {primary_valid_name: valid_samples}
    return SampleSplits(
        train_samples=train_samples,
        valid_samples=valid_samples,
        valid_sets=valid_sets,
        train_buckets=train_buckets,
        train_mix_counts=train_mix_counts,
        primary_valid_name=primary_valid_name,
    )


def prepare_train_valid_samples(cfg: MTPConfig) -> tuple[list[WindowSample], list[WindowSample]]:
    splits = prepare_sample_splits(cfg)
    return splits.train_samples, splits.valid_samples


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


def _epoch_progress_record(
    *,
    epoch: int,
    train_loss: float,
    valid_score: float,
    valid_metrics: dict[str, Any],
    is_best: bool,
) -> dict[str, Any]:
    return {
        "event": "epoch",
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "valid_score": float(valid_score),
        "valid_oracle_topk_rmse_bins": float(valid_metrics["oracle_topk_rmse_bins"]),
        "valid_weighted_mean_rmse_bins": float(
            valid_metrics["weighted_mean_rmse_bins"]
        ),
        "valid_top1_rmse_bins": float(valid_metrics["top1_rmse_bins"]),
        "valid_oracle_topk_rmse_ft": float(valid_metrics["oracle_topk_rmse_ft"]),
        "is_best": bool(is_best),
    }


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


def _ensure_parquet_engine() -> None:
    if find_spec("pyarrow") is not None or find_spec("fastparquet") is not None:
        return
    raise RuntimeError(
        "Writing window_predictions.parquet requires pyarrow or fastparquet. "
        "Install pyarrow or run `uv sync --extra dev` for the MTPNet environment."
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(np.asarray(x, dtype=np.float32))
    ry = _rankdata(np.asarray(y, dtype=np.float32))
    sx = float(rx.std())
    sy = float(ry.std())
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _evaluate(
    model: MTPNet, samples: list[WindowSample], cfg: MTPConfig, device: torch.device
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    errors_top1: list[float] = []
    errors_weighted: list[float] = []
    errors_oracle: list[float] = []
    errors_top3: list[float] = []
    errors_top5: list[float] = []
    errors_best_mae: list[float] = []
    errors_top1_ft: list[float] = []
    errors_weighted_ft: list[float] = []
    errors_oracle_ft: list[float] = []
    errors_top3_ft: list[float] = []
    errors_top5_ft: list[float] = []
    target_in_crop: list[float] = []
    target_at_edge: list[float] = []
    classification_correct: list[float] = []
    mode_entropy: list[float] = []
    target_bin_values: list[float] = []
    pred_bin_values: list[float] = []
    pred_bin_oob: list[float] = []
    raw_path_oob: list[float] = []
    top1_pred_bin_oob: list[float] = []
    weighted_pred_bin_oob: list[float] = []
    best_modes: list[int] = []
    best_mode_ranks: list[float] = []
    best_mode_top1: list[float] = []
    best_mode_top3: list[float] = []
    best_mode_top5: list[float] = []
    logit_error_spearman: list[float] = []
    bounded_output = bool(getattr(model, "bounded_output", cfg.model.bounded_output))
    with torch.no_grad():
        for batch in _loader(samples, cfg, shuffle=False):
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            target_tvt = batch["target_tvt"].to(device)
            crop_tvt = batch["crop_tvt"].to(device)
            if hasattr(model, "forward_raw") and hasattr(model, "bound_paths"):
                raw_paths, logits = model.forward_raw(x)
                paths = model.bound_paths(raw_paths)
            else:
                paths, logits = model(x)
                raw_paths = paths
            prob = F.softmax(logits, dim=1)
            entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1)
            err = torch.sqrt(((paths - target[:, None, :]) ** 2).mean(dim=-1))
            mae = torch.abs(paths - target[:, None, :]).mean(dim=-1)
            top1 = prob.argmax(dim=1)
            batch_idx = torch.arange(paths.shape[0], device=device)
            weighted = (paths * prob[:, :, None]).sum(dim=1)
            max_bin = float(crop_tvt.shape[1] - 1)
            path_oob = (paths < 0.0) | (paths > max_bin)
            raw_path_oob_batch = (raw_paths < 0.0) | (raw_paths > max_bin)
            weighted_oob = (weighted < 0.0) | (weighted > max_bin)
            top3_idx = torch.topk(prob, k=min(3, prob.shape[1]), dim=1).indices
            top5_idx = torch.topk(prob, k=min(5, prob.shape[1]), dim=1).indices
            top3_err = torch.gather(err, 1, top3_idx).min(dim=1).values
            top5_err = torch.gather(err, 1, top5_idx).min(dim=1).values
            oracle_err, best_k = err.min(dim=1)
            logit_order = torch.argsort(logits, dim=1, descending=True)
            rank_positions = torch.empty_like(logit_order)
            rank_values = torch.arange(logits.shape[1], device=device)[None, :].expand_as(
                logit_order
            )
            rank_positions.scatter_(1, logit_order, rank_values)
            best_rank = rank_positions[batch_idx, best_k] + 1
            errors_top1.extend(err[batch_idx, top1].cpu().tolist())
            errors_weighted.extend(
                torch.sqrt(((weighted - target) ** 2).mean(dim=-1)).cpu().tolist()
            )
            errors_oracle.extend(oracle_err.cpu().tolist())
            errors_top3.extend(top3_err.cpu().tolist())
            errors_top5.extend(top5_err.cpu().tolist())
            errors_best_mae.extend(mae[batch_idx, best_k].cpu().tolist())
            classification_correct.extend((top1 == best_k).float().cpu().tolist())
            best_mode_ranks.extend(best_rank.float().cpu().tolist())
            best_mode_top1.extend((best_rank <= 1).float().cpu().tolist())
            best_mode_top3.extend((best_rank <= 3).float().cpu().tolist())
            best_mode_top5.extend((best_rank <= 5).float().cpu().tolist())
            mode_entropy.extend(entropy.cpu().tolist())
            pred_bin_oob.extend(path_oob.float().mean(dim=(1, 2)).cpu().tolist())
            raw_path_oob.extend(
                raw_path_oob_batch.float().mean(dim=(1, 2)).cpu().tolist()
            )
            top1_pred_bin_oob.extend(
                path_oob[batch_idx, top1].float().mean(dim=1).cpu().tolist()
            )
            weighted_pred_bin_oob.extend(weighted_oob.float().mean(dim=1).cpu().tolist())
            best_modes.extend(best_k.cpu().tolist())
            paths_np = paths.cpu().numpy()
            pred_bin_values.extend(paths_np.reshape(-1).tolist())
            target_tvt_np = target_tvt.cpu().numpy()
            crop_tvt_np = crop_tvt.cpu().numpy()
            path_tvt = _bins_to_tvt(paths_np, crop_tvt_np)
            weighted_bins_np = weighted.cpu().numpy()
            weighted_tvt = _bins_to_tvt(weighted_bins_np[:, None, :], crop_tvt_np)[:, 0, :]
            ft_err = np.sqrt(((path_tvt - target_tvt_np[:, None, :]) ** 2).mean(axis=-1))
            top1_np = top1.cpu().numpy()
            best_k_np = best_k.cpu().numpy()
            top3_idx_np = top3_idx.cpu().numpy()
            top5_idx_np = top5_idx.cpu().numpy()
            logits_np = logits.cpu().numpy()
            err_np = err.cpu().numpy()
            for i in range(paths_np.shape[0]):
                logit_error_spearman.append(_spearman(logits_np[i], err_np[i]))
            batch_indices_np = np.arange(paths_np.shape[0])
            errors_top1_ft.extend(ft_err[batch_indices_np, top1_np].tolist())
            errors_weighted_ft.extend(
                np.sqrt(((weighted_tvt - target_tvt_np) ** 2).mean(axis=-1)).tolist()
            )
            errors_oracle_ft.extend(ft_err.min(axis=1).tolist())
            errors_top3_ft.extend(
                np.take_along_axis(ft_err, top3_idx_np, axis=1).min(axis=1).tolist()
            )
            errors_top5_ft.extend(
                np.take_along_axis(ft_err, top5_idx_np, axis=1).min(axis=1).tolist()
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
            target_bin_values.extend(target_bins_np.reshape(-1).tolist())
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
                        "sample_type": batch["sample_type"][i],
                        "top1_mode": int(top1[i].cpu()),
                        "best_mode": int(best_k[i].cpu()),
                        "best_mode_rank_by_logit": int(best_rank[i].cpu()),
                        "top1_rmse_bins": float(err[i, top1[i]].cpu()),
                        "oracle_rmse_bins": float(oracle_err[i].cpu()),
                        "best_mode_mae_bins": float(mae[i, best_k[i]].cpu()),
                        "top1_rmse_ft": float(ft_err[i, int(top1_np[i])]),
                        "oracle_rmse_ft": float(ft_err[i, int(best_k_np[i])]),
                        "weighted_rmse_ft": float(
                            np.sqrt(
                                ((weighted_tvt[i] - target_tvt_np[i]) ** 2).mean()
                            )
                        ),
                        "raw_path_oob_frac_before_bound": float(
                            raw_path_oob_batch[i].float().mean().cpu()
                        ),
                        "top1_pred_bin_oob_frac": float(
                            path_oob[i, int(top1_np[i])].float().mean().cpu()
                        ),
                        "weighted_pred_bin_oob_frac": float(
                            weighted_oob[i].float().mean().cpu()
                        ),
                        "top1_pred_bins": paths_np[i, int(top1_np[i])].tolist(),
                        "best_pred_bins": paths_np[i, int(best_k_np[i])].tolist(),
                        "weighted_pred_bins": weighted_bins_np[i].tolist(),
                        "top1_pred_tvt": path_tvt[i, int(top1_np[i])].tolist(),
                        "best_pred_tvt": path_tvt[i, int(best_k_np[i])].tolist(),
                        "weighted_pred_tvt": weighted_tvt[i].tolist(),
                        "target_tvt": target_tvt_np[i].tolist(),
                        "target_bins": target_bins_np[i].tolist(),
                        "crop_tvt": crop_tvt_np[i].tolist(),
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
        "oracle_top3_by_logit_rmse_bins": float(np.mean(errors_top3)),
        "oracle_top5_by_logit_rmse_bins": float(np.mean(errors_top5)),
        "best_mode_mae_bins": float(np.mean(errors_best_mae)),
        "top1_rmse_ft": float(np.mean(errors_top1_ft)),
        "weighted_mean_rmse_ft": float(np.mean(errors_weighted_ft)),
        "oracle_topk_rmse_ft": float(np.mean(errors_oracle_ft)),
        "oracle_top3_rmse_ft": float(np.mean(errors_top3_ft)),
        "oracle_top3_by_logit_rmse_ft": float(np.mean(errors_top3_ft)),
        "oracle_top5_by_logit_rmse_ft": float(np.mean(errors_top5_ft)),
        "target_in_crop_rate": float(np.mean(target_in_crop)),
        "target_at_crop_edge_frac": float(np.mean(target_at_edge)),
        "classification_accuracy_best_mode": float(np.mean(classification_correct)),
        "best_mode_rank_by_logit_mean": float(np.mean(best_mode_ranks)),
        "best_mode_rank_by_logit_median": float(np.median(best_mode_ranks)),
        "best_mode_top1_rate": float(np.mean(best_mode_top1)),
        "best_mode_top3_rate": float(np.mean(best_mode_top3)),
        "best_mode_top5_rate": float(np.mean(best_mode_top5)),
        "logit_error_spearman": float(np.nanmean(logit_error_spearman)),
        "mode_entropy_mean": float(np.mean(mode_entropy)),
        "target_bin_min": float(np.min(target_bin_values)),
        "target_bin_max": float(np.max(target_bin_values)),
        "bounded_output": bounded_output,
        "raw_path_oob_frac_before_bound": float(np.mean(raw_path_oob)),
        "pred_bin_oob_frac": float(np.mean(pred_bin_oob)),
        "top1_pred_bin_oob_frac": float(np.mean(top1_pred_bin_oob)),
        "weighted_pred_bin_oob_frac": float(np.mean(weighted_pred_bin_oob)),
        "pred_bin_min": float(np.min(pred_bin_values)),
        "pred_bin_max": float(np.max(pred_bin_values)),
        "mode_usage_histogram": mode_hist,
    }
    return metrics, pd.DataFrame(rows)


def _channel_indices(cfg: MTPConfig, names: set[str]) -> list[int]:
    return [index for index, name in enumerate(cfg.window.channels) if name in names]


ANCHOR_CHANNELS = {
    "anchor_sdf",
    "base_sdf",
    "anchor_offset_value",
    "base_offset_value",
}
B2_CHANNELS = {"b2_sdf", "b2_delta_value"}
A_CHANNELS = {"a_p50_sdf", "a_density", "a_p10_p90_band"}
ALL_PRIOR_CHANNELS = ANCHOR_CHANNELS | B2_CHANNELS | A_CHANNELS


def _apply_anchor_jitter(x: np.ndarray, cfg: MTPConfig, jitter_ft: float) -> None:
    if abs(float(jitter_ft)) < 1e-8:
        return
    height = max(int(x.shape[1]), 1)
    bin_size_ft = (2.0 * cfg.window.vertical_radius_ft) / max(height - 1, 1)
    jitter_bins = float(jitter_ft) / max(bin_size_ft, 1e-6)
    for channel_index in _channel_indices(cfg, {"anchor_sdf", "base_sdf"}):
        x[channel_index] = x[channel_index] - float(jitter_bins / height)
    for channel_index in _channel_indices(
        cfg, {"anchor_offset_value", "base_offset_value"}
    ):
        x[channel_index] = x[channel_index] + float(jitter_ft / cfg.window.vertical_radius_ft)


def _swap_anchor_channels(x: np.ndarray, cfg: MTPConfig, target: str) -> None:
    channels = tuple(cfg.window.channels)

    def copy_channel(src_name: str, dst_names: set[str]) -> None:
        if src_name not in channels:
            return
        src = channels.index(src_name)
        for dst in dst_names:
            if dst in channels:
                x[channels.index(dst)] = x[src]

    normalized = target.lower()
    if normalized in {"schema10", "anchor", "base"}:
        return
    if normalized == "b2":
        copy_channel("b2_sdf", {"anchor_sdf", "base_sdf"})
        return
    if normalized in {"a_p50", "a_weighted_mean"}:
        copy_channel("a_p50_sdf", {"anchor_sdf", "base_sdf"})
        return
    if normalized == "noisy_anchor":
        return
    raise ValueError(f"Unsupported anchor_swap target: {target}")


def _augment_prior_conditioning_samples(
    samples: list[WindowSample], cfg: MTPConfig
) -> list[WindowSample]:
    aug = cfg.augmentation
    if not aug.enabled:
        return samples
    rng = np.random.default_rng(cfg.train.seed + 2027)
    all_prior = _channel_indices(cfg, ALL_PRIOR_CHANNELS)
    anchor = _channel_indices(cfg, ANCHOR_CHANNELS)
    b2 = _channel_indices(cfg, B2_CHANNELS)
    a_density = _channel_indices(cfg, {"a_density"})
    out: list[WindowSample] = []
    for sample in samples:
        x = sample.x.copy()
        if all_prior and rng.random() < aug.drop_all_priors_prob:
            for channel_index in all_prior:
                x[channel_index] = 0.0
            out.append(replace(sample, x=x))
            continue
        if aug.anchor_swap and rng.random() < aug.anchor_swap_prob:
            target = str(rng.choice(np.asarray(aug.anchor_swap, dtype=object)))
            _swap_anchor_channels(x, cfg, target)
        if aug.anchor_jitter_ft:
            magnitude = float(rng.choice(np.asarray(aug.anchor_jitter_ft, dtype=np.float32)))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            _apply_anchor_jitter(x, cfg, sign * magnitude)
        if rng.random() < aug.drop_anchor_sdf_prob:
            for channel_index in anchor:
                x[channel_index] = 0.0
        if rng.random() < aug.drop_b2_sdf_prob:
            for channel_index in b2:
                x[channel_index] = 0.0
        if rng.random() < aug.drop_a_density_prob:
            for channel_index in a_density:
                x[channel_index] = 0.0
        out.append(replace(sample, x=x))
    return out


class _StaticModeModel(torch.nn.Module):
    bounded_output = True

    def __init__(self, cfg: MTPConfig, height: int, future_steps: int) -> None:
        super().__init__()
        self.height = height
        self.future_steps = future_steps
        k_modes = cfg.model.k_modes
        max_bin = float(height - 1)
        center = max_bin / 2.0
        span = min(float(cfg.model.mode_bias_span_bins), center)
        self.register_buffer(
            "centers", torch.linspace(center - span, center + span, k_modes)
        )
        self.register_buffer("logits", -torch.abs(self.centers - center))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        paths = self.centers[None, :, None].expand(
            batch, len(self.centers), self.future_steps
        )
        logits = self.logits[None, :].expand(batch, len(self.centers))
        return paths.to(x.device), logits.to(x.device)


def _static_mode_metrics(
    samples: list[WindowSample], cfg: MTPConfig, device: torch.device
) -> dict[str, Any]:
    first = samples[0]
    model = _StaticModeModel(
        cfg, height=first.x.shape[1], future_steps=cfg.window.future_steps
    ).to(device)
    metrics, _ = _evaluate(model, samples, cfg, device)
    return {
        "static_top1_rmse_ft": metrics["top1_rmse_ft"],
        "static_weighted_mean_rmse_ft": metrics["weighted_mean_rmse_ft"],
        "static_oracle_topk_rmse_ft": metrics["oracle_topk_rmse_ft"],
        "static_oracle_top3_by_logit_rmse_ft": metrics[
            "oracle_top3_by_logit_rmse_ft"
        ],
        "static_best_mode_top3_rate": metrics["best_mode_top3_rate"],
    }


def _sanity_samples(
    samples: list[WindowSample],
    cfg: MTPConfig,
    *,
    kind: str,
) -> list[WindowSample]:
    gr_channels = _channel_indices(
        cfg, {"gr_diff", "abs_gr_diff", "gr_z_diff", "dgr_diff"}
    )
    history_channels = _channel_indices(cfg, {"history_mask", "history_sdf"})
    prior_channels = _channel_indices(
        cfg,
        ALL_PRIOR_CHANNELS,
    )
    anchor_channels = _channel_indices(cfg, ANCHOR_CHANNELS)
    rng = np.random.default_rng(cfg.train.seed + 1009)
    transformed: list[WindowSample] = []
    for sample in samples:
        x = sample.x.copy()
        if kind == "shuffled_gr":
            for channel_index in gr_channels:
                for row_index in range(x.shape[1]):
                    x[channel_index, row_index] = rng.permutation(
                        x[channel_index, row_index]
                    )
        elif kind == "no_gr":
            for channel_index in gr_channels:
                x[channel_index] = 0.0
        elif kind == "no_history":
            for channel_index in history_channels:
                x[channel_index] = 0.0
        elif kind == "no_base_b2_a":
            for channel_index in prior_channels:
                x[channel_index] = 0.0
        elif kind == "no_anchor":
            for channel_index in anchor_channels:
                x[channel_index] = 0.0
        elif kind == "anchor_jitter_20ft":
            _apply_anchor_jitter(x, cfg, 20.0)
        elif kind == "base_b2_a_only":
            keep = set(prior_channels)
            for channel_index in range(x.shape[0]):
                if channel_index not in keep:
                    x[channel_index] = 0.0
        else:
            raise ValueError(f"Unsupported sanity sample kind: {kind}")
        transformed.append(replace(sample, x=x))
    return transformed


def _metric_value(
    metrics: dict[str, Any], path: tuple[str, ...], default: str = "n/a"
) -> Any:
    current: Any = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def write_geometry_report(
    metrics: dict[str, Any],
    run_dir: str | Path,
    *,
    parquet_rows: int | None = None,
) -> Path:
    run_path = Path(run_dir)
    valid = metrics.get("valid", metrics)
    train = metrics.get("train", {})
    sanity = metrics.get("sanity", {})
    run_name = str(metrics.get("run_name", ""))
    title = (
        "MTP_V1_CONDITIONING_REPORT"
        if "mtp_v1" in run_name
        else "MTP_V0_2_MIXED_REPORT"
        if "mtp_v0_2" in run_name
        else "MTP_V0_1_DIVERSITY_REPORT"
        if "mtp_v0_1" in run_name
        else "MTP_V0_GEOMETRY_REPORT"
    )
    lines = [
        title,
        "",
        "data:",
        f"  train wells: {metrics.get('num_train_wells', 'n/a')}",
        f"  valid wells: {metrics.get('num_valid_wells', 'n/a')}",
        f"  train windows: {train.get('num_windows', 'n/a')}",
        f"  valid windows: {valid.get('num_windows', 'n/a')}",
        f"  target_in_crop_rate: {valid.get('target_in_crop_rate', 'n/a')}",
        f"  share target at edge: {valid.get('target_at_crop_edge_frac', 'n/a')}",
        "",
        "metrics bins:",
        f"  top1: {valid.get('top1_rmse_bins', 'n/a')}",
        f"  weighted: {valid.get('weighted_mean_rmse_bins', 'n/a')}",
        f"  top3 oracle: {valid.get('oracle_top3_rmse_bins', 'n/a')}",
        f"  topK oracle: {valid.get('oracle_topk_rmse_bins', 'n/a')}",
        "",
        "metrics ft:",
        f"  top1: {valid.get('top1_rmse_ft', 'n/a')}",
        f"  weighted: {valid.get('weighted_mean_rmse_ft', 'n/a')}",
        f"  top3 oracle: {valid.get('oracle_top3_rmse_ft', 'n/a')}",
        f"  topK oracle: {valid.get('oracle_topk_rmse_ft', 'n/a')}",
        "",
        "mode:",
        "  classification_accuracy_best_mode: "
        f"{valid.get('classification_accuracy_best_mode', 'n/a')}",
        f"  mode_usage_histogram: {valid.get('mode_usage_histogram', 'n/a')}",
        f"  entropy mean: {valid.get('mode_entropy_mean', 'n/a')}",
        f"  best_mode_rank_mean: {valid.get('best_mode_rank_by_logit_mean', 'n/a')}",
        f"  best_mode_rank_median: {valid.get('best_mode_rank_by_logit_median', 'n/a')}",
        f"  best_mode_top1_rate: {valid.get('best_mode_top1_rate', 'n/a')}",
        f"  best_mode_top3_rate: {valid.get('best_mode_top3_rate', 'n/a')}",
        f"  best_mode_top5_rate: {valid.get('best_mode_top5_rate', 'n/a')}",
        f"  logit_error_spearman: {valid.get('logit_error_spearman', 'n/a')}",
        "  oracle_top3_by_logit_ft: "
        f"{valid.get('oracle_top3_by_logit_rmse_ft', 'n/a')}",
        "  oracle_top5_by_logit_ft: "
        f"{valid.get('oracle_top5_by_logit_rmse_ft', 'n/a')}",
        "",
        "sanity:",
        "  no_GR baseline top1_ft: "
        f"{_metric_value(sanity, ('no_gr', 'top1_rmse_ft'))}",
        "  no_GR baseline weighted_ft: "
        f"{_metric_value(sanity, ('no_gr', 'weighted_mean_rmse_ft'))}",
        "  no_GR baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_gr', 'oracle_topk_rmse_ft'))}",
        "  shuffled_GR baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('shuffled_gr', 'oracle_topk_rmse_ft'))}",
        "  no_history baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_history', 'oracle_topk_rmse_ft'))}",
        "  no_base_b2_a baseline top1_ft: "
        f"{_metric_value(sanity, ('no_base_b2_a', 'top1_rmse_ft'))}",
        "  no_base_b2_a baseline weighted_ft: "
        f"{_metric_value(sanity, ('no_base_b2_a', 'weighted_mean_rmse_ft'))}",
        "  no_base_b2_a baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_base_b2_a', 'oracle_topk_rmse_ft'))}",
        "  no_anchor baseline top1_ft: "
        f"{_metric_value(sanity, ('no_anchor', 'top1_rmse_ft'))}",
        "  no_anchor baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_anchor', 'oracle_topk_rmse_ft'))}",
        "  anchor_jitter_20ft baseline top1_ft: "
        f"{_metric_value(sanity, ('anchor_jitter_20ft', 'top1_rmse_ft'))}",
        "  anchor_jitter_20ft baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('anchor_jitter_20ft', 'oracle_topk_rmse_ft'))}",
        "  base_b2_a_only baseline top1_ft: "
        f"{_metric_value(sanity, ('base_b2_a_only', 'top1_rmse_ft'))}",
        "  base_b2_a_only baseline weighted_ft: "
        f"{_metric_value(sanity, ('base_b2_a_only', 'weighted_mean_rmse_ft'))}",
        "  base_b2_a_only baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('base_b2_a_only', 'oracle_topk_rmse_ft'))}",
        "",
        "static modes:",
        "  static_top1_ft: "
        f"{_metric_value(metrics, ('static_modes', 'static_top1_rmse_ft'))}",
        "  static_weighted_ft: "
        f"{_metric_value(metrics, ('static_modes', 'static_weighted_mean_rmse_ft'))}",
        "  static_oracle_topK_ft: "
        f"{_metric_value(metrics, ('static_modes', 'static_oracle_topk_rmse_ft'))}",
        "  static_oracle_top3_by_logit_ft: "
        f"{_metric_value(metrics, ('static_modes', 'static_oracle_top3_by_logit_rmse_ft'))}",
        "",
        "extra:",
        "  bounded_output: "
        f"{_metric_value(metrics, ('model', 'bounded_output'), valid.get('bounded_output', 'n/a'))}",
        f"  mode_bias_init: {_metric_value(metrics, ('model', 'mode_bias_init'))}",
        f"  alpha_cls: {_metric_value(metrics, ('loss', 'alpha_cls'))}",
        f"  cls_warmup_epochs: {_metric_value(metrics, ('loss', 'cls_warmup_epochs'))}",
        f"  entropy_lambda: {_metric_value(metrics, ('loss', 'entropy_lambda'))}",
        f"  diversity_lambda: {_metric_value(metrics, ('loss', 'diversity_lambda'))}",
        f"  soft_prob_alpha: {_metric_value(metrics, ('loss', 'soft_prob_alpha'))}",
        f"  soft_prob_tau_bins: {_metric_value(metrics, ('loss', 'soft_prob_tau_bins'))}",
        f"  top3_margin_alpha: {_metric_value(metrics, ('loss', 'top3_margin_alpha'))}",
        f"  top3_margin: {_metric_value(metrics, ('loss', 'top3_margin'))}",
        f"  continuation_alpha: {_metric_value(metrics, ('loss', 'continuation_alpha'))}",
        f"  continuation_tau_bins: {_metric_value(metrics, ('loss', 'continuation_tau_bins'))}",
        "  raw_path_oob_frac_before_bound: "
        f"{valid.get('raw_path_oob_frac_before_bound', 'n/a')}",
        f"  pred_bin_oob_frac: {valid.get('pred_bin_oob_frac', 'n/a')}",
        f"  top1_pred_bin_oob_frac: {valid.get('top1_pred_bin_oob_frac', 'n/a')}",
        "  weighted_pred_bin_oob_frac: "
        f"{valid.get('weighted_pred_bin_oob_frac', 'n/a')}",
        f"  pred_bin_min: {valid.get('pred_bin_min', 'n/a')}",
        f"  pred_bin_max: {valid.get('pred_bin_max', 'n/a')}",
    ]
    for set_name in ("valid_first_chunk_known_tail", "valid_base_center_all_hidden"):
        set_metrics = metrics.get(set_name)
        if not isinstance(set_metrics, dict):
            continue
        lines.extend(
            [
                "",
                f"{set_name}:",
                f"  windows: {set_metrics.get('num_windows', 'n/a')}",
                f"  top1_ft: {set_metrics.get('top1_rmse_ft', 'n/a')}",
                f"  weighted_ft: {set_metrics.get('weighted_mean_rmse_ft', 'n/a')}",
                f"  oracle_top3_ft: {set_metrics.get('oracle_top3_rmse_ft', 'n/a')}",
                f"  oracle_topK_ft: {set_metrics.get('oracle_topk_rmse_ft', 'n/a')}",
                f"  entropy_mean: {set_metrics.get('mode_entropy_mean', 'n/a')}",
                f"  mode_usage_histogram: {set_metrics.get('mode_usage_histogram', 'n/a')}",
            ]
        )
    if parquet_rows is not None:
        lines.extend(["", f"parquet rows: {parquet_rows}"])
    report_path = run_path / "geometry_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


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


def train_from_config(
    config_path: str | Path,
    *,
    output_dir: Path | None = None,
    run_name: str | None = None,
    train_wells: tuple[str, ...] | None = None,
    valid_wells: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if output_dir is not None or run_name is not None:
        cfg = replace(
            cfg,
            run=replace(
                cfg.run,
                name=run_name if run_name is not None else cfg.run.name,
                output_dir=output_dir if output_dir is not None else cfg.run.output_dir,
            ),
        )
    if train_wells is not None or valid_wells is not None:
        cfg = replace(
            cfg,
            validation=replace(
                cfg.validation,
                train_wells=tuple(train_wells or cfg.validation.train_wells),
                valid_wells=tuple(valid_wells or cfg.validation.valid_wells),
            ),
        )
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    output_dir = cfg.run.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_sample_splits(cfg)
    train_samples = splits.train_samples
    valid_samples = splits.valid_samples
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
            history_bins = batch["history_bins"].to(device)
            paths, logits = model(x)
            loss, _ = mtp_loss(
                paths,
                logits,
                target,
                cfg.loss,
                epoch=epoch,
                history_bins=history_bins,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_metrics, _ = _evaluate(model, valid_samples, cfg, device)
        train_loss = float(np.mean(losses))
        valid_score = _selection_score(valid_metrics)
        is_best = valid_score < best_valid
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
        progress = _epoch_progress_record(
            epoch=epoch,
            train_loss=train_loss,
            valid_score=valid_score,
            valid_metrics=valid_metrics,
            is_best=is_best,
        )
        history.append(progress)
        print(json.dumps(progress), flush=True)

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    valid_metrics, pred_frame = _evaluate(model, valid_samples, cfg, device)
    valid_set_metrics = {
        name: _evaluate(model, samples, cfg, device)[0]
        for name, samples in splits.valid_sets.items()
    }
    if splits.primary_valid_name in valid_set_metrics:
        valid_metrics = valid_set_metrics[splits.primary_valid_name]
        _, pred_frame = _evaluate(
            model, splits.valid_sets[splits.primary_valid_name], cfg, device
        )
    train_metrics, _ = _evaluate(model, train_samples, cfg, device)
    sanity_metrics = {
        "no_gr": _evaluate(
            model,
            _sanity_samples(valid_samples, cfg, kind="no_gr"),
            cfg,
            device,
        )[0],
        "shuffled_gr": _evaluate(
            model,
            _sanity_samples(valid_samples, cfg, kind="shuffled_gr"),
            cfg,
            device,
        )[0],
        "no_history": _evaluate(
            model,
            _sanity_samples(valid_samples, cfg, kind="no_history"),
            cfg,
            device,
        )[0],
        "no_base_b2_a": _evaluate(
            model,
            _sanity_samples(valid_samples, cfg, kind="no_base_b2_a"),
            cfg,
            device,
        )[0],
        "no_anchor": _evaluate(
            model,
            _sanity_samples(valid_samples, cfg, kind="no_anchor"),
            cfg,
            device,
        )[0],
        "anchor_jitter_20ft": _evaluate(
            model,
            _sanity_samples(valid_samples, cfg, kind="anchor_jitter_20ft"),
            cfg,
            device,
        )[0],
        "base_b2_a_only": _evaluate(
            model,
            _sanity_samples(valid_samples, cfg, kind="base_b2_a_only"),
            cfg,
            device,
        )[0],
    }
    valid_metrics["checkpoint_epoch"] = int(checkpoint["best_epoch"])
    valid_metrics["checkpoint_score"] = float(checkpoint["best_valid_score"])
    summary = {
        "run_name": cfg.run.name,
        "train": train_metrics,
        "valid": valid_metrics,
        **valid_set_metrics,
        "history": history,
        "sanity": sanity_metrics,
        "primary_valid_name": splits.primary_valid_name,
        "train_mix_counts": splits.train_mix_counts,
        "train_bucket_counts": {
            name: len(samples) for name, samples in splits.train_buckets.items()
        },
        "model": {
            "bounded_output": cfg.model.bounded_output,
            "mode_bias_init": cfg.model.mode_bias_init,
            "mode_bias_span_bins": cfg.model.mode_bias_span_bins,
        },
        "loss": {
            "alpha_cls": cfg.loss.alpha_cls,
            "cls_warmup_epochs": cfg.loss.cls_warmup_epochs,
            "alpha_cls_warmup_value": cfg.loss.alpha_cls_warmup_value,
            "entropy_lambda": cfg.loss.entropy_lambda,
            "entropy_warmup_epochs": cfg.loss.entropy_warmup_epochs,
            "entropy_final_lambda": cfg.loss.entropy_final_lambda,
            "diversity_lambda": cfg.loss.diversity_lambda,
            "diversity_margin_bins": cfg.loss.diversity_margin_bins,
            "soft_prob_alpha": cfg.loss.soft_prob_alpha,
            "soft_prob_tau_bins": cfg.loss.soft_prob_tau_bins,
            "top3_margin_alpha": cfg.loss.top3_margin_alpha,
            "top3_margin": cfg.loss.top3_margin,
            "continuation_alpha": cfg.loss.continuation_alpha,
            "continuation_tau_bins": cfg.loss.continuation_tau_bins,
        },
        "priors": _json_safe_config(cfg)["priors"],
        "augmentation": _json_safe_config(cfg)["augmentation"],
        "static_modes": _static_mode_metrics(valid_samples, cfg, device),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_valid_score": float(checkpoint["best_valid_score"]),
        "best_valid_oracle_topk_rmse_bins": valid_metrics[
            "oracle_topk_rmse_bins"
        ],
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
    _ensure_parquet_engine()
    pred_frame.to_parquet(output_dir / "window_predictions.parquet", index=False)
    write_geometry_report(summary, output_dir, parquet_rows=len(pred_frame))
    print(json.dumps(summary["valid"], indent=2), flush=True)
    return summary
