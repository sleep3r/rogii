from __future__ import annotations

import json
import random
import time
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
from .loss import corr_vertical_kl_loss, mode_corr_scores, mtp_loss
from .model import MTPNet
from .priors import PriorTables, load_prior_tables
from .progress import ProgressLogger, format_eta
from .windows import (
    WindowDataset,
    WindowSample,
    build_windows_for_well,
    split_wells,
)
from .synthetic import generate_synthetic_samples, templates_from_samples


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
    synthetic_templates: list[Any] | None = None
    synthetic_count: int = 0


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
    real_train_samples = train_samples
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
    if cfg.synthetic.enabled:
        train_templates = templates_from_samples(real_train_samples)
        valid_templates = templates_from_samples(valid_samples)
        if not train_templates:
            raise RuntimeError("Synthetic training requested but no templates were built")
        synthetic_count = int(cfg.synthetic.windows_per_epoch)
        if cfg.synthetic.real_fraction > 0.0 and train_samples:
            synthetic_count = min(
                synthetic_count,
                int(
                    round(
                        len(train_samples)
                        * cfg.synthetic.real_fraction
                        / max(1e-6, 1.0 - cfg.synthetic.real_fraction)
                    )
                ),
            )
        train_buckets = {**train_buckets, "synthetic_templates": []}
        train_mix_counts = {**train_mix_counts, "synthetic": synthetic_count}
        synthetic_valid = generate_synthetic_samples(
            valid_templates or train_templates,
            cfg,
            count=int(cfg.synthetic.valid_windows),
            seed_offset=1_000_000,
        )
        valid_sets = {**valid_sets, "valid_synthetic": synthetic_valid}
        if cfg.train.selection_source == "synthetic":
            primary_valid_name = "valid_synthetic"
            valid_samples = synthetic_valid
    return SampleSplits(
        train_samples=train_samples,
        valid_samples=valid_samples,
        valid_sets=valid_sets,
        train_buckets=train_buckets,
        train_mix_counts=train_mix_counts,
        primary_valid_name=primary_valid_name,
        synthetic_templates=train_templates if cfg.synthetic.enabled else None,
        synthetic_count=synthetic_count if cfg.synthetic.enabled else 0,
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


def _epoch_train_samples(
    splits: SampleSplits, cfg: MTPConfig, *, epoch: int
) -> list[WindowSample]:
    if not cfg.synthetic.enabled:
        return splits.train_samples
    if not splits.synthetic_templates or splits.synthetic_count <= 0:
        return splits.train_samples
    seed_offset = 100_000 + (int(epoch) - 1) * int(cfg.synthetic.windows_per_epoch)
    synthetic = generate_synthetic_samples(
        splits.synthetic_templates,
        cfg,
        count=splits.synthetic_count,
        seed_offset=seed_offset,
    )
    if cfg.synthetic.real_fraction <= 0.0:
        return synthetic
    return [*splits.train_samples, *synthetic]


def _path_selection_score(metrics: dict[str, Any]) -> float:
    return float(
        0.5 * metrics["oracle_topk_rmse_bins"]
        + 0.3 * metrics["weighted_mean_rmse_bins"]
        + 0.2 * metrics["top1_rmse_bins"]
    )


def _selection_score(metrics: dict[str, Any], *, selection_source: str = "real") -> float:
    path_score = _path_selection_score(metrics)
    corr_nll = metrics.get("corr_nll")
    if corr_nll is None or not np.isfinite(corr_nll):
        return path_score
    top3 = metrics.get("corr_target_top3_rate", 0.0)
    if not np.isfinite(top3):
        top3 = 0.0
    if selection_source == "synthetic":
        return float(
            0.4 * path_score
            + 0.4 * float(corr_nll)
            + 0.2 * (1.0 - float(top3))
        )
    return float(0.8 * path_score + 0.2 * float(corr_nll))


def _epoch_progress_record(
    *,
    epoch: int,
    train_loss: float,
    valid_score: float,
    valid_metrics: dict[str, Any],
    is_best: bool,
    epoch_seconds: float | None = None,
    train_components: dict[str, float] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
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
    if epoch_seconds is not None:
        record["epoch_seconds"] = round(float(epoch_seconds), 3)
    if train_components:
        record["train_components"] = {
            key: round(float(value), 6) for key, value in train_components.items()
        }
    for corr_key in (
        "corr_top1_rmse_ft",
        "corr_target_top3_rate",
        "corr_nll",
        "corr_mode_top1_ft",
    ):
        if corr_key in valid_metrics:
            value = valid_metrics[corr_key]
            try:
                record[f"valid_{corr_key}"] = float(value)
            except (TypeError, ValueError):
                continue
    return record


def _contrastive_corruption_batch(x: torch.Tensor, cfg: MTPConfig) -> torch.Tensor:
    corrupted = x.clone()
    gr_channels = _channel_indices(
        cfg, {"gr_diff", "abs_gr_diff", "gr_z_diff", "dgr_diff"}
    )
    for channel_index in gr_channels:
        channel = corrupted[:, channel_index]
        order = torch.rand(channel.shape, device=corrupted.device).argsort(dim=-1)
        corrupted[:, channel_index] = torch.gather(channel, dim=-1, index=order)
    anchor_channels = _channel_indices(cfg, ANCHOR_CHANNELS)
    if anchor_channels:
        height = max(int(corrupted.shape[2]), 1)
        bin_size_ft = (2.0 * cfg.window.vertical_radius_ft) / max(height - 1, 1)
        jitter_bins = 80.0 / max(bin_size_ft, 1e-6)
        for channel_index in anchor_channels:
            corrupted[:, channel_index] = corrupted[:, channel_index] - float(
                jitter_bins / height
            )
    return corrupted


def _best_mode_mae(paths: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    errors = torch.abs(paths - target[:, None, :]).mean(dim=-1)
    return errors.min(dim=1).values


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
    corr_top1_ft: list[float] = []
    corr_mode_top1_ft: list[float] = []
    corr_mode_weighted_ft: list[float] = []
    corr_target_rank: list[float] = []
    corr_target_top3: list[float] = []
    corr_nll: list[float] = []
    bounded_output = bool(getattr(model, "bounded_output", cfg.model.bounded_output))
    with torch.no_grad():
        for batch in _loader(samples, cfg, shuffle=False):
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            target_tvt = batch["target_tvt"].to(device)
            crop_tvt = batch["crop_tvt"].to(device)
            if hasattr(model, "forward_all"):
                output = model.forward_all(x)
                raw_paths = output.raw_paths
                paths = output.paths
                logits = output.logits
                corr_logits = output.corr_logits
            elif hasattr(model, "forward_raw") and hasattr(model, "bound_paths"):
                raw_paths, logits = model.forward_raw(x)
                paths = model.bound_paths(raw_paths)
                corr_logits = None
            else:
                paths, logits = model(x)
                raw_paths = paths
                corr_logits = None
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
            corr_scores_np: np.ndarray | None = None
            corr_top1_bins_np: np.ndarray | None = None
            corr_mode_top1_np: np.ndarray | None = None
            if corr_logits is not None:
                future_corr = corr_logits[
                    :,
                    :,
                    cfg.window.history_steps : cfg.window.history_steps
                    + cfg.window.future_steps,
                ]
                corr_log_prob = F.log_softmax(future_corr, dim=1)
                corr_top1_bins = future_corr.argmax(dim=1).float()
                corr_scores = mode_corr_scores(
                    corr_logits, paths, future_start=cfg.window.history_steps
                )
                corr_prob = F.softmax(corr_scores, dim=1)
                corr_mode_top1 = corr_scores.argmax(dim=1)
                corr_mode_weighted = (paths * corr_prob[:, :, None]).sum(dim=1)
                corr_scores_np = corr_scores.cpu().numpy()
                corr_top1_bins_np = corr_top1_bins.cpu().numpy()
                corr_mode_top1_np = corr_mode_top1.cpu().numpy()
                corr_mode_weighted_np = corr_mode_weighted.cpu().numpy()
                corr_top1_tvt = _bins_to_tvt(
                    corr_top1_bins_np[:, None, :], crop_tvt_np
                )[:, 0, :]
                corr_mode_weighted_tvt = _bins_to_tvt(
                    corr_mode_weighted_np[:, None, :], crop_tvt_np
                )[:, 0, :]
                corr_top1_ft.extend(
                    np.sqrt(((corr_top1_tvt - target_tvt_np) ** 2).mean(axis=-1)).tolist()
                )
                corr_mode_top1_ft.extend(
                    ft_err[batch_indices_np, corr_mode_top1_np].tolist()
                )
                corr_mode_weighted_ft.extend(
                    np.sqrt(
                        ((corr_mode_weighted_tvt - target_tvt_np) ** 2).mean(axis=-1)
                    ).tolist()
                )
                in_crop_step = (
                    (target >= 0.0) & (target <= future_corr.shape[1] - 1)
                )
                target_rounded = target.round().long().clamp(0, future_corr.shape[1] - 1)
                log_prob_by_step = corr_log_prob.permute(0, 2, 1)
                gathered = torch.gather(
                    log_prob_by_step, dim=2, index=target_rounded[:, :, None]
                ).squeeze(2)
                valid_steps = in_crop_step.to(dtype=gathered.dtype)
                step_counts = valid_steps.sum(dim=1).clamp_min(1.0)
                sample_has_valid = valid_steps.sum(dim=1) > 0
                nll_per_sample = -(gathered * valid_steps).sum(dim=1) / step_counts
                corr_nll.extend(
                    nll_per_sample[sample_has_valid].cpu().tolist()
                )
                corr_order = torch.argsort(future_corr, dim=1, descending=True)
                rank_positions = torch.empty_like(corr_order)
                rank_values = torch.arange(
                    future_corr.shape[1], device=device
                )[None, :, None].expand_as(corr_order)
                rank_positions.scatter_(1, corr_order, rank_values)
                rank_matrix = torch.gather(
                    rank_positions, dim=1, index=target_rounded[:, None, :]
                ).squeeze(1) + 1
                rank_float = rank_matrix.float()
                top3_flag = (rank_matrix <= 3).to(dtype=rank_float.dtype)
                rank_mean_per_sample = (rank_float * valid_steps).sum(dim=1) / step_counts
                top3_per_sample = (top3_flag * valid_steps).sum(dim=1) / step_counts
                corr_target_rank.extend(
                    rank_mean_per_sample[sample_has_valid].cpu().tolist()
                )
                corr_target_top3.extend(
                    top3_per_sample[sample_has_valid].cpu().tolist()
                )
            for i in range(paths.shape[0]):
                row = {
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
                        np.sqrt(((weighted_tvt[i] - target_tvt_np[i]) ** 2).mean())
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
                if corr_scores_np is not None and corr_top1_bins_np is not None:
                    row["corr_scores"] = corr_scores_np[i].tolist()
                    row["corr_top1_pred_bins"] = corr_top1_bins_np[i].tolist()
                    row["corr_top1_mode"] = int(corr_mode_top1_np[i])
                rows.append(row)
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
    if corr_top1_ft:
        metrics.update(
            {
                "corr_top1_rmse_ft": float(np.mean(corr_top1_ft)),
                "corr_target_rank_mean": float(np.mean(corr_target_rank))
                if corr_target_rank
                else float("nan"),
                "corr_target_top3_rate": float(np.mean(corr_target_top3))
                if corr_target_top3
                else float("nan"),
                "corr_nll": float(np.mean(corr_nll)) if corr_nll else float("nan"),
                "corr_mode_top1_ft": float(np.mean(corr_mode_top1_ft)),
                "corr_mode_weighted_ft": float(np.mean(corr_mode_weighted_ft)),
            }
        )
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
SDF_JITTER_CHANNELS = {"anchor_sdf", "base_sdf", "b2_sdf", "a_p50_sdf"}
VALUE_JITTER_CHANNELS = {
    "anchor_offset_value",
    "base_offset_value",
    "b2_delta_value",
}
BATCH_AUGMENTATION_VARIANTS = {
    "normal",
    "no_priors",
    "jittered_priors",
    "wrong_priors",
}


def _zero_channels(x: np.ndarray, indices: list[int]) -> None:
    for channel_index in indices:
        x[channel_index] = 0.0


def _apply_path_jitter(
    x: np.ndarray, cfg: MTPConfig, names: set[str], jitter_ft: float
) -> None:
    if abs(float(jitter_ft)) < 1e-8:
        return
    height = max(int(x.shape[1]), 1)
    bin_size_ft = (2.0 * cfg.window.vertical_radius_ft) / max(height - 1, 1)
    jitter_bins = float(jitter_ft) / max(bin_size_ft, 1e-6)
    sdf_names = names.intersection(SDF_JITTER_CHANNELS)
    for channel_index in _channel_indices(cfg, sdf_names):
        x[channel_index] = x[channel_index] - float(jitter_bins / height)
    value_names = names.intersection(VALUE_JITTER_CHANNELS)
    for channel_index in _channel_indices(cfg, value_names):
        x[channel_index] = x[channel_index] + float(
            jitter_ft / max(cfg.window.vertical_radius_ft, 1e-6)
        )


def _apply_anchor_jitter(x: np.ndarray, cfg: MTPConfig, jitter_ft: float) -> None:
    _apply_path_jitter(x, cfg, ANCHOR_CHANNELS, jitter_ft)


def _apply_wrong_anchor(
    x: np.ndarray,
    cfg: MTPConfig,
    rng: np.random.Generator,
    shifts_ft: tuple[float, ...],
) -> None:
    if not shifts_ft:
        return
    magnitude = float(rng.choice(np.asarray(shifts_ft, dtype=np.float32)))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    _apply_anchor_jitter(x, cfg, sign * magnitude)


def _apply_stochastic_prior_dropout(
    x: np.ndarray,
    cfg: MTPConfig,
    rng: np.random.Generator,
    *,
    include_all_priors: bool = True,
) -> bool:
    aug = cfg.augmentation
    all_prior = _channel_indices(cfg, ALL_PRIOR_CHANNELS)
    if include_all_priors and all_prior and rng.random() < aug.drop_all_priors_prob:
        _zero_channels(x, all_prior)
        return True
    anchor = _channel_indices(cfg, ANCHOR_CHANNELS)
    b2 = _channel_indices(cfg, B2_CHANNELS)
    a_density = _channel_indices(cfg, {"a_density"})
    if rng.random() < aug.drop_anchor_sdf_prob:
        _zero_channels(x, anchor)
    if rng.random() < aug.drop_b2_sdf_prob:
        _zero_channels(x, b2)
    if rng.random() < aug.drop_a_density_prob:
        _zero_channels(x, a_density)
    return False


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
    if aug.batch_mix:
        if any(variant not in BATCH_AUGMENTATION_VARIANTS for variant in aug.batch_mix):
            raise ValueError(
                f"augmentation.batch_mix supports {sorted(BATCH_AUGMENTATION_VARIANTS)}"
            )
        if any(weight <= 0.0 for weight in aug.batch_mix.values()):
            raise ValueError("augmentation.batch_mix weights must be positive")
        total_weight = float(sum(aug.batch_mix.values()))
        total = len(samples)
        mixed: list[WindowSample] = []
        items = list(aug.batch_mix.items())
        for index, (variant, weight) in enumerate(items):
            count = (
                total - len(mixed)
                if index == len(items) - 1
                else int(round(total * float(weight) / total_weight))
            )
            selected = rng.choice(
                np.arange(len(samples)), size=count, replace=count > len(samples)
            )
            for sample_index in selected:
                sample = samples[int(sample_index)]
                x = sample.x.copy()
                if variant == "no_priors":
                    _zero_channels(x, all_prior)
                else:
                    if aug.anchor_swap and rng.random() < aug.anchor_swap_prob:
                        target = str(rng.choice(np.asarray(aug.anchor_swap, dtype=object)))
                        _swap_anchor_channels(x, cfg, target)
                if variant == "jittered_priors":
                    if aug.anchor_jitter_ft:
                        magnitude = float(
                            rng.choice(np.asarray(aug.anchor_jitter_ft, dtype=np.float32))
                        )
                        sign = -1.0 if rng.random() < 0.5 else 1.0
                        _apply_anchor_jitter(x, cfg, sign * magnitude)
                elif variant == "wrong_priors":
                    _apply_wrong_anchor(x, cfg, rng, aug.wrong_anchor_shift_ft)
                elif variant not in {"normal", "no_priors"}:
                    raise ValueError(f"Unsupported augmentation variant: {variant}")
                if variant != "no_priors":
                    _apply_stochastic_prior_dropout(
                        x, cfg, rng, include_all_priors=False
                    )
                mixed.append(replace(sample, x=x, sample_type=f"{sample.sample_type}:{variant}"))
        rng.shuffle(mixed)
        return mixed
    out: list[WindowSample] = []
    for sample in samples:
        x = sample.x.copy()
        if aug.anchor_swap and rng.random() < aug.anchor_swap_prob:
            target = str(rng.choice(np.asarray(aug.anchor_swap, dtype=object)))
            _swap_anchor_channels(x, cfg, target)
        if aug.anchor_jitter_ft:
            magnitude = float(rng.choice(np.asarray(aug.anchor_jitter_ft, dtype=np.float32)))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            _apply_anchor_jitter(x, cfg, sign * magnitude)
        if rng.random() < aug.wrong_anchor_prob:
            _apply_wrong_anchor(x, cfg, rng, aug.wrong_anchor_shift_ft)
        _apply_stochastic_prior_dropout(x, cfg, rng, include_all_priors=True)
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
        elif kind in {"no_base_b2_a", "no_all_priors"}:
            _zero_channels(x, prior_channels)
        elif kind == "no_anchor":
            _zero_channels(x, anchor_channels)
        elif kind in {"anchor_jitter_20ft", "anchor_jitter_20"}:
            _apply_anchor_jitter(x, cfg, 20.0)
        elif kind in {"anchor_jitter_40ft", "anchor_jitter_40"}:
            _apply_anchor_jitter(x, cfg, 40.0)
        elif kind in {"wrong_anchor_80ft", "wrong_anchor_80"}:
            _apply_anchor_jitter(x, cfg, 80.0)
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


def _sanity_gap(
    valid: dict[str, Any], sanity: dict[str, Any], kind: str, metric: str
) -> float | None:
    baseline = sanity.get(kind, {})
    if not isinstance(baseline, dict) or metric not in valid or metric not in baseline:
        return None
    return float(baseline[metric]) - float(valid[metric])


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
        "GEOMTP_V4_SIM2REAL_REPORT"
        if "mtp_v4" in run_name
        else
        "GEOMTP_V3_GR_FORCED_REPORT"
        if "mtp_v3_gr_forced" in run_name
        else "GEOMTP_V2_ANCHOR_DROPOUT_REPORT"
        if "mtp_v2_anchor_dropout" in run_name
        else "MTP_V1_CONDITIONING_REPORT"
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
        f"  corr_top1_ft: {valid.get('corr_top1_rmse_ft', 'n/a')}",
        f"  corr_mode_top1_ft: {valid.get('corr_mode_top1_ft', 'n/a')}",
        f"  corr_mode_weighted_ft: {valid.get('corr_mode_weighted_ft', 'n/a')}",
        f"  corr_target_top3_rate: {valid.get('corr_target_top3_rate', 'n/a')}",
        f"  corr_target_rank_mean: {valid.get('corr_target_rank_mean', 'n/a')}",
        f"  corr_nll: {valid.get('corr_nll', 'n/a')}",
        "  normal_vs_shuffled_top1_gap: "
        f"{_metric_value(metrics, ('sanity_gaps', 'shuffled_gr_top1_gap_ft'))}",
        "  normal_vs_no_GR_top1_gap: "
        f"{_metric_value(metrics, ('sanity_gaps', 'no_gr_top1_gap_ft'))}",
        "  normal_vs_no_all_priors_top1_gap: "
        f"{_metric_value(metrics, ('sanity_gaps', 'no_all_priors_top1_gap_ft'))}",
        "  normal_vs_shuffled_corr_gap: "
        f"{_metric_value(metrics, ('sanity_gaps', 'shuffled_gr_corr_top1_gap_ft'))}",
        "  normal_vs_no_GR_corr_gap: "
        f"{_metric_value(metrics, ('sanity_gaps', 'no_gr_corr_top1_gap_ft'))}",
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
        "  no_GR baseline corr_top1_ft: "
        f"{_metric_value(sanity, ('no_gr', 'corr_top1_rmse_ft'))}",
        "  shuffled_GR baseline top1_ft: "
        f"{_metric_value(sanity, ('shuffled_gr', 'top1_rmse_ft'))}",
        "  shuffled_GR baseline weighted_ft: "
        f"{_metric_value(sanity, ('shuffled_gr', 'weighted_mean_rmse_ft'))}",
        "  shuffled_GR baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('shuffled_gr', 'oracle_topk_rmse_ft'))}",
        "  shuffled_GR baseline corr_top1_ft: "
        f"{_metric_value(sanity, ('shuffled_gr', 'corr_top1_rmse_ft'))}",
        "  no_history baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_history', 'oracle_topk_rmse_ft'))}",
        "  no_base_b2_a baseline top1_ft: "
        f"{_metric_value(sanity, ('no_base_b2_a', 'top1_rmse_ft'))}",
        "  no_base_b2_a baseline weighted_ft: "
        f"{_metric_value(sanity, ('no_base_b2_a', 'weighted_mean_rmse_ft'))}",
        "  no_base_b2_a baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_base_b2_a', 'oracle_topk_rmse_ft'))}",
        "  no_all_priors baseline top1_ft: "
        f"{_metric_value(sanity, ('no_all_priors', 'top1_rmse_ft'))}",
        "  no_all_priors baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_all_priors', 'oracle_topk_rmse_ft'))}",
        "  no_anchor baseline top1_ft: "
        f"{_metric_value(sanity, ('no_anchor', 'top1_rmse_ft'))}",
        "  no_anchor baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_anchor', 'oracle_topk_rmse_ft'))}",
        "  anchor_jitter_20ft baseline top1_ft: "
        f"{_metric_value(sanity, ('anchor_jitter_20ft', 'top1_rmse_ft'))}",
        "  anchor_jitter_20ft baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('anchor_jitter_20ft', 'oracle_topk_rmse_ft'))}",
        "  anchor_jitter_40ft baseline top1_ft: "
        f"{_metric_value(sanity, ('anchor_jitter_40ft', 'top1_rmse_ft'))}",
        "  anchor_jitter_40ft baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('anchor_jitter_40ft', 'oracle_topk_rmse_ft'))}",
        "  wrong_anchor_80ft baseline top1_ft: "
        f"{_metric_value(sanity, ('wrong_anchor_80ft', 'top1_rmse_ft'))}",
        "  wrong_anchor_80ft baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('wrong_anchor_80ft', 'oracle_topk_rmse_ft'))}",
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
        f"  contrastive_alpha: {_metric_value(metrics, ('loss', 'contrastive_alpha'))}",
        "  contrastive_margin_bins: "
        f"{_metric_value(metrics, ('loss', 'contrastive_margin_bins'))}",
        f"  selection_source: {_metric_value(metrics, ('train_config', 'selection_source'))}",
        f"  pretrain_scope: {metrics.get('pretrain_scope', 'n/a')}",
        f"  synthetic_enabled: {_metric_value(metrics, ('synthetic', 'enabled'))}",
        f"  synthetic_real_fraction: {_metric_value(metrics, ('synthetic', 'real_fraction'))}",
        f"  corr_head_enabled: {_metric_value(metrics, ('corr_head', 'enabled'))}",
        f"  corr_alpha_synth: {_metric_value(metrics, ('corr_head', 'alpha_synth'))}",
        f"  corr_alpha_real: {_metric_value(metrics, ('corr_head', 'alpha_real'))}",
        "  raw_path_oob_frac_before_bound: "
        f"{valid.get('raw_path_oob_frac_before_bound', 'n/a')}",
        f"  pred_bin_oob_frac: {valid.get('pred_bin_oob_frac', 'n/a')}",
        f"  top1_pred_bin_oob_frac: {valid.get('top1_pred_bin_oob_frac', 'n/a')}",
        "  weighted_pred_bin_oob_frac: "
        f"{valid.get('weighted_pred_bin_oob_frac', 'n/a')}",
        f"  pred_bin_min: {valid.get('pred_bin_min', 'n/a')}",
        f"  pred_bin_max: {valid.get('pred_bin_max', 'n/a')}",
    ]
    for set_name in (
        "valid_synthetic",
        "valid_first_chunk_known_tail",
        "valid_base_center_all_hidden",
    ):
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
                f"  corr_top1_ft: {set_metrics.get('corr_top1_rmse_ft', 'n/a')}",
                "  corr_target_top3_rate: "
                f"{set_metrics.get('corr_target_top3_rate', 'n/a')}",
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


def _load_init_checkpoint_checked(
    model: MTPNet, checkpoint_path: Path, device: torch.device
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing init_checkpoint: {checkpoint_path}")
    init_state = torch.load(checkpoint_path, map_location=device)
    state = init_state.get("model", init_state)
    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    incompatible: list[str] = []
    unexpected: list[str] = []
    for key, value in state.items():
        if key not in current:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(current[key].shape):
            incompatible.append(key)
            continue
        compatible[key] = value
    missing = sorted(set(current).difference(compatible))
    allowed_prefixes = ("corr_head.",)
    disallowed_unexpected = [
        key for key in unexpected if not key.startswith(allowed_prefixes)
    ]
    disallowed_incompatible = [
        key for key in incompatible if not key.startswith(allowed_prefixes)
    ]
    disallowed_missing = [
        key for key in missing if not key.startswith(allowed_prefixes)
    ]
    if disallowed_unexpected or disallowed_incompatible or disallowed_missing:
        raise ValueError(
            "Incompatible init_checkpoint: "
            f"unexpected={disallowed_unexpected[:5]} "
            f"incompatible={disallowed_incompatible[:5]} "
            f"missing={disallowed_missing[:5]}"
        )
    model.load_state_dict(compatible, strict=False)


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
    logger = ProgressLogger(run=cfg.run.name)
    logger.log(
        "train_run_start",
        run_dir=str(output_dir),
        device=str(device),
        epochs=int(cfg.train.epochs),
        batch_size=int(cfg.train.batch_size),
        synthetic_enabled=bool(cfg.synthetic.enabled),
        corr_head_enabled=bool(cfg.corr_head.enabled),
        corr_head_source=str(cfg.corr_head.source) if cfg.corr_head.enabled else "off",
        selection_source=str(cfg.train.selection_source),
        init_checkpoint=str(cfg.train.init_checkpoint)
        if cfg.train.init_checkpoint is not None
        else None,
    )

    with logger.stage("prepare_sample_splits") as stage:
        splits = prepare_sample_splits(cfg)
        train_samples = splits.train_samples
        valid_samples = splits.valid_samples
        stage["train_windows"] = int(len(train_samples))
        stage["valid_windows"] = int(len(valid_samples))
        stage["train_buckets"] = {
            name: int(len(items)) for name, items in splits.train_buckets.items()
        }
        stage["valid_sets"] = {
            name: int(len(items)) for name, items in splits.valid_sets.items()
        }
        stage["primary_valid_name"] = splits.primary_valid_name
        stage["synthetic_per_epoch"] = int(splits.synthetic_count)
    first = train_samples[0]
    model = MTPNet(
        in_channels=first.x.shape[0],
        height=first.x.shape[1],
        width=first.x.shape[2],
        future_steps=cfg.window.future_steps,
        cfg=cfg.model,
        corr_head=cfg.corr_head,
    ).to(device)
    if cfg.train.init_checkpoint is not None:
        with logger.stage(
            "load_init_checkpoint",
            checkpoint=str(cfg.train.init_checkpoint),
        ):
            _load_init_checkpoint_checked(model, cfg.train.init_checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    best_valid = float("inf")
    history: list[dict[str, float]] = []
    last_epoch_train_samples = train_samples
    heartbeat_every = max(
        1, int(round(len(train_samples) / max(1, cfg.train.batch_size) / 10))
    )
    for epoch in range(1, cfg.train.epochs + 1):
        epoch_start = time.monotonic()
        epoch_train_samples = _epoch_train_samples(splits, cfg, epoch=epoch)
        last_epoch_train_samples = epoch_train_samples
        synth_count = sum(
            1
            for sample in epoch_train_samples
            if str(sample.sample_type).startswith("synthetic")
        )
        total_batches = max(
            1,
            (len(epoch_train_samples) + cfg.train.batch_size - 1) // cfg.train.batch_size,
        )
        logger.log(
            "epoch_start",
            epoch=int(epoch),
            train_samples=int(len(epoch_train_samples)),
            synthetic_samples=int(synth_count),
            real_samples=int(len(epoch_train_samples) - synth_count),
            total_batches=int(total_batches),
            heartbeat_every=int(heartbeat_every),
        )
        model.train()
        losses: list[float] = []
        mtp_losses: list[float] = []
        corr_losses: list[float] = []
        contrastive_losses: list[float] = []
        batch_index = 0
        for batch in _loader(epoch_train_samples, cfg, shuffle=True):
            batch_index += 1
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            history_bins = batch["history_bins"].to(device)
            output = model.forward_all(x)
            paths = output.paths
            logits = output.logits
            mtp_loss_value, _ = mtp_loss(
                paths,
                logits,
                target,
                cfg.loss,
                epoch=epoch,
                history_bins=history_bins,
            )
            loss = mtp_loss_value
            mtp_losses.append(float(mtp_loss_value.detach().cpu()))
            if output.corr_logits is not None:
                future_corr = output.corr_logits[
                    :,
                    :,
                    cfg.window.history_steps : cfg.window.history_steps
                    + cfg.window.future_steps,
                ]
                corr_alpha = torch.tensor(
                    [
                        cfg.corr_head.alpha_synth
                        if str(sample_type).startswith("synthetic")
                        else cfg.corr_head.alpha_real
                        for sample_type in batch["sample_type"]
                    ],
                    device=device,
                    dtype=future_corr.dtype,
                )
                corr_loss = corr_vertical_kl_loss(
                    future_corr,
                    target,
                    tau_bins=cfg.corr_head.target_tau_bins,
                    reduction="none",
                )
                weighted_corr = (corr_alpha * corr_loss).mean()
                loss = loss + weighted_corr
                corr_losses.append(float(weighted_corr.detach().cpu()))
            if cfg.loss.contrastive_alpha > 0.0:
                corrupted_x = _contrastive_corruption_batch(x, cfg)
                corrupted_paths, _ = model(corrupted_x)
                normal_best = _best_mode_mae(paths, target)
                corrupted_best = _best_mode_mae(corrupted_paths, target).detach()
                contrastive = F.relu(
                    normal_best
                    - corrupted_best
                    + float(cfg.loss.contrastive_margin_bins)
                ).mean()
                weighted_contrastive = float(cfg.loss.contrastive_alpha) * contrastive
                loss = loss + weighted_contrastive
                contrastive_losses.append(float(weighted_contrastive.detach().cpu()))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (
                batch_index == 1
                or batch_index % heartbeat_every == 0
                or batch_index == total_batches
            ):
                elapsed = time.monotonic() - epoch_start
                rate = batch_index / max(elapsed, 1e-6)
                remaining = max(total_batches - batch_index, 0)
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                logger.log(
                    "train_heartbeat",
                    epoch=int(epoch),
                    batch=int(batch_index),
                    total_batches=int(total_batches),
                    progress=round(batch_index / total_batches, 4),
                    loss=round(float(np.mean(losses[-heartbeat_every:])), 5),
                    mtp_loss=round(float(np.mean(mtp_losses[-heartbeat_every:])), 5),
                    corr_loss=round(
                        float(np.mean(corr_losses[-heartbeat_every:])), 5
                    )
                    if corr_losses
                    else None,
                    contrastive_loss=round(
                        float(np.mean(contrastive_losses[-heartbeat_every:])), 5
                    )
                    if contrastive_losses
                    else None,
                    batches_per_second=round(rate, 3),
                    eta=format_eta(eta_seconds),
                )
        with logger.stage("epoch_valid_eval", epoch=int(epoch)):
            valid_metrics, _ = _evaluate(model, valid_samples, cfg, device)
        train_loss = float(np.mean(losses))
        train_components = {
            "mtp": float(np.mean(mtp_losses)) if mtp_losses else 0.0,
        }
        if corr_losses:
            train_components["corr"] = float(np.mean(corr_losses))
        if contrastive_losses:
            train_components["contrastive"] = float(np.mean(contrastive_losses))
        valid_score = _selection_score(
            valid_metrics, selection_source=cfg.train.selection_source
        )
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
            epoch_seconds=time.monotonic() - epoch_start,
            train_components=train_components,
        )
        history.append(progress)
        print(json.dumps(progress), flush=True)

    with logger.stage("final_eval") as stage:
        checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        stage["best_epoch"] = int(checkpoint["best_epoch"])
        stage["best_valid_score"] = float(checkpoint["best_valid_score"])
        with logger.stage("eval_primary_valid"):
            valid_metrics, pred_frame = _evaluate(model, valid_samples, cfg, device)
        valid_set_metrics: dict[str, Any] = {}
        for name, samples in splits.valid_sets.items():
            with logger.stage("eval_valid_set", set_name=name, samples=int(len(samples))):
                valid_set_metrics[name] = _evaluate(model, samples, cfg, device)[0]
        if splits.primary_valid_name in valid_set_metrics:
            valid_metrics = valid_set_metrics[splits.primary_valid_name]
            with logger.stage(
                "eval_primary_pred_frame", set_name=splits.primary_valid_name
            ):
                _, pred_frame = _evaluate(
                    model, splits.valid_sets[splits.primary_valid_name], cfg, device
                )
    with logger.stage("eval_train_metrics", samples=int(len(last_epoch_train_samples))):
        train_metrics, _ = _evaluate(model, last_epoch_train_samples, cfg, device)
    sanity_kinds = (
        "no_gr",
        "shuffled_gr",
        "no_history",
        "no_base_b2_a",
        "no_all_priors",
        "no_anchor",
        "anchor_jitter_20ft",
        "anchor_jitter_40ft",
        "wrong_anchor_80ft",
        "base_b2_a_only",
    )
    sanity_metrics: dict[str, Any] = {}
    with logger.stage("sanity_sweep", kinds=list(sanity_kinds)):
        for kind in sanity_kinds:
            with logger.stage("sanity_eval", kind=kind):
                sanity_metrics[kind] = _evaluate(
                    model,
                    _sanity_samples(valid_samples, cfg, kind=kind),
                    cfg,
                    device,
                )[0]
    sanity_gaps = {
        "no_gr_top1_gap_ft": _sanity_gap(valid_metrics, sanity_metrics, "no_gr", "top1_rmse_ft"),
        "shuffled_gr_top1_gap_ft": _sanity_gap(
            valid_metrics, sanity_metrics, "shuffled_gr", "top1_rmse_ft"
        ),
        "no_all_priors_top1_gap_ft": _sanity_gap(
            valid_metrics, sanity_metrics, "no_all_priors", "top1_rmse_ft"
        ),
        "no_gr_corr_top1_gap_ft": _sanity_gap(
            valid_metrics, sanity_metrics, "no_gr", "corr_top1_rmse_ft"
        ),
        "shuffled_gr_corr_top1_gap_ft": _sanity_gap(
            valid_metrics, sanity_metrics, "shuffled_gr", "corr_top1_rmse_ft"
        ),
    }
    sanity_gaps = {
        key: value for key, value in sanity_gaps.items() if value is not None
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
        "sanity_gaps": sanity_gaps,
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
            "contrastive_alpha": cfg.loss.contrastive_alpha,
            "contrastive_margin_bins": cfg.loss.contrastive_margin_bins,
        },
        "priors": _json_safe_config(cfg)["priors"],
        "augmentation": _json_safe_config(cfg)["augmentation"],
        "synthetic": _json_safe_config(cfg)["synthetic"],
        "corr_head": _json_safe_config(cfg)["corr_head"],
        "train_config": _json_safe_config(cfg)["train"],
        "pretrain_scope": (
            "fold_local"
            if cfg.train.init_checkpoint is not None
            and "pretrain" in cfg.train.init_checkpoint.parts
            else "single_split/global"
            if cfg.train.init_checkpoint is not None
            else "none"
        ),
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
