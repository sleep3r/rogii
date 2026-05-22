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


def _build_sample_type_windows(
    wells: list[Any], cfg: MTPConfig, sample_type: str
) -> list[WindowSample]:
    if sample_type not in SAMPLE_TYPE_TO_WINDOW_ARGS:
        raise ValueError(f"Unsupported sample_type: {sample_type}")
    history_mode, center_source = SAMPLE_TYPE_TO_WINDOW_ARGS[sample_type]
    return build_windows_for_wells(
        wells,
        cfg,
        history_mode=history_mode,
        center_source=center_source,
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


def prepare_sample_splits(cfg: MTPConfig) -> SampleSplits:
    wells = discover_wells(cfg.data)
    well_ids = [well.well_id for well in wells]
    train_ids, valid_ids = split_wells(
        well_ids, cfg.validation.valid_fraction, cfg.validation.seed
    )
    by_id = {well.well_id: well for well in wells}
    train_wells = [by_id[well_id] for well_id in train_ids]
    valid_wells = [by_id[well_id] for well_id in valid_ids]
    if cfg.window.train_sample_mix:
        train_buckets = {
            sample_type: _build_sample_type_windows(train_wells, cfg, sample_type)
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
        )
        train_buckets = {"legacy_train": train_samples}
        train_mix_counts = {"legacy_train": len(train_samples)}

    if cfg.window.valid_sample_types:
        valid_sets = {
            VALID_SET_NAMES.get(sample_type, f"valid_{sample_type}"): _build_sample_type_windows(
                valid_wells, cfg, sample_type
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
    mode_entropy: list[float] = []
    target_bin_values: list[float] = []
    pred_bin_values: list[float] = []
    pred_bin_oob: list[float] = []
    raw_path_oob: list[float] = []
    top1_pred_bin_oob: list[float] = []
    weighted_pred_bin_oob: list[float] = []
    best_modes: list[int] = []
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
        "best_mode_mae_bins": float(np.mean(errors_best_mae)),
        "top1_rmse_ft": float(np.mean(errors_top1_ft)),
        "weighted_mean_rmse_ft": float(np.mean(errors_weighted_ft)),
        "oracle_topk_rmse_ft": float(np.mean(errors_oracle_ft)),
        "oracle_top3_rmse_ft": float(np.mean(errors_top3_ft)),
        "target_in_crop_rate": float(np.mean(target_in_crop)),
        "target_at_crop_edge_frac": float(np.mean(target_at_edge)),
        "classification_accuracy_best_mode": float(np.mean(classification_correct)),
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
        elif kind == "no_history":
            for channel_index in history_channels:
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
        "MTP_V0_2_MIXED_REPORT"
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
        "",
        "sanity:",
        "  shuffled_GR baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('shuffled_gr', 'oracle_topk_rmse_ft'))}",
        "  no_history baseline oracle_topK_ft: "
        f"{_metric_value(sanity, ('no_history', 'oracle_topk_rmse_ft'))}",
        "",
        "extra:",
        "  bounded_output: "
        f"{_metric_value(metrics, ('model', 'bounded_output'), valid.get('bounded_output', 'n/a'))}",
        f"  mode_bias_init: {_metric_value(metrics, ('model', 'mode_bias_init'))}",
        f"  alpha_cls: {_metric_value(metrics, ('loss', 'alpha_cls'))}",
        f"  cls_warmup_epochs: {_metric_value(metrics, ('loss', 'cls_warmup_epochs'))}",
        f"  entropy_lambda: {_metric_value(metrics, ('loss', 'entropy_lambda'))}",
        f"  diversity_lambda: {_metric_value(metrics, ('loss', 'diversity_lambda'))}",
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


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
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
            paths, logits = model(x)
            loss, _ = mtp_loss(paths, logits, target, cfg.loss, epoch=epoch)
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
        },
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
