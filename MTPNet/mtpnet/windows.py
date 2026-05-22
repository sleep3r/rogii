from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import WindowConfig
from .heatmap import PRIOR_CHANNELS, build_channels, fill_nan
from .priors import PriorTables


@dataclass(frozen=True)
class WindowSample:
    x: np.ndarray
    target_bins: np.ndarray
    target_tvt: np.ndarray
    history_tvt: np.ndarray
    crop_tvt: np.ndarray
    well_id: str
    start_step: int
    center_tvt: float
    sample_type: str = "teacher_forcing_hidden"


def _compress(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step).mean(axis=1)


def _compress_mask(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    return (
        np.asarray(values[:usable], dtype=np.float32)
        .reshape(-1, rows_per_step)
        .mean(axis=1)
    )


def _compress_nanmean(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    arr = np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step)
    finite = np.isfinite(arr)
    sums = np.where(finite, arr, 0.0).sum(axis=1)
    counts = finite.sum(axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out.astype(np.float32)


def _target_bins(target_tvt: np.ndarray, typewell_tvt_crop: np.ndarray) -> np.ndarray:
    return np.abs(typewell_tvt_crop[:, None] - target_tvt[None, :]).argmin(axis=0).astype(
        np.float32
    )


def _crop_typewell(
    typewell: pd.DataFrame, center_tvt: float, cfg: WindowConfig
) -> tuple[np.ndarray, np.ndarray]:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    gr_raw = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr, _ = fill_nan(gr_raw)
    order = np.argsort(tvt)
    tvt_sorted = tvt[order]
    gr_sorted = gr[order]
    finite = np.isfinite(tvt_sorted) & np.isfinite(gr_sorted)
    if not finite.any():
        crop_tvt = np.linspace(
            center_tvt - cfg.vertical_radius_ft,
            center_tvt + cfg.vertical_radius_ft,
            cfg.vertical_bins,
            dtype=np.float32,
        )
        return crop_tvt, np.zeros(cfg.vertical_bins, dtype=np.float32)
    crop_tvt = np.linspace(
        center_tvt - cfg.vertical_radius_ft,
        center_tvt + cfg.vertical_radius_ft,
        cfg.vertical_bins,
        dtype=np.float32,
    )
    crop_gr = np.interp(crop_tvt, tvt_sorted[finite], gr_sorted[finite]).astype(np.float32)
    return crop_tvt.astype(np.float32), crop_gr.astype(np.float32)


def _linear_tail_base_path(comp_tvt_input: np.ndarray, first_hidden: int) -> np.ndarray:
    base = np.asarray(comp_tvt_input, dtype=np.float32).copy()
    known = np.flatnonzero(np.isfinite(base[:first_hidden]))
    if len(known) == 0:
        return base
    last_known = int(known[-1])
    diffs = np.diff(base[known])
    finite_diffs = diffs[np.isfinite(diffs)]
    if len(finite_diffs) == 0:
        slope = 0.0
    else:
        slope = float(np.median(finite_diffs[-min(8, len(finite_diffs)) :]))
    for step in range(last_known + 1, len(base)):
        base[step] = float(base[last_known] + slope * (step - last_known))
    return base.astype(np.float32)


def _fill_with_fallback(values: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if values is None:
        return fallback.astype(np.float32)
    out = np.asarray(values, dtype=np.float32).copy()
    mask = ~np.isfinite(out)
    out[mask] = np.asarray(fallback, dtype=np.float32)[mask]
    return out.astype(np.float32)


def _aligned_prior_columns(
    well_id: str, horizontal: pd.DataFrame, prior_tables: PriorTables | None
) -> dict[str, np.ndarray]:
    if prior_tables is None:
        return {}
    if "id" in horizontal.columns:
        ids = horizontal["id"].astype(str).to_numpy()
    else:
        ids = np.array(
            [f"{well_id}_{row_index}" for row_index in range(len(horizontal))],
            dtype=object,
        )
    aligned = prior_tables.frame.reindex(ids)
    return {
        column: pd.to_numeric(aligned[column], errors="coerce").to_numpy(
            dtype=np.float32
        )
        for column in aligned.columns
    }


def _target_bins_with_nan(path_tvt: np.ndarray, crop_tvt: np.ndarray) -> np.ndarray:
    out = np.full(len(path_tvt), np.nan, dtype=np.float32)
    finite = np.isfinite(path_tvt)
    if finite.any():
        out[finite] = _target_bins(path_tvt[finite], crop_tvt)
    return out


def _prior_channels_for_window(
    *,
    cfg: WindowConfig,
    crop_tvt: np.ndarray,
    center_tvt: float,
    base_path: np.ndarray,
    b2_path: np.ndarray | None,
    a_p50_path: np.ndarray | None,
    a_p10_path: np.ndarray | None,
    a_p90_path: np.ndarray | None,
    start: int,
    total_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    requested = PRIOR_CHANNELS.intersection(cfg.channels)
    if not requested:
        return {}, {}
    if any(name.startswith("b2") for name in requested) and b2_path is None:
        raise ValueError("b2 prior channels require loaded b2_tvt priors")
    if any(name.startswith("a_") for name in requested) and a_p50_path is None:
        raise ValueError("A prior channels require loaded a_p50_tvt priors")

    sl = slice(start, start + total_steps)
    base = base_path[sl]
    b2 = _fill_with_fallback(b2_path[sl], base) if b2_path is not None else None
    a_p50 = (
        _fill_with_fallback(a_p50_path[sl], base)
        if a_p50_path is not None
        else None
    )
    a_p10 = (
        _fill_with_fallback(a_p10_path[sl], a_p50)
        if a_p10_path is not None and a_p50 is not None
        else a_p50
    )
    a_p90 = (
        _fill_with_fallback(a_p90_path[sl], a_p50)
        if a_p90_path is not None and a_p50 is not None
        else a_p50
    )
    prior_bins: dict[str, np.ndarray] = {
        "base": _target_bins_with_nan(base, crop_tvt),
    }
    prior_values: dict[str, np.ndarray] = {
        "base_offset": ((base - float(center_tvt)) / cfg.vertical_radius_ft).astype(
            np.float32
        )
    }
    if b2 is not None:
        prior_bins["b2"] = _target_bins_with_nan(b2, crop_tvt)
        prior_values["b2_delta"] = ((b2 - base) / cfg.vertical_radius_ft).astype(
            np.float32
        )
    if a_p50 is not None:
        prior_bins["a_p50"] = _target_bins_with_nan(a_p50, crop_tvt)
        prior_bins["a_p10"] = _target_bins_with_nan(a_p10, crop_tvt)
        prior_bins["a_p90"] = _target_bins_with_nan(a_p90, crop_tvt)
    return prior_bins, prior_values


def _sample_type_for(history_mode: str) -> str:
    if history_mode == "teacher_forcing":
        return "teacher_forcing_hidden"
    if history_mode == "known_tail_start":
        return "known_tail_start"
    if history_mode == "base_path":
        return "base_center_hidden"
    raise ValueError(f"Unsupported history_mode: {history_mode}")


def build_windows_for_well(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: WindowConfig,
    *,
    history_mode: str = "teacher_forcing",
    center_source: str = "true_tvt",
    prior_tables: PriorTables | None = None,
) -> list[WindowSample]:
    total_steps = cfg.history_steps + cfg.future_steps
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr, finite = fill_nan(gr_raw)

    comp_tvt = _compress(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress(tvt_input, cfg.rows_per_step)
    comp_gr = _compress(gr, cfg.rows_per_step)
    comp_finite = _compress_mask(finite, cfg.rows_per_step)
    aligned_priors = _aligned_prior_columns(well_id, horizontal, prior_tables)
    comp_priors = {
        name: _compress_nanmean(values, cfg.rows_per_step)
        for name, values in aligned_priors.items()
    }
    if len(comp_tvt) < total_steps:
        return []

    hidden = ~np.isfinite(comp_tvt_input)
    hidden_steps = np.flatnonzero(hidden)
    if len(hidden_steps) == 0:
        return []
    first_hidden = int(hidden_steps[0])
    if history_mode == "known_tail_only":
        history_mode = "known_tail_start"
    if history_mode not in {"teacher_forcing", "known_tail_start", "base_path"}:
        raise ValueError(f"Unsupported history_mode: {history_mode}")
    if center_source not in {"true_tvt", "tvt_input_tail", "base_path"}:
        raise ValueError(f"Unsupported center_source: {center_source}")
    linear_base_path = _linear_tail_base_path(comp_tvt_input, first_hidden)
    base_path = _fill_with_fallback(comp_priors.get("base_tvt"), linear_base_path)
    b2_path = (
        _fill_with_fallback(comp_priors.get("b2_tvt"), base_path)
        if "b2_tvt" in comp_priors
        else None
    )
    a_p50_path = (
        _fill_with_fallback(comp_priors.get("a_p50_tvt"), base_path)
        if "a_p50_tvt" in comp_priors
        else None
    )
    a_p10_path = (
        _fill_with_fallback(comp_priors.get("a_p10_tvt"), a_p50_path)
        if "a_p10_tvt" in comp_priors and a_p50_path is not None
        else None
    )
    a_p90_path = (
        _fill_with_fallback(comp_priors.get("a_p90_tvt"), a_p50_path)
        if "a_p90_tvt" in comp_priors and a_p50_path is not None
        else None
    )
    if PRIOR_CHANNELS.intersection(cfg.channels) and prior_tables is None:
        raise ValueError("Prior channels requested but no prior tables were loaded")
    if (
        {"base_sdf", "base_offset_value"}.intersection(cfg.channels)
        and "base_tvt" not in comp_priors
    ):
        raise ValueError("base prior channels require loaded base_tvt priors")
    min_start = max(0, first_hidden - cfg.history_steps)
    max_start = len(comp_tvt) - total_steps
    if history_mode == "known_tail_start":
        starts = [first_hidden - cfg.history_steps]
    else:
        starts = list(range(min_start, max_start + 1, max(1, cfg.stride_steps)))
    windows: list[WindowSample] = []
    for start in starts:
        if start < 0 or start > max_start:
            continue
        hist_slice = slice(start, start + cfg.history_steps)
        fut_slice = slice(start + cfg.history_steps, start + total_steps)
        if not np.isfinite(comp_tvt[hist_slice]).all() or not np.isfinite(
            comp_tvt[fut_slice]
        ).all():
            continue
        if history_mode == "known_tail_start":
            if not np.isfinite(comp_tvt_input[hist_slice]).all():
                continue
            if not hidden[fut_slice].all():
                continue
        if history_mode == "base_path":
            if not np.isfinite(base_path[hist_slice]).all():
                continue
            if not hidden[fut_slice].all():
                continue
        if center_source == "tvt_input_tail":
            center_tvt = float(comp_tvt_input[start + cfg.history_steps - 1])
            if not np.isfinite(center_tvt):
                continue
        elif center_source == "base_path":
            center_tvt = float(base_path[start + cfg.history_steps - 1])
            if not np.isfinite(center_tvt):
                continue
        else:
            center_tvt = float(comp_tvt[start + cfg.history_steps - 1])
        crop_tvt, crop_gr = _crop_typewell(typewell, center_tvt, cfg)
        prior_bins, prior_values = _prior_channels_for_window(
            cfg=cfg,
            crop_tvt=crop_tvt,
            center_tvt=center_tvt,
            base_path=base_path,
            b2_path=b2_path,
            a_p50_path=a_p50_path,
            a_p10_path=a_p10_path,
            a_p90_path=a_p90_path,
            start=start,
            total_steps=total_steps,
        )
        path_all = _target_bins(comp_tvt[start : start + total_steps], crop_tvt)
        if history_mode == "base_path":
            history_path = _target_bins(
                base_path[start : start + cfg.history_steps], crop_tvt
            )
        else:
            history_path = path_all[: cfg.history_steps]
        history_bins = np.full(total_steps, np.nan, dtype=np.float32)
        history_bins[: cfg.history_steps] = history_path
        history_tvt = (
            comp_tvt_input[hist_slice]
            if history_mode == "known_tail_start"
            else base_path[hist_slice]
            if history_mode == "base_path"
            else comp_tvt[hist_slice]
        )
        x = build_channels(
            horizontal_gr=comp_gr[start : start + total_steps],
            typewell_gr=crop_gr,
            history_bins=history_bins,
            finite_steps=comp_finite[start : start + total_steps],
            channels=cfg.channels,
            prior_bins=prior_bins,
            prior_values=prior_values,
        )
        windows.append(
            WindowSample(
                x=x,
                target_bins=path_all[cfg.history_steps :].astype(np.float32),
                target_tvt=comp_tvt[fut_slice].astype(np.float32),
                history_tvt=history_tvt.astype(np.float32),
                crop_tvt=crop_tvt.astype(np.float32),
                well_id=well_id,
                start_step=start,
                center_tvt=center_tvt,
                sample_type=_sample_type_for(history_mode),
            )
        )
        if len(windows) >= cfg.max_windows_per_well:
            break
    return windows


def split_wells(
    well_ids: list[str], valid_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    ordered = sorted(well_ids)
    if len(ordered) < 2:
        return ordered, []
    rng = np.random.default_rng(seed)
    shuffled = np.array(ordered, dtype=object)
    rng.shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * valid_fraction)))
    valid = sorted(str(v) for v in shuffled[:n_valid])
    train = sorted(str(v) for v in shuffled[n_valid:])
    return train, valid


class WindowDataset(Dataset):
    def __init__(self, samples: list[WindowSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "x": torch.from_numpy(sample.x).float(),
            "target_bins": torch.from_numpy(sample.target_bins).float(),
            "target_tvt": torch.from_numpy(sample.target_tvt).float(),
            "crop_tvt": torch.from_numpy(sample.crop_tvt).float(),
            "well_id": sample.well_id,
            "start_step": sample.start_step,
            "sample_type": sample.sample_type,
        }
