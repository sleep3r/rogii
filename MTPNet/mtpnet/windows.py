from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import WindowConfig
from .heatmap import build_channels, fill_nan


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
    base_path = _linear_tail_base_path(comp_tvt_input, first_hidden)
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
