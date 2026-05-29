from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import GADataConfig

N_LATERAL_FEATURES = 10
N_TYPEWELL_FEATURES = 4

LATERAL_FEATURE_COLUMNS = (
    "step_frac",
    "md_z",
    "x_z",
    "y_z",
    "z_z",
    "gr_z",
    "gr_valid_frac",
    "tvt_input_delta",
    "known_mask",
    "hidden_progress",
)
TYPEWELL_FEATURE_COLUMNS = ("tvt_frac", "gr_z", "dgr_z", "bias")


@dataclass
class WellPaths:
    well_id: str
    horizontal_path: Path
    typewell_path: Path


@dataclass
class AlignmentSample:
    well_id: str
    lateral_features: np.ndarray
    typewell_features: np.ndarray
    target_bins: np.ndarray
    target_tvt: np.ndarray
    comp_tvt_input: np.ndarray
    hidden_mask: np.ndarray
    typewell_tvt: np.ndarray
    seq_len: int
    typewell_len: int
    last_known_tvt: float
    first_hidden_step: int
    tail_class: str = "unknown"
    feature_columns: tuple[str, ...] = LATERAL_FEATURE_COLUMNS
    hidden_row_ids: np.ndarray | None = None
    hidden_row_steps: np.ndarray | None = None
    hidden_row_tvt: np.ndarray | None = None


def _nanmean_steps(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    arr = np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step)
    finite = np.isfinite(arr)
    sums = np.where(finite, arr, 0.0).sum(axis=1)
    counts = finite.sum(axis=1).astype(np.float32)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out.astype(np.float32)


def _valid_frac_steps(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    arr = np.isfinite(np.asarray(values[:usable], dtype=np.float32)).reshape(-1, rows_per_step)
    return arr.mean(axis=1).astype(np.float32)


def _fill_nan(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr, finite
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32), finite
    idx = np.arange(len(arr), dtype=np.float32)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite]).astype(np.float32)
    return arr.astype(np.float32), finite


def _zscore(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(arr) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(arr))
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(arr[mask].mean())
    std = float(arr[mask].std())
    std = std if std > 1e-6 else 1.0
    return np.where(np.isfinite(arr), (arr - mean) / std, 0.0).astype(np.float32)


def _column_or_index(frame: pd.DataFrame, column: str, fallback: np.ndarray) -> np.ndarray:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)
    return np.asarray(fallback, dtype=np.float32)


def regular_typewell_grid(
    typewell: pd.DataFrame, vertical_step_ft: float, max_bins: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    gr_raw = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float32)
    gr, _ = _fill_nan(gr_raw)
    order = np.argsort(tvt)
    tvt_sorted = tvt[order]
    gr_sorted = gr[order]
    finite = np.isfinite(tvt_sorted) & np.isfinite(gr_sorted)
    if not finite.any():
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    lo = float(np.nanmin(tvt_sorted[finite]))
    hi = float(np.nanmax(tvt_sorted[finite]))
    grid = np.arange(lo, hi + vertical_step_ft * 0.5, vertical_step_ft, dtype=np.float32)
    if max_bins is not None and len(grid) > max_bins:
        grid = grid[:max_bins]
    gr_grid = np.interp(grid, tvt_sorted[finite], gr_sorted[finite]).astype(np.float32)
    return grid.astype(np.float32), gr_grid


def build_alignment_sample(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: GADataConfig,
    tail_class: str = "unknown",
) -> AlignmentSample | None:
    n_raw = len(horizontal)
    n_steps = min(n_raw // cfg.rows_per_step, cfg.max_horizontal_steps)
    if n_steps < 4:
        return None
    raw_limit = n_steps * cfg.rows_per_step
    row_index = np.arange(n_raw, dtype=np.float32)

    tvt_raw = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float32)
    tvt_input_raw = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float32)
    md_raw = _column_or_index(horizontal, "MD", row_index)
    x_raw = _column_or_index(horizontal, "X", np.zeros(n_raw, dtype=np.float32))
    y_raw = _column_or_index(horizontal, "Y", np.zeros(n_raw, dtype=np.float32))
    z_raw = _column_or_index(horizontal, "Z", np.zeros(n_raw, dtype=np.float32))

    comp_tvt = _nanmean_steps(tvt_raw, cfg.rows_per_step)[:n_steps]
    comp_tvt_input = _nanmean_steps(tvt_input_raw, cfg.rows_per_step)[:n_steps]
    comp_gr = _nanmean_steps(gr_raw, cfg.rows_per_step)[:n_steps]
    comp_gr_valid = _valid_frac_steps(gr_raw, cfg.rows_per_step)[:n_steps]
    comp_md = _nanmean_steps(md_raw, cfg.rows_per_step)[:n_steps]
    comp_x = _nanmean_steps(x_raw, cfg.rows_per_step)[:n_steps]
    comp_y = _nanmean_steps(y_raw, cfg.rows_per_step)[:n_steps]
    comp_z = _nanmean_steps(z_raw, cfg.rows_per_step)[:n_steps]

    gr_filled, _ = _fill_nan(comp_gr)
    hidden = ~np.isfinite(comp_tvt_input)
    hidden_idx = np.flatnonzero(hidden)
    known_idx = np.flatnonzero(~hidden)
    if hidden_idx.size == 0 or known_idx.size == 0:
        return None
    last_known_step = int(known_idx[-1])
    last_known_tvt = float(comp_tvt_input[last_known_step])
    if not np.isfinite(last_known_tvt):
        return None
    target_finite = np.isfinite(comp_tvt)
    if not np.any(target_finite & hidden):
        return None

    typewell_tvt, typewell_gr = regular_typewell_grid(
        typewell, cfg.vertical_step_ft, max_bins=cfg.max_typewell_bins
    )
    if len(typewell_tvt) < 4:
        return None

    target_bins = ((comp_tvt - float(typewell_tvt[0])) / cfg.vertical_step_ft).astype(np.float32)
    step_frac = (
        np.arange(n_steps, dtype=np.float32) / max(float(n_steps - 1), 1.0)
    ).astype(np.float32)
    hidden_progress = np.zeros(n_steps, dtype=np.float32)
    first_hidden = int(hidden_idx[0])
    denom = max(float(n_steps - first_hidden - 1), 1.0)
    hidden_progress[first_hidden:] = (
        np.arange(first_hidden, n_steps, dtype=np.float32) - first_hidden
    ) / denom
    tvt_input_delta = np.where(
        np.isfinite(comp_tvt_input), (comp_tvt_input - last_known_tvt) / 100.0, 0.0
    ).astype(np.float32)

    lateral_features = np.stack(
        [
            step_frac,
            _zscore(comp_md),
            _zscore(comp_x),
            _zscore(comp_y),
            _zscore(comp_z),
            _zscore(gr_filled, comp_gr_valid > 0),
            comp_gr_valid.astype(np.float32),
            tvt_input_delta,
            (~hidden).astype(np.float32),
            hidden_progress,
        ],
        axis=1,
    ).astype(np.float32)
    dgr = np.gradient(typewell_gr).astype(np.float32)
    typewell_features = np.stack(
        [
            ((typewell_tvt - float(typewell_tvt[0])) / max(float(typewell_tvt[-1] - typewell_tvt[0]), 1.0)).astype(np.float32),
            _zscore(typewell_gr),
            _zscore(dgr),
            np.ones_like(typewell_tvt, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    raw_hidden = ~np.isfinite(tvt_input_raw[:raw_limit]) & np.isfinite(tvt_raw[:raw_limit])
    raw_steps = (np.arange(raw_limit, dtype=np.int32) // cfg.rows_per_step).astype(np.int32)
    ids = (
        horizontal["id"].astype(str).to_numpy(dtype=object)[:raw_limit]
        if "id" in horizontal.columns
        else np.array([f"{well_id}_{i}" for i in range(raw_limit)], dtype=object)
    )

    return AlignmentSample(
        well_id=well_id,
        lateral_features=lateral_features,
        typewell_features=typewell_features,
        target_bins=target_bins,
        target_tvt=comp_tvt.astype(np.float32),
        comp_tvt_input=comp_tvt_input.astype(np.float32),
        hidden_mask=hidden.astype(bool),
        typewell_tvt=typewell_tvt.astype(np.float32),
        seq_len=n_steps,
        typewell_len=len(typewell_tvt),
        last_known_tvt=last_known_tvt,
        first_hidden_step=first_hidden,
        tail_class=tail_class,
        hidden_row_ids=ids[raw_hidden],
        hidden_row_steps=raw_steps[raw_hidden],
        hidden_row_tvt=tvt_raw[:raw_limit][raw_hidden].astype(np.float32),
    )


def discover_wells(data_dir: Path, k_wells: int = -1) -> list[WellPaths]:
    root = Path(data_dir)
    train_dir = root / "train" if (root / "train").exists() else root
    horizontals = sorted(train_dir.glob("*__horizontal_well.csv"))
    if k_wells > 0:
        horizontals = horizontals[:k_wells]
    wells: list[WellPaths] = []
    suffix = "__horizontal_well.csv"
    for h_path in horizontals:
        well_id = h_path.name[: -len(suffix)]
        t_path = h_path.with_name(f"{well_id}__typewell.csv")
        if t_path.exists():
            wells.append(WellPaths(well_id=well_id, horizontal_path=h_path, typewell_path=t_path))
    if not wells:
        raise FileNotFoundError(f"No train wells found under {data_dir}")
    return wells


def load_alignment_samples(
    data_dir: Path, cfg: GADataConfig, k_wells: int = -1
) -> list[AlignmentSample]:
    samples: list[AlignmentSample] = []
    for well in discover_wells(data_dir, k_wells=k_wells):
        horizontal = pd.read_csv(well.horizontal_path)
        typewell = pd.read_csv(well.typewell_path)
        sample = build_alignment_sample(well.well_id, horizontal, typewell, cfg)
        if sample is not None:
            samples.append(sample)
    return samples


class AlignmentDataset(Dataset):
    def __init__(self, samples: list[AlignmentSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AlignmentSample:
        return self.samples[idx]


def _variant_sample(sample: AlignmentSample, variant: str, rng: np.random.Generator) -> AlignmentSample:
    lateral = sample.lateral_features.copy()
    typewell = sample.typewell_features.copy()
    if variant == "shuffled_gr":
        lateral[:, 5] = rng.permutation(lateral[:, 5])
    elif variant == "zero_gr":
        lateral[:, 5] = 0.0
        lateral[:, 6] = 0.0
    elif variant == "typewell_shuffled":
        typewell[:, 1] = rng.permutation(typewell[:, 1])
        typewell[:, 2] = np.gradient(typewell[:, 1]).astype(np.float32)
    elif variant != "normal":
        raise ValueError(f"Unknown GeoAligner variant: {variant}")
    return replace(sample, lateral_features=lateral, typewell_features=typewell)


def make_variant_samples(samples: list[AlignmentSample], variant: str, seed: int = 1729) -> list[AlignmentSample]:
    rng = np.random.default_rng(seed)
    return [_variant_sample(sample, variant, rng) for sample in samples]


def collate_alignment_samples(samples: list[AlignmentSample]) -> dict[str, torch.Tensor | list[AlignmentSample]]:
    batch = len(samples)
    max_t = max(s.seq_len for s in samples)
    max_h = max(s.typewell_len for s in samples)
    lateral = torch.zeros(batch, max_t, N_LATERAL_FEATURES, dtype=torch.float32)
    typewell = torch.zeros(batch, max_h, N_TYPEWELL_FEATURES, dtype=torch.float32)
    target_bins = torch.full((batch, max_t), -100, dtype=torch.long)
    target_tvt = torch.full((batch, max_t), float("nan"), dtype=torch.float32)
    hidden_mask = torch.zeros(batch, max_t, dtype=torch.bool)
    lat_pad = torch.ones(batch, max_t, dtype=torch.bool)
    tw_pad = torch.ones(batch, max_h, dtype=torch.bool)
    for i, sample in enumerate(samples):
        t = sample.seq_len
        h = sample.typewell_len
        lateral[i, :t] = torch.from_numpy(sample.lateral_features)
        typewell[i, :h] = torch.from_numpy(sample.typewell_features)
        bins = np.rint(sample.target_bins).astype(np.int64)
        valid_bins = np.isfinite(sample.target_bins) & (bins >= 0) & (bins < h)
        target_bins[i, :t] = torch.from_numpy(np.where(valid_bins, bins, -100))
        target_tvt[i, :t] = torch.from_numpy(sample.target_tvt)
        hidden_mask[i, :t] = torch.from_numpy(sample.hidden_mask.astype(bool))
        lat_pad[i, :t] = False
        tw_pad[i, :h] = False
    return {
        "lateral_features": lateral,
        "typewell_features": typewell,
        "target_bins": target_bins,
        "target_tvt": target_tvt,
        "hidden_mask": hidden_mask,
        "lateral_pad_mask": lat_pad,
        "typewell_pad_mask": tw_pad,
        "samples": samples,
    }

