#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor


KAGGLE_INPUT_DIR = Path("/kaggle/input/rogii-wellbore-geology-prediction")
FORMATION_ORDER = {
    "ANCC": 0,
    "ASTNU": 1,
    "ASTNL": 2,
    "EGFDU": 3,
    "EGFDL": 4,
    "BUDA": 5,
}
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "data_dir": None,
        "train_dir": None,
        "test_dir": None,
        "sample_submission": None,
        "max_train_wells": None,
        "max_test_wells": None,
        "target_rows": "hidden_only",
    },
    "features": {
        "tail_windows": [25, 100, 250],
        "rolling_windows": [5, 25, 101],
        "include_typewell": True,
        "include_kaggle_top_signals": True,
        "kaggle_top": {
            "beam_configs": [
                [20.0, 144.0, 2, "cons"],
                [8.0, 64.0, 2, "loose"],
                [14.0, 90.0, 5, "sm5"],
                [25.0, 180.0, 2, "stiff"],
            ],
            "ncc_windows": [8, 15, 25],
            "ncc_stride": 3,
            "dtw_enabled": True,
            "dtw_max_query_points": 700,
            "dtw_max_ref_points": 700,
            "dtw_radius": 35,
            "spatial_k": 10,
            "dense_k": 20,
            "dense_samples_per_well": 60,
        },
    },
    "model": {
        "name": "hist_gradient_boosting",
        "params": {
            "loss": "squared_error",
            "learning_rate": 0.04,
            "max_iter": 350,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.05,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 25,
        },
    },
    "validation": {
        "enabled": True,
        "n_splits": 5,
        "max_wells": 150,
    },
    "postprocess": {
        "residual_weight": "auto",
        "residual_weight_grid": [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
        "residual_clip": 250.0,
    },
    "outputs": {
        "output_dir": "artifacts/hgb",
        "submission_path": "submission.csv",
    },
}


@dataclass(frozen=True)
class WellFeatures:
    well: str
    features: pd.DataFrame
    flat_prediction: np.ndarray
    target: np.ndarray | None
    target_mask: np.ndarray


class KaggleTopContext:
    """Spatial priors inspired by the current public Kaggle top notebooks."""

    def __init__(self, train_paths: list[Path], config: dict[str, Any]) -> None:
        top_cfg = config["features"].get("kaggle_top", {})
        self.spatial_k = int(top_cfg.get("spatial_k", 10))
        self.dense_k = int(top_cfg.get("dense_k", 20))
        self.dense_samples_per_well = int(top_cfg.get("dense_samples_per_well", 60))
        self.formation_tree: cKDTree | None = None
        self.dense_tree: cKDTree | None = None
        self._build_formation_tree(train_paths)
        self._build_dense_ancc_tree(train_paths)

    def _build_formation_tree(self, train_paths: list[Path]) -> None:
        rows: list[dict[str, float | str]] = []
        for path in train_paths:
            try:
                df = pd.read_csv(path, usecols=["X", "Y", *FORMATIONS]).dropna()
            except Exception:
                continue
            if df.empty:
                continue
            row: dict[str, float | str] = {
                "well": well_name(path),
                "x": float(df["X"].median()),
                "y": float(df["Y"].median()),
            }
            for formation in FORMATIONS:
                row[formation] = float(df[formation].median())
            rows.append(row)

        self.formation_df = pd.DataFrame(rows)
        if self.formation_df.empty:
            self.formation_xy = np.empty((0, 2), dtype=float)
            self.formation_values = np.empty((0, len(FORMATIONS)), dtype=float)
            self.formation_wells = np.array([], dtype=str)
            self.formation_scale = np.ones(2, dtype=float)
            return

        self.formation_xy = self.formation_df[["x", "y"]].to_numpy(dtype=float)
        self.formation_values = self.formation_df[FORMATIONS].to_numpy(dtype=float)
        self.formation_wells = self.formation_df["well"].astype(str).to_numpy()
        scale = np.nanstd(self.formation_xy, axis=0)
        self.formation_scale = np.where(scale < 1e-6, 1.0, scale)
        self.formation_tree = cKDTree(self.formation_xy / self.formation_scale)

    def _build_dense_ancc_tree(self, train_paths: list[Path]) -> None:
        xy_parts: list[np.ndarray] = []
        ancc_parts: list[np.ndarray] = []
        well_parts: list[np.ndarray] = []
        for path in train_paths:
            try:
                df = pd.read_csv(path, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if df.empty:
                continue
            take = np.linspace(0, len(df) - 1, min(self.dense_samples_per_well, len(df)), dtype=int)
            sample = df.iloc[take]
            xy_parts.append(sample[["X", "Y"]].to_numpy(dtype=float))
            ancc_parts.append(sample["ANCC"].to_numpy(dtype=float))
            well_parts.append(np.full(len(sample), well_name(path), dtype=object))

        if not xy_parts:
            self.dense_xy = np.empty((0, 2), dtype=float)
            self.dense_ancc = np.empty(0, dtype=float)
            self.dense_wells = np.array([], dtype=object)
            self.dense_scale = np.ones(2, dtype=float)
            return

        self.dense_xy = np.vstack(xy_parts)
        self.dense_ancc = np.concatenate(ancc_parts)
        self.dense_wells = np.concatenate(well_parts)
        scale = np.nanstd(self.dense_xy, axis=0)
        self.dense_scale = np.where(scale < 1e-6, 1.0, scale)
        self.dense_tree = cKDTree(self.dense_xy / self.dense_scale)

    def impute_formations(self, xy: np.ndarray, self_well: str | None) -> tuple[np.ndarray, np.ndarray]:
        if self.formation_tree is None or len(self.formation_values) == 0:
            return (
                np.full((len(xy), len(FORMATIONS)), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
            )
        k_fetch = min(len(self.formation_values), self.spatial_k + 8)
        dist, idx = self.formation_tree.query(xy / self.formation_scale, k=k_fetch)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if len(xy) == 1:
            dist = dist.reshape(1, -1)
            idx = idx.reshape(1, -1)
        if self_well is not None:
            dist = np.where(self.formation_wells[idx] == self_well, np.inf, dist)

        pred = np.empty((len(xy), len(FORMATIONS)), dtype=float)
        nearest_dist = np.empty(len(xy), dtype=float)
        global_mean = np.nanmean(self.formation_values, axis=0)
        for row in range(len(xy)):
            order = np.argsort(dist[row])[: self.spatial_k]
            valid = np.isfinite(dist[row, order])
            if not valid.any():
                pred[row] = global_mean
                nearest_dist[row] = np.nan
                continue
            chosen = order[valid]
            weights = 1.0 / (dist[row, chosen] + 1e-3)
            weights /= weights.sum()
            pred[row] = weights @ self.formation_values[idx[row, chosen]]
            nearest_dist[row] = float(np.nanmin(dist[row, chosen]))
        return pred, nearest_dist

    def impute_dense_ancc(self, xy: np.ndarray, self_well: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dense_tree is None or len(self.dense_ancc) == 0:
            return (
                np.full(len(xy), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
            )
        k_fetch = min(len(self.dense_ancc), max(self.dense_k + 80, self.dense_k))
        dist, idx = self.dense_tree.query(xy / self.dense_scale, k=k_fetch)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if len(xy) == 1:
            dist = dist.reshape(1, -1)
            idx = idx.reshape(1, -1)
        if self_well is not None:
            dist = np.where(self.dense_wells[idx] == self_well, np.inf, dist)

        pred = np.empty(len(xy), dtype=float)
        std = np.empty(len(xy), dtype=float)
        nearest_dist = np.empty(len(xy), dtype=float)
        global_mean = float(np.nanmean(self.dense_ancc))
        for row in range(len(xy)):
            order = np.argsort(dist[row])[: self.dense_k]
            valid = np.isfinite(dist[row, order])
            if not valid.any():
                pred[row] = global_mean
                std[row] = np.nan
                nearest_dist[row] = np.nan
                continue
            chosen = order[valid]
            values = self.dense_ancc[idx[row, chosen]]
            weights = 1.0 / (dist[row, chosen] + 1e-3)
            weights /= weights.sum()
            mean = float(weights @ values)
            pred[row] = mean
            std[row] = float(np.sqrt(weights @ ((values - mean) ** 2)))
            nearest_dist[row] = float(np.nanmin(dist[row, chosen]))
        return pred, std, nearest_dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ROGII TVT residual model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hgb.yml"),
        help="YAML config path.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data.data_dir.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override outputs.output_dir.")
    parser.add_argument("--submission", type=Path, default=None, help="Override outputs.submission_path.")
    parser.add_argument("--no-cv", action="store_true", help="Skip group CV and train final model only.")
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        config = deep_update(config, loaded)
    else:
        raise FileNotFoundError(f"Config file not found: {path}")
    return config


def path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def resolve_data_dir(config: dict[str, Any]) -> Path:
    configured = path_or_none(config["data"].get("data_dir"))
    if configured is not None:
        return configured
    if KAGGLE_INPUT_DIR.exists():
        return KAGGLE_INPUT_DIR
    return Path("data")


def well_name(path: Path) -> str:
    return path.name.split("__horizontal_well.csv", 1)[0]


def limited(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    return paths[: int(limit)]


def resolve_train_dir(data_dir: Path, config: dict[str, Any]) -> Path:
    configured = path_or_none(config["data"].get("train_dir"))
    candidates = [configured, data_dir / "train"]
    for path in candidates:
        if path is not None and path.exists():
            return path
    raise FileNotFoundError(
        f"No train directory found under {data_dir}. Run `make unzip-data` first, "
        "or use configs/quick.yml for the small public sample."
    )


def resolve_test_dir(data_dir: Path, config: dict[str, Any]) -> Path:
    configured = path_or_none(config["data"].get("test_dir"))
    candidates = [configured, data_dir / "test"]
    for path in candidates:
        if path is not None and path.exists() and list(path.glob("*__horizontal_well.csv")):
            return path
    raise FileNotFoundError(
        f"No test horizontal well files found under {data_dir}. Run `make unzip-data` first, "
        "or use configs/quick.yml for the small public sample."
    )


def resolve_sample_submission(data_dir: Path, test_dir: Path, config: dict[str, Any]) -> Path | None:
    configured = path_or_none(config["data"].get("sample_submission"))
    candidates = [
        configured,
        data_dir / "sample_submission.csv",
        test_dir / "sample_submission.csv",
        data_dir / "public_test" / "sample_submission.csv",
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    matches = sorted(data_dir.rglob("sample_submission.csv"))
    return matches[0] if matches else None


def horizontal_files(directory: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(directory.glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No horizontal well files in {directory}")
    return limited(paths, limit)


def typewell_path(horizontal_path: Path) -> Path | None:
    name = well_name(horizontal_path)
    root = horizontal_path.parent.parent
    candidates = [
        horizontal_path.with_name(f"{name}__typewell.csv"),
        root / f"{name}__typewell.csv",
        root / "train" / f"{name}__typewell.csv",
        root / "test" / f"{name}__typewell.csv",
        root / "public_train" / f"{name}__typewell.csv",
        root / "public_test" / f"{name}__typewell.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def as_float_array(series: pd.Series | np.ndarray, default: float = np.nan) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if default == default:
        values[~np.isfinite(values)] = default
    return values


def centered_rolling(values: np.ndarray, window: int, statistic: str) -> np.ndarray:
    series = pd.Series(values)
    rolled = series.rolling(window=window, center=True, min_periods=1)
    if statistic == "mean":
        return rolled.mean().to_numpy(dtype=float)
    if statistic == "std":
        return rolled.std().fillna(0.0).to_numpy(dtype=float)
    if statistic == "min":
        return rolled.min().to_numpy(dtype=float)
    if statistic == "max":
        return rolled.max().to_numpy(dtype=float)
    raise ValueError(f"Unknown rolling statistic: {statistic}")


def safe_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2:
        return np.zeros_like(y)
    dx = np.gradient(x)
    dy = np.gradient(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = dy / dx
    grad[~np.isfinite(grad)] = 0.0
    return grad


def tail_stat(values: np.ndarray, known: np.ndarray, window: int, statistic: str) -> float:
    tail = values[known][-window:]
    if len(tail) == 0:
        return 0.0
    if statistic == "median":
        return float(np.nanmedian(tail))
    if statistic == "mean":
        return float(np.nanmean(tail))
    if statistic == "std":
        return float(np.nanstd(tail))
    raise ValueError(f"Unknown tail statistic: {statistic}")


def tail_slope(x: np.ndarray, y: np.ndarray, known: np.ndarray, window: int) -> float:
    idx = np.flatnonzero(known)[-window:]
    if len(idx) < 2:
        return 0.0
    xx = x[idx]
    yy = y[idx]
    valid = np.isfinite(xx) & np.isfinite(yy)
    if valid.sum() < 2:
        return 0.0
    xx = xx[valid]
    yy = yy[valid]
    if float(np.nanstd(xx)) == 0.0:
        return 0.0
    return float(np.polyfit(xx - xx.mean(), yy, 1)[0])


def collapse_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    counts = np.bincount(inverse)
    y_mean = np.bincount(inverse, weights=y_sorted) / counts
    return unique_x, y_mean


def flat_tvt_prediction(md: np.ndarray, tvt_input: np.ndarray, tail_window: int) -> np.ndarray:
    known = np.isfinite(md) & np.isfinite(tvt_input)
    if known.sum() == 0:
        return np.zeros(len(md), dtype=float)
    if known.sum() == 1:
        return np.full(len(md), float(tvt_input[known][0]), dtype=float)

    x_known, y_known = collapse_duplicate_x(md[known], tvt_input[known])
    pred = np.interp(md, x_known, y_known)
    window = max(1, min(tail_window, len(y_known)))
    pred[md < x_known[0]] = float(np.nanmedian(y_known[:window]))
    pred[md > x_known[-1]] = float(np.nanmedian(y_known[-window:]))
    pred[~np.isfinite(md)] = float(np.nanmedian(y_known[-window:]))
    return pred


def fill_numeric(values: np.ndarray, fallback: float) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.interpolate(limit_direction="both").fillna(fallback).to_numpy(dtype=float)


def nearest_index(sorted_values: np.ndarray, value: float) -> int:
    pos = int(np.searchsorted(sorted_values, value, side="left"))
    if pos >= len(sorted_values):
        return len(sorted_values) - 1
    if pos > 0 and abs(sorted_values[pos - 1] - value) <= abs(sorted_values[pos] - value):
        return pos - 1
    return pos


def smooth_for_alignment(values: np.ndarray, radius: int, fallback: float) -> np.ndarray:
    filled = fill_numeric(values, fallback)
    if radius <= 0:
        return filled
    return (
        pd.Series(filled)
        .rolling(radius * 2 + 1, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def greedy_beam_signal(
    gr_query: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    move_cost: float,
    emit_scale: float,
    smooth_radius: int,
) -> np.ndarray:
    """Fast deterministic proxy for the public top-solution beam-search signal."""
    if len(gr_query) == 0:
        return np.array([], dtype=float)
    smoothed_gr = smooth_for_alignment(gr_query, smooth_radius, float(np.nanmean(tw_gr)))
    idx = nearest_index(tw_tvt, start_tvt)
    path = np.empty(len(smoothed_gr), dtype=int)
    for i, gr_value in enumerate(smoothed_gr):
        candidates = np.arange(max(0, idx - 2), min(len(tw_gr), idx + 3))
        costs = ((gr_value - tw_gr[candidates]) ** 2) / max(float(emit_scale), 1e-6)
        costs += float(move_cost) * np.abs(candidates - idx)
        idx = int(candidates[int(np.argmin(costs))])
        path[i] = idx
    return tw_tvt[path].astype(float)


def multi_scale_ncc(
    known_gr: np.ndarray,
    known_tvt: np.ndarray,
    query_gr: np.ndarray,
    windows: list[int],
    stride: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if len(query_gr) == 0:
        return result
    known_filled = smooth_for_alignment(known_gr, 2, float(np.nanmean(known_gr)))
    query_filled = smooth_for_alignment(query_gr, 2, float(np.nanmean(known_filled)))

    for half_window in windows:
        win = 2 * int(half_window) + 1
        if len(known_filled) < win + 1:
            result[f"ncc_{half_window}_tvt"] = np.full(len(query_filled), known_tvt[-1], dtype=float)
            result[f"ncc_{half_window}_score"] = np.zeros(len(query_filled), dtype=float)
            continue

        starts = np.arange(0, len(known_filled) - win + 1, max(int(stride), 1), dtype=int)
        window_offsets = np.arange(win, dtype=int)
        known_windows = known_filled[starts[:, None] + window_offsets[None, :]]
        known_norm = (known_windows - known_windows.mean(axis=1, keepdims=True)) / (
            known_windows.std(axis=1, keepdims=True) + 1e-6
        )

        padded_query = np.pad(query_filled, half_window, mode="edge")
        query_windows = padded_query[np.arange(len(query_filled))[:, None] + window_offsets[None, :]]
        query_norm = (query_windows - query_windows.mean(axis=1, keepdims=True)) / (
            query_windows.std(axis=1, keepdims=True) + 1e-6
        )

        scores = query_norm @ known_norm.T / win
        best = np.argmax(scores, axis=1)
        centers = np.clip(starts[best] + half_window, 0, len(known_tvt) - 1)
        result[f"ncc_{half_window}_tvt"] = known_tvt[centers].astype(float)
        result[f"ncc_{half_window}_score"] = np.max(scores, axis=1).astype(float)
    return result


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def lowres_dtw_signal(
    full_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    max_query_points: int,
    max_ref_points: int,
    radius: int,
) -> np.ndarray:
    """Low-resolution Sakoe-Chiba DTW signal, used as a cheap alignment feature."""
    if len(full_gr) == 0 or len(tw_gr) == 0:
        return np.full(len(full_gr), np.nan, dtype=float)

    q_idx = downsample_indices(len(full_gr), max_query_points)
    r_idx = downsample_indices(len(tw_gr), max_ref_points)
    q = full_gr[q_idx]
    r = tw_gr[r_idx]
    q = (q - np.nanmean(q)) / (np.nanstd(q) + 1e-6)
    r = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-6)

    n = len(q)
    m = len(r)
    inf = 1e18
    dp = np.full((n, m), inf, dtype=float)
    parent = np.full((n, m), -1, dtype=np.int8)
    slope = (m - 1) / max(n - 1, 1)
    radius = max(int(radius), 1)

    for i in range(n):
        center = int(round(i * slope))
        lo = max(0, center - radius)
        hi = min(m - 1, center + radius)
        for j in range(lo, hi + 1):
            cost = (q[i] - r[j]) ** 2
            if i == 0 and j == 0:
                dp[i, j] = cost
                continue
            choices: list[tuple[float, int]] = []
            if i > 0 and j > 0:
                choices.append((dp[i - 1, j - 1], 0))
            if i > 0:
                choices.append((dp[i - 1, j], 1))
            if j > 0:
                choices.append((dp[i, j - 1], 2))
            prev_cost, prev_code = min(choices, key=lambda item: item[0])
            dp[i, j] = cost + prev_cost
            parent[i, j] = prev_code

    j_end = int(np.nanargmin(dp[-1]))
    i = n - 1
    j = j_end
    j_for_i = np.zeros(n, dtype=int)
    while i >= 0 and j >= 0:
        j_for_i[i] = j
        code = parent[i, j]
        if i == 0 and j == 0:
            break
        if code == 0:
            i -= 1
            j -= 1
        elif code == 1:
            i -= 1
        else:
            j -= 1
    coarse_tvt = tw_tvt[r_idx[j_for_i]]
    return np.interp(np.arange(len(full_gr)), q_idx, coarse_tvt).astype(float)


def read_typewell_features(path: Path | None, horizontal_gr: np.ndarray, flat_pred: np.ndarray) -> dict[str, np.ndarray | float]:
    n = len(horizontal_gr)
    if path is None:
        return {
            "typewell_tvt_min": 0.0,
            "typewell_tvt_max": 0.0,
            "typewell_tvt_range": 0.0,
            "typewell_gr_mean": 0.0,
            "typewell_gr_std": 0.0,
            "typewell_nearest_tvt_by_gr": np.zeros(n, dtype=float),
            "typewell_nearest_gr_diff": np.zeros(n, dtype=float),
            "typewell_nearest_geology_code": np.full(n, -1.0, dtype=float),
            "typewell_nearest_tvt_minus_flat": np.zeros(n, dtype=float),
        }

    typewell = pd.read_csv(path)
    tvt = as_float_array(typewell.get("TVT", pd.Series(dtype=float)))
    gr = as_float_array(typewell.get("GR", pd.Series(dtype=float)))
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if valid.sum() == 0:
        return read_typewell_features(None, horizontal_gr, flat_pred)

    tvt_valid = tvt[valid]
    gr_valid = gr[valid]
    geology = typewell.get("Geology")
    if geology is None:
        geology_codes = np.full(len(typewell), -1.0, dtype=float)
    else:
        geology_codes = geology.astype(str).map(FORMATION_ORDER).fillna(-1).to_numpy(dtype=float)
    geology_valid = geology_codes[valid]

    order = np.argsort(gr_valid)
    gr_sorted = gr_valid[order]
    tvt_sorted = tvt_valid[order]
    geology_sorted = geology_valid[order]

    positions = np.searchsorted(gr_sorted, horizontal_gr, side="left")
    left = np.clip(positions - 1, 0, len(gr_sorted) - 1)
    right = np.clip(positions, 0, len(gr_sorted) - 1)
    choose_right = np.abs(gr_sorted[right] - horizontal_gr) < np.abs(gr_sorted[left] - horizontal_gr)
    nearest = np.where(choose_right, right, left)
    nearest_tvt = tvt_sorted[nearest]
    nearest_gr = gr_sorted[nearest]
    nearest_geology = geology_sorted[nearest]

    return {
        "typewell_tvt_min": float(np.nanmin(tvt_valid)),
        "typewell_tvt_max": float(np.nanmax(tvt_valid)),
        "typewell_tvt_range": float(np.nanmax(tvt_valid) - np.nanmin(tvt_valid)),
        "typewell_gr_mean": float(np.nanmean(gr_valid)),
        "typewell_gr_std": float(np.nanstd(gr_valid)),
        "typewell_nearest_tvt_by_gr": nearest_tvt,
        "typewell_nearest_gr_diff": np.abs(nearest_gr - horizontal_gr),
        "typewell_nearest_geology_code": nearest_geology,
        "typewell_nearest_tvt_minus_flat": nearest_tvt - flat_pred,
    }


def empty_top_signal_features(n: int) -> dict[str, np.ndarray | float]:
    keys = [
        "kg_hidden_row",
        "kg_beam_mean_minus_flat",
        "kg_beam_std",
        "kg_ncc_mean_minus_flat",
        "kg_ncc_score_mean",
        "kg_dtw_minus_flat",
        "kg_dtw_vs_beam",
        "kg_signal_mean_minus_flat",
        "kg_signal_std",
        "kg_form_ancc_minus_flat",
        "kg_form_mean_minus_flat",
        "kg_form_std",
        "kg_form_range",
        "kg_form_knn_dist",
        "kg_dense_ancc_minus_flat",
        "kg_dense_ancc_std",
        "kg_dense_ancc_dist",
        "kg_dense_vs_form",
    ]
    return {key: np.zeros(n, dtype=float) for key in keys}


def build_kaggle_top_signal_features(
    df: pd.DataFrame,
    horizontal_path: Path,
    context: KaggleTopContext | None,
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    flat_pred: np.ndarray,
    config: dict[str, Any],
    train: bool,
) -> dict[str, np.ndarray | float]:
    n = len(df)
    features = empty_top_signal_features(n)
    top_cfg = config["features"].get("kaggle_top", {})
    known = np.isfinite(tvt_input)
    hidden = ~known
    hidden_idx = np.flatnonzero(hidden)
    known_idx = np.flatnonzero(known)
    if len(hidden_idx) == 0 or len(known_idx) < 10:
        return features

    tw_path = typewell_path(horizontal_path)
    if tw_path is None:
        return features
    typewell = pd.read_csv(tw_path).sort_values("TVT")
    if "TVT" not in typewell.columns or "GR" not in typewell.columns:
        return features
    tw_tvt = as_float_array(typewell["TVT"])
    tw_gr = as_float_array(typewell["GR"])
    valid_tw = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[valid_tw]
    tw_gr = tw_gr[valid_tw]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    if len(tw_tvt) < 5:
        return features

    well = well_name(horizontal_path)
    self_well = well if train else None
    last_known_idx = int(known_idx[-1])
    last_tvt = float(tvt_input[last_known_idx])
    gr_full = fill_numeric(gr, float(np.nanmean(tw_gr)))
    hidden_gr = gr_full[hidden_idx]
    known_gr = gr_full[known_idx]
    known_tvt = tvt_input[known_idx]

    beam_signals: list[np.ndarray] = []
    for move_cost, emit_scale, smooth_radius, tag in top_cfg.get("beam_configs", []):
        signal = greedy_beam_signal(
            hidden_gr,
            tw_tvt,
            tw_gr,
            last_tvt,
            float(move_cost),
            float(emit_scale),
            int(smooth_radius),
        )
        beam_signals.append(signal)
        features[f"kg_beam_{tag}_minus_flat"] = np.zeros(n, dtype=float)
        features[f"kg_beam_{tag}_minus_flat"][hidden_idx] = signal - flat_pred[hidden_idx]
        features[f"kg_beam_{tag}_minus_last"] = np.zeros(n, dtype=float)
        features[f"kg_beam_{tag}_minus_last"][hidden_idx] = signal - last_tvt

    if beam_signals:
        beam_matrix = np.vstack(beam_signals).T
        beam_mean = np.nanmean(beam_matrix, axis=1)
        features["kg_beam_mean_minus_flat"][hidden_idx] = beam_mean - flat_pred[hidden_idx]
        features["kg_beam_std"][hidden_idx] = np.nanstd(beam_matrix, axis=1)
    else:
        beam_mean = flat_pred[hidden_idx]

    ncc = multi_scale_ncc(
        known_gr,
        known_tvt,
        hidden_gr,
        [int(item) for item in top_cfg.get("ncc_windows", [8, 15, 25])],
        int(top_cfg.get("ncc_stride", 3)),
    )
    ncc_signals: list[np.ndarray] = []
    ncc_scores: list[np.ndarray] = []
    for key, value in ncc.items():
        full = np.zeros(n, dtype=float)
        if key.endswith("_tvt"):
            full[hidden_idx] = value - flat_pred[hidden_idx]
            features[f"kg_{key}_minus_flat"] = full
            ncc_signals.append(value)
        else:
            full[hidden_idx] = value
            features[f"kg_{key}"] = full
            ncc_scores.append(value)
    if ncc_signals:
        ncc_matrix = np.vstack(ncc_signals).T
        features["kg_ncc_mean_minus_flat"][hidden_idx] = np.nanmean(ncc_matrix, axis=1) - flat_pred[hidden_idx]
    if ncc_scores:
        features["kg_ncc_score_mean"][hidden_idx] = np.nanmean(np.vstack(ncc_scores).T, axis=1)

    if top_cfg.get("dtw_enabled", True):
        dtw_signal = lowres_dtw_signal(
            gr_full,
            tw_tvt,
            tw_gr,
            int(top_cfg.get("dtw_max_query_points", 700)),
            int(top_cfg.get("dtw_max_ref_points", 700)),
            int(top_cfg.get("dtw_radius", 35)),
        )
        features["kg_dtw_minus_flat"][hidden_idx] = dtw_signal[hidden_idx] - flat_pred[hidden_idx]
        features["kg_dtw_vs_beam"][hidden_idx] = dtw_signal[hidden_idx] - beam_mean
        signal_stack = [beam_mean, dtw_signal[hidden_idx]]
    else:
        signal_stack = [beam_mean]

    if context is not None:
        xy_hidden = np.column_stack([x[hidden_idx], y[hidden_idx]])
        form_hidden, form_dist = context.impute_formations(xy_hidden, self_well)
        xy_known = np.column_stack([x[known_idx], y[known_idx]])
        form_known, _ = context.impute_formations(xy_known, self_well)
        if form_hidden.shape[1] == len(FORMATIONS) and np.isfinite(form_hidden).any():
            form_signals = []
            for formation_idx, formation in enumerate(FORMATIONS):
                residual_base = known_tvt + z[known_idx] - form_known[:, formation_idx]
                b = float(np.nanmedian(residual_base)) if np.isfinite(residual_base).any() else 0.0
                signal = -z[hidden_idx] + form_hidden[:, formation_idx] + b
                form_signals.append(signal)
                col = f"kg_form_{formation}_minus_flat"
                features[col] = np.zeros(n, dtype=float)
                features[col][hidden_idx] = signal - flat_pred[hidden_idx]
            form_matrix = np.vstack(form_signals).T
            form_mean = np.nanmean(form_matrix, axis=1)
            features["kg_form_ancc_minus_flat"][hidden_idx] = form_matrix[:, 0] - flat_pred[hidden_idx]
            features["kg_form_mean_minus_flat"][hidden_idx] = form_mean - flat_pred[hidden_idx]
            features["kg_form_std"][hidden_idx] = np.nanstd(form_matrix, axis=1)
            features["kg_form_range"][hidden_idx] = np.nanmax(form_matrix, axis=1) - np.nanmin(form_matrix, axis=1)
            features["kg_form_knn_dist"][hidden_idx] = form_dist
            signal_stack.append(form_mean)
        else:
            form_mean = flat_pred[hidden_idx]

        dense_ancc, dense_std, dense_dist = context.impute_dense_ancc(xy_hidden, self_well)
        dense_known, _, _ = context.impute_dense_ancc(xy_known, self_well)
        dense_residual = known_tvt + z[known_idx] - dense_known
        dense_b = float(np.nanmedian(dense_residual)) if np.isfinite(dense_residual).any() else 0.0
        dense_signal = -z[hidden_idx] + dense_ancc + dense_b
        features["kg_dense_ancc_minus_flat"][hidden_idx] = dense_signal - flat_pred[hidden_idx]
        features["kg_dense_ancc_std"][hidden_idx] = dense_std
        features["kg_dense_ancc_dist"][hidden_idx] = dense_dist
        features["kg_dense_vs_form"][hidden_idx] = dense_signal - form_mean
        signal_stack.append(dense_signal)

    signal_matrix = np.vstack(signal_stack).T
    features["kg_signal_mean_minus_flat"][hidden_idx] = np.nanmean(signal_matrix, axis=1) - flat_pred[hidden_idx]
    features["kg_signal_std"][hidden_idx] = np.nanstd(signal_matrix, axis=1)
    features["kg_hidden_row"][hidden_idx] = 1.0
    return features


def build_target_mask(df: pd.DataFrame, target_rows: str) -> np.ndarray:
    has_target = "TVT" in df.columns and pd.to_numeric(df["TVT"], errors="coerce").notna().to_numpy()
    if target_rows == "all":
        return has_target
    if target_rows != "hidden_only":
        raise ValueError("data.target_rows must be 'hidden_only' or 'all'")
    hidden = pd.to_numeric(df["TVT_input"], errors="coerce").isna().to_numpy()
    return has_target & hidden


def build_well_features(
    horizontal_path: Path,
    config: dict[str, Any],
    train: bool,
    top_context: KaggleTopContext | None = None,
) -> WellFeatures:
    df = pd.read_csv(horizontal_path)
    n = len(df)
    well = well_name(horizontal_path)
    idx = np.arange(n, dtype=float)

    md = as_float_array(df.get("MD", pd.Series(np.arange(n))))
    x = as_float_array(df.get("X", pd.Series(np.zeros(n))))
    y = as_float_array(df.get("Y", pd.Series(np.zeros(n))))
    z = as_float_array(df.get("Z", pd.Series(np.zeros(n))))
    gr = as_float_array(df.get("GR", pd.Series(np.zeros(n))), default=np.nan)
    tvt_input = as_float_array(df.get("TVT_input", pd.Series(np.full(n, np.nan))), default=np.nan)

    tail_windows = [int(item) for item in config["features"]["tail_windows"]]
    rolling_windows = [int(item) for item in config["features"]["rolling_windows"]]
    flat_pred = flat_tvt_prediction(md, tvt_input, max(tail_windows))

    known = np.isfinite(tvt_input)
    known_idx = np.flatnonzero(known)
    if len(known_idx):
        first_known = int(known_idx[0])
        last_known = int(known_idx[-1])
        anchor_idx = np.full(n, last_known, dtype=float)
        anchor_tvt = np.full(n, float(np.nanmedian(tvt_input[known_idx[-max(tail_windows):]])), dtype=float)
        first_tvt = float(tvt_input[first_known])
        last_tvt = float(tvt_input[last_known])
    else:
        first_known = 0
        last_known = 0
        anchor_idx = np.zeros(n, dtype=float)
        anchor_tvt = np.zeros(n, dtype=float)
        first_tvt = 0.0
        last_tvt = 0.0

    prev_known_idx = pd.Series(np.where(known, idx, np.nan)).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    prev_known_tvt = pd.Series(tvt_input).ffill().bfill().fillna(float(np.nanmedian(flat_pred))).to_numpy(dtype=float)
    prev_known_md = np.interp(prev_known_idx, idx, md)
    prev_known_z = np.interp(prev_known_idx, idx, z)

    features: dict[str, np.ndarray | float] = {
        "idx": idx,
        "idx_frac": idx / max(n - 1, 1),
        "well_n_rows": float(n),
        "known_count": float(len(known_idx)),
        "known_frac": float(len(known_idx) / max(n, 1)),
        "md": md,
        "x": x,
        "y": y,
        "z": z,
        "gr": gr,
        "flat_tvt": flat_pred,
        "tvt_input_isna": (~known).astype(float),
        "first_known_idx": float(first_known),
        "last_known_idx": float(last_known),
        "first_known_tvt": first_tvt,
        "last_known_tvt": last_tvt,
        "anchor_tvt": anchor_tvt,
        "idx_from_last_known": idx - float(last_known),
        "md_from_start": md - md[0],
        "md_from_last_known": md - md[last_known],
        "z_from_last_known": z - z[last_known],
        "x_from_last_known": x - x[last_known],
        "y_from_last_known": y - y[last_known],
        "xy_dist_from_last_known": np.sqrt((x - x[last_known]) ** 2 + (y - y[last_known]) ** 2),
        "prev_known_tvt": prev_known_tvt,
        "idx_from_prev_known": idx - prev_known_idx,
        "md_from_prev_known": md - prev_known_md,
        "z_from_prev_known": z - prev_known_z,
        "gr_from_well_mean": gr - float(np.nanmean(gr)),
        "gr_grad_md": safe_gradient(gr, md),
        "z_grad_md": safe_gradient(z, md),
        "x_grad_md": safe_gradient(x, md),
        "y_grad_md": safe_gradient(y, md),
    }

    for window in tail_windows:
        features[f"tail_tvt_median_{window}"] = tail_stat(tvt_input, known, window, "median")
        features[f"tail_tvt_mean_{window}"] = tail_stat(tvt_input, known, window, "mean")
        features[f"tail_tvt_std_{window}"] = tail_stat(tvt_input, known, window, "std")
        features[f"tail_tvt_slope_md_{window}"] = tail_slope(md, tvt_input, known, window)
        features[f"tail_z_slope_md_{window}"] = tail_slope(md, z, np.isfinite(md) & np.isfinite(z), window)
        features[f"tail_gr_mean_{window}"] = tail_stat(gr, np.isfinite(gr), window, "mean")
        features[f"tail_gr_std_{window}"] = tail_stat(gr, np.isfinite(gr), window, "std")

    for window in rolling_windows:
        gr_mean = centered_rolling(gr, window, "mean")
        gr_std = centered_rolling(gr, window, "std")
        features[f"gr_roll_mean_{window}"] = gr_mean
        features[f"gr_roll_std_{window}"] = gr_std
        features[f"gr_roll_min_{window}"] = centered_rolling(gr, window, "min")
        features[f"gr_roll_max_{window}"] = centered_rolling(gr, window, "max")
        features[f"gr_minus_roll_mean_{window}"] = gr - gr_mean
        features[f"z_roll_mean_{window}"] = centered_rolling(z, window, "mean")
        features[f"z_minus_roll_mean_{window}"] = z - centered_rolling(z, window, "mean")

    if config["features"].get("include_typewell", True):
        features.update(read_typewell_features(typewell_path(horizontal_path), gr, flat_pred))

    if config["features"].get("include_kaggle_top_signals", False):
        features.update(
            build_kaggle_top_signal_features(
                df,
                horizontal_path,
                top_context,
                md,
                x,
                y,
                z,
                gr,
                tvt_input,
                flat_pred,
                config,
                train,
            )
        )

    feature_frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
    target_mask = build_target_mask(df, config["data"].get("target_rows", "hidden_only")) if train else np.zeros(n, dtype=bool)
    target = as_float_array(df["TVT"], default=np.nan) if train and "TVT" in df.columns else None
    return WellFeatures(well=well, features=feature_frame, flat_prediction=flat_pred, target=target, target_mask=target_mask)


def build_training_table(
    paths: list[Path],
    config: dict[str, Any],
    top_context: KaggleTopContext | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_parts: list[pd.DataFrame] = []
    residual_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    flat_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []

    for i, path in enumerate(paths, start=1):
        wf = build_well_features(path, config, train=True, top_context=top_context)
        mask = wf.target_mask
        if wf.target is None or not mask.any():
            continue
        feature_parts.append(wf.features.loc[mask].astype("float32"))
        true_values = wf.target[mask]
        flat_values = wf.flat_prediction[mask]
        residual_parts.append((true_values - flat_values).astype("float32"))
        group_parts.append(np.full(mask.sum(), wf.well))
        flat_parts.append(flat_values.astype("float32"))
        true_parts.append(true_values.astype("float32"))
        if i % 100 == 0 or i == len(paths):
            print(f"Loaded train wells: {i}/{len(paths)}")

    if not feature_parts:
        raise ValueError("No training rows were built. Check data.target_rows and train files.")

    X = pd.concat(feature_parts, axis=0, ignore_index=True)
    residual = np.concatenate(residual_parts)
    groups = np.concatenate(group_parts)
    flat = np.concatenate(flat_parts)
    y_true = np.concatenate(true_parts)
    return X, residual, groups, flat, y_true


def make_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    if config["model"].get("name") != "hist_gradient_boosting":
        raise ValueError("Only model.name=hist_gradient_boosting is currently supported.")
    params = dict(config["model"].get("params", {}))
    params["random_state"] = seed
    return HistGradientBoostingRegressor(**params)


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def shuffled_group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_splits = max(2, min(n_splits, len(unique_groups)))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        valid_groups = set(unique_groups[fold::n_splits])
        valid_idx = np.array([group in valid_groups for group in groups])
        train_idx = ~valid_idx
        folds.append((train_idx, valid_idx))
    return folds


def apply_postprocess(
    flat: np.ndarray,
    residual: np.ndarray,
    config: dict[str, Any],
    residual_weight: float | None = None,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    clip_value = config["postprocess"].get("residual_clip")
    if clip_value not in (None, ""):
        clip = float(clip_value)
        residual = np.clip(residual, -clip, clip)
    if residual_weight is None:
        configured = config["postprocess"].get("residual_weight", 1.0)
        residual_weight = 1.0 if configured == "auto" else float(configured)
    weight = float(residual_weight)
    return np.asarray(flat, dtype=float) + weight * residual


def tune_residual_weight(
    flat: np.ndarray,
    residual_pred: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, list[dict[str, float]]]:
    configured = config["postprocess"].get("residual_weight", 1.0)
    if configured != "auto":
        weight = float(configured)
        pred = apply_postprocess(flat, residual_pred, config, residual_weight=weight)
        return weight, [{"weight": weight, "rmse": rmse(pred, y_true)}]

    grid = config["postprocess"].get("residual_weight_grid") or [0.0, 0.25, 0.5, 0.75, 1.0]
    scores: list[dict[str, float]] = []
    for weight in grid:
        weight = float(weight)
        pred = apply_postprocess(flat, residual_pred, config, residual_weight=weight)
        scores.append({"weight": weight, "rmse": rmse(pred, y_true)})
    best = min(scores, key=lambda item: item["rmse"])
    return float(best["weight"]), scores


def run_cv(
    X: pd.DataFrame,
    residual: np.ndarray,
    groups: np.ndarray,
    flat: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    validation = config["validation"]
    unique_wells = np.array(sorted(set(groups)))
    max_wells = validation.get("max_wells")
    if max_wells not in (None, "") and len(unique_wells) > int(max_wells):
        rng = np.random.default_rng(seed)
        selected = set(rng.choice(unique_wells, size=int(max_wells), replace=False))
        cv_mask = np.array([group in selected for group in groups])
    else:
        cv_mask = np.ones(len(groups), dtype=bool)

    X_cv = X.loc[cv_mask].reset_index(drop=True)
    residual_cv = residual[cv_mask]
    groups_cv = groups[cv_mask]
    flat_cv = flat[cv_mask]
    y_true_cv = y_true[cv_mask]

    unique_cv_groups = sorted(set(groups_cv))
    if len(unique_cv_groups) < 2:
        print("Skipping CV: need at least two wells.")
        return {"enabled": False, "reason": "not enough groups"}

    folds = shuffled_group_folds(groups_cv, int(validation.get("n_splits", 5)), seed)
    oof_residual = np.zeros(len(X_cv), dtype=float)
    fold_ids = np.zeros(len(X_cv), dtype=int)
    fold_metrics: list[dict[str, Any]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(
            f"CV fold {fold_id}/{len(folds)}: "
            f"train_rows={train_idx.sum():,} valid_rows={valid_idx.sum():,}"
        )
        model = make_model(config, seed + fold_id)
        model.fit(X_cv.loc[train_idx], residual_cv[train_idx])
        residual_pred = model.predict(X_cv.loc[valid_idx])
        oof_residual[valid_idx] = residual_pred
        fold_ids[valid_idx] = fold_id

    best_weight, weight_scores = tune_residual_weight(flat_cv, oof_residual, y_true_cv, config)
    oof_pred = apply_postprocess(flat_cv, oof_residual, config, residual_weight=best_weight)
    for fold_id in range(1, len(folds) + 1):
        valid_idx = fold_ids == fold_id
        fold_rmse = rmse(oof_pred[valid_idx], y_true_cv[valid_idx])
        fold_metrics.append(
            {
                "fold": fold_id,
                "rmse": fold_rmse,
                "valid_rows": int(valid_idx.sum()),
                "valid_wells": int(len(set(groups_cv[valid_idx]))),
            }
        )
        print(f"  fold_{fold_id}_rmse_at_weight_{best_weight:g}={fold_rmse:.5f}")

    overall_rmse = rmse(oof_pred, y_true_cv)
    flat_rmse = rmse(flat_cv, y_true_cv)
    print(f"CV RMSE: {overall_rmse:.5f}")
    print(f"Flat baseline RMSE on same rows: {flat_rmse:.5f}")
    print(f"Best residual_weight: {best_weight:g}")
    return {
        "enabled": True,
        "rmse": overall_rmse,
        "flat_rmse": flat_rmse,
        "best_residual_weight": best_weight,
        "residual_weight_scores": weight_scores,
        "rows": int(len(X_cv)),
        "wells": int(len(unique_cv_groups)),
        "folds": fold_metrics,
    }


def predict_test(
    model: HistGradientBoostingRegressor,
    test_paths: list[Path],
    sample_submission_path: Path | None,
    config: dict[str, Any],
    feature_names: list[str],
    top_context: KaggleTopContext | None = None,
) -> pd.DataFrame:
    predictions_by_well: dict[str, np.ndarray] = {}

    for i, path in enumerate(test_paths, start=1):
        wf = build_well_features(path, config, train=False, top_context=top_context)
        test_features = wf.features.reindex(columns=feature_names).astype("float32")
        residual_pred = model.predict(test_features)
        predictions_by_well[wf.well] = apply_postprocess(wf.flat_prediction, residual_pred, config)
        if i % 50 == 0 or i == len(test_paths):
            print(f"Predicted test wells: {i}/{len(test_paths)}")

    if sample_submission_path is not None:
        sample = pd.read_csv(sample_submission_path)
        rows: list[tuple[str, float]] = []
        missing: set[str] = set()
        for row_id in sample["id"].astype(str):
            well, row_index_text = row_id.rsplit("_", 1)
            row_index = int(row_index_text)
            pred = predictions_by_well.get(well)
            if pred is None:
                missing.add(well)
                continue
            rows.append((row_id, float(pred[row_index])))
        if missing:
            raise FileNotFoundError(f"Missing test predictions for wells: {', '.join(sorted(missing))}")
        return pd.DataFrame(rows, columns=["id", "tvt"])

    rows = []
    for path in test_paths:
        well = well_name(path)
        df = pd.read_csv(path, usecols=["TVT_input"])
        target_mask = pd.to_numeric(df["TVT_input"], errors="coerce").isna().to_numpy()
        for row_index in np.flatnonzero(target_mask):
            rows.append((f"{well}_{row_index}", float(predictions_by_well[well][row_index])))
    return pd.DataFrame(rows, columns=["id", "tvt"])


def save_outputs(
    model: HistGradientBoostingRegressor,
    feature_names: list[str],
    config: dict[str, Any],
    metrics: dict[str, Any],
    config_path: Path,
) -> None:
    output_dir = Path(config["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    with (output_dir / "features.json").open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    if config_path.exists():
        shutil.copyfile(config_path, output_dir / "config.yml")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = str(args.data_dir)
    if args.output_dir is not None:
        config["outputs"]["output_dir"] = str(args.output_dir)
    if args.submission is not None:
        config["outputs"]["submission_path"] = str(args.submission)
    if args.no_cv:
        config["validation"]["enabled"] = False

    seed = int(config.get("seed", 42))
    data_dir = resolve_data_dir(config)
    train_dir = resolve_train_dir(data_dir, config)
    test_dir = resolve_test_dir(data_dir, config)
    sample_submission_path = resolve_sample_submission(data_dir, test_dir, config)

    print(f"Config: {args.config}")
    print(f"Data dir: {data_dir}")
    print(f"Train dir: {train_dir}")
    print(f"Test dir: {test_dir}")
    print(f"Sample submission: {sample_submission_path}")

    train_paths = horizontal_files(train_dir, config["data"].get("max_train_wells"))
    test_paths = horizontal_files(test_dir, config["data"].get("max_test_wells"))
    print(f"Train wells: {len(train_paths)}")
    print(f"Test wells: {len(test_paths)}")

    top_context = None
    if config["features"].get("include_kaggle_top_signals", False):
        print("Building Kaggle top-solution spatial context...")
        top_context = KaggleTopContext(train_paths, config)
        print(
            "  formation wells="
            f"{len(getattr(top_context, 'formation_values', []))}, "
            f"dense ANCC points={len(getattr(top_context, 'dense_ancc', [])):,}"
        )

    X, residual, groups, flat, y_true = build_training_table(train_paths, config, top_context)
    print(f"Training rows: {len(X):,}")
    print(f"Features: {len(X.columns)}")
    print(f"Flat train RMSE: {rmse(flat, y_true):.5f}")

    metrics: dict[str, Any] = {}
    if config["validation"].get("enabled", True):
        metrics["cv"] = run_cv(X, residual, groups, flat, y_true, config, seed)
        best_weight = metrics["cv"].get("best_residual_weight")
        if best_weight is not None and config["postprocess"].get("residual_weight") == "auto":
            config["postprocess"]["residual_weight"] = best_weight
    else:
        metrics["cv"] = {"enabled": False}

    print("Training final model...")
    model = make_model(config, seed)
    model.fit(X, residual)
    train_pred = apply_postprocess(flat, model.predict(X), config)
    metrics["train"] = {
        "rows": int(len(X)),
        "wells": int(len(set(groups))),
        "rmse": rmse(train_pred, y_true),
        "flat_rmse": rmse(flat, y_true),
    }
    print(f"Final train RMSE: {metrics['train']['rmse']:.5f}")

    feature_names = list(X.columns)
    submission = predict_test(model, test_paths, sample_submission_path, config, feature_names, top_context)
    if submission["tvt"].isna().any():
        raise ValueError("Submission contains NaN predictions.")
    submission_path = Path(config["outputs"]["submission_path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"Wrote {submission_path} with {len(submission):,} rows.")

    save_outputs(model, feature_names, config, metrics, args.config)
    print(f"Saved artifacts to {config['outputs']['output_dir']}")


if __name__ == "__main__":
    main()
