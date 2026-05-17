from __future__ import annotations

import numpy as np
import pandas as pd


def as_float_array(
    series: pd.Series | np.ndarray, default: float = np.nan
) -> np.ndarray:
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


def tail_stat(
    values: np.ndarray, known: np.ndarray, window: int, statistic: str
) -> float:
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


def flat_tvt_prediction(
    md: np.ndarray, tvt_input: np.ndarray, tail_window: int
) -> np.ndarray:
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
    return (
        series.interpolate(limit_direction="both")
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


def nearest_index(sorted_values: np.ndarray, value: float) -> int:
    pos = int(np.searchsorted(sorted_values, value, side="left"))
    if pos >= len(sorted_values):
        return len(sorted_values) - 1
    if pos > 0 and abs(sorted_values[pos - 1] - value) <= abs(
        sorted_values[pos] - value
    ):
        return pos - 1
    return pos


def smooth_for_alignment(
    values: np.ndarray, radius: int, fallback: float
) -> np.ndarray:
    filled = fill_numeric(values, fallback)
    if radius <= 0:
        return filled
    return (
        pd.Series(filled)
        .rolling(radius * 2 + 1, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
