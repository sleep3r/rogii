from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

try:
    import pywt
except Exception:  # pragma: no cover - optional in some runtimes.
    pywt = None


NOTEBOOK_V4_FEATURE_NAMES: tuple[str, ...] = (
    "nbv4_gr_dwt_approx",
    "nbv4_gr_dwt_detail_energy",
    "nbv4_gr_dwt_residual",
    "nbv4_anchor_slope_k10",
    "nbv4_anchor_slope_k25",
    "nbv4_anchor_slope_k50",
    "nbv4_anchor_slope_k100",
    "nbv4_slope_accel_10_50",
    "nbv4_slope_accel_25_100",
    "nbv4_tvt_extrap_k10",
    "nbv4_tvt_extrap_k25",
    "nbv4_tvt_extrap_k50",
    "nbv4_tvt_extrap_k10_minus_last",
    "nbv4_tvt_extrap_k25_minus_last",
    "nbv4_tvt_extrap_k50_minus_last",
    "nbv4_tvt_extrap_k10_minus_flat",
    "nbv4_tvt_extrap_k25_minus_flat",
    "nbv4_tvt_extrap_k50_minus_flat",
    "nbv4_path_vs_ncc",
    "nbv4_path_vs_pf",
    "nbv4_path_vs_beam",
    "nbv4_path_vs_dwt",
    "nbv4_path_vs_structural",
    "nbv4_ncc_vs_pf",
    "nbv4_ncc_vs_beam",
    "nbv4_ncc_vs_dwt",
    "nbv4_beam_vs_pf",
    "nbv4_beam_vs_dwt",
    "nbv4_pf_vs_dwt",
    "nbv4_pf_vs_dense",
    "nbv4_spatial_vs_dense",
    "nbv4_estimator_tvt_mean",
    "nbv4_estimator_tvt_std",
    "nbv4_estimator_tvt_range",
    "nbv4_estimator_tvt_min",
    "nbv4_estimator_tvt_max",
    "nbv4_estimator_drift_mean",
    "nbv4_estimator_drift_std",
    "nbv4_estimator_drift_range",
    "nbv4_estimator_count",
)


def _nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def _as_array(value: object, n: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if len(arr) != n:
        return None
    return arr.astype(float, copy=False)


def _candidate(
    features: Mapping[str, object],
    names: tuple[str, ...],
    n: int,
) -> np.ndarray | None:
    for name in names:
        arr = _as_array(features.get(name), n)
        if arr is not None and np.isfinite(arr).any():
            return arr
    return None


def _tail_slope(md: np.ndarray, tvt_input: np.ndarray, known_idx: np.ndarray, window: int) -> float:
    if len(known_idx) < 2:
        return 0.0
    selected = known_idx[-min(int(window), len(known_idx)) :]
    if len(selected) < 2:
        return 0.0
    x = md[selected].astype(float)
    y = tvt_input[selected].astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    cx = x - float(np.mean(x))
    denom = float(np.dot(cx, cx))
    if denom <= 1e-12:
        return 0.0
    slope = float(np.dot(cx, y - float(np.mean(y))) / denom)
    return slope if np.isfinite(slope) else 0.0


def _wavelet_gr_features(gr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(gr)
    fallback = float(np.nanmean(gr)) if np.isfinite(gr).any() else 0.0
    filled = (
        pd.Series(gr, dtype=float)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .fillna(fallback)
        .to_numpy(dtype=float)
    )
    if pywt is None or n < 8:
        approx = filled.copy()
        detail_energy = np.zeros(n, dtype=float)
        return approx, detail_energy, filled - approx

    try:
        wavelet = pywt.Wavelet("db4")
        max_level = pywt.dwt_max_level(n, wavelet.dec_len)
        level = max(1, min(5, int(max_level)))
        coeffs = pywt.wavedec(filled, wavelet, mode="periodization", level=level)

        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        approx = pywt.waverec(approx_coeffs, wavelet, mode="periodization")[:n]

        detail_coeffs = [np.zeros_like(c) for c in coeffs]
        if len(coeffs) > 1:
            detail_index = min(3, len(coeffs) - 1)
            detail_coeffs[detail_index] = coeffs[detail_index]
            detail = pywt.waverec(detail_coeffs, wavelet, mode="periodization")[:n]
            detail_energy = (
                pd.Series(detail**2, dtype=float)
                .rolling(16, center=True, min_periods=1)
                .mean()
                .pow(0.5)
                .to_numpy(dtype=float)
            )
        else:
            detail_energy = np.zeros(n, dtype=float)
    except Exception:
        approx = filled.copy()
        detail_energy = np.zeros(n, dtype=float)

    return approx.astype(float), detail_energy.astype(float), (filled - approx).astype(float)


def _pairwise(a: np.ndarray | None, b: np.ndarray | None, n: int) -> np.ndarray:
    if a is None or b is None:
        return _nan_array(n)
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


def _candidate_stats(
    candidates: list[np.ndarray],
    last_tvt: float,
    n: int,
) -> dict[str, np.ndarray]:
    if not candidates:
        return {
            "mean": _nan_array(n),
            "std": _nan_array(n),
            "range": _nan_array(n),
            "min": _nan_array(n),
            "max": _nan_array(n),
            "drift_mean": _nan_array(n),
            "drift_std": _nan_array(n),
            "drift_range": _nan_array(n),
            "count": np.zeros(n, dtype=float),
        }

    matrix = np.vstack([np.asarray(c, dtype=float) for c in candidates]).T
    valid = np.isfinite(matrix)
    count = valid.sum(axis=1).astype(float)
    safe = np.where(valid, matrix, 0.0)
    mean = np.divide(
        safe.sum(axis=1),
        count,
        out=_nan_array(n),
        where=count > 0,
    )
    second = np.divide(
        (safe**2).sum(axis=1),
        count,
        out=_nan_array(n),
        where=count > 0,
    )
    std = np.sqrt(np.maximum(second - mean**2, 0.0))
    min_v = np.where(valid, matrix, np.inf).min(axis=1)
    max_v = np.where(valid, matrix, -np.inf).max(axis=1)
    min_v = np.where(count > 0, min_v, np.nan)
    max_v = np.where(count > 0, max_v, np.nan)
    range_v = max_v - min_v
    return {
        "mean": mean,
        "std": std,
        "range": range_v,
        "min": min_v,
        "max": max_v,
        "drift_mean": mean - last_tvt,
        "drift_std": std,
        "drift_range": range_v,
        "count": count,
    }


def build_notebook_v4_features(
    *,
    md: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    flat_pred: np.ndarray,
    features: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Build small test-safe features inspired by Mitch Gansemer's v4 notebook.

    The implementation intentionally uses only current-well public inputs and
    already-computed test-safe expert predictions. It does not depend on the
    notebook's external model bundle or train-only formation columns.
    """
    n = len(md)
    out = {name: _nan_array(n) for name in NOTEBOOK_V4_FEATURE_NAMES}
    tvt_input = np.asarray(tvt_input, dtype=float)
    flat_pred = np.asarray(flat_pred, dtype=float)
    known_idx = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_idx) == 0:
        return out

    last_idx = int(known_idx[-1])
    last_tvt = float(tvt_input[last_idx])
    dmd = np.asarray(md, dtype=float) - float(md[last_idx])

    approx, detail_energy, residual = _wavelet_gr_features(np.asarray(gr, dtype=float))
    out["nbv4_gr_dwt_approx"] = approx
    out["nbv4_gr_dwt_detail_energy"] = detail_energy
    out["nbv4_gr_dwt_residual"] = residual

    slopes = {
        10: _tail_slope(np.asarray(md, dtype=float), tvt_input, known_idx, 10),
        25: _tail_slope(np.asarray(md, dtype=float), tvt_input, known_idx, 25),
        50: _tail_slope(np.asarray(md, dtype=float), tvt_input, known_idx, 50),
        100: _tail_slope(np.asarray(md, dtype=float), tvt_input, known_idx, 100),
    }
    for window, slope in slopes.items():
        out[f"nbv4_anchor_slope_k{window}"] = np.full(n, slope, dtype=float)
    out["nbv4_slope_accel_10_50"] = np.full(n, slopes[10] - slopes[50], dtype=float)
    out["nbv4_slope_accel_25_100"] = np.full(n, slopes[25] - slopes[100], dtype=float)
    for window in (10, 25, 50):
        extrap = last_tvt + slopes[window] * dmd
        out[f"nbv4_tvt_extrap_k{window}"] = extrap
        out[f"nbv4_tvt_extrap_k{window}_minus_last"] = extrap - last_tvt
        out[f"nbv4_tvt_extrap_k{window}_minus_flat"] = extrap - flat_pred

    path = _candidate(
        features,
        ("kg_path_stage12_tvt", "kg_path_cem_raw_tvt", "kg_path_geo_consensus_tvt"),
        n,
    )
    ncc = _candidate(features, ("kg_ncc_mean_tvt", "GR_ncc_tvt"), n)
    beam = _candidate(features, ("kg_beam_mean_tvt",), n)
    pf = _candidate(features, ("pf_ancc", "kg_pf_ancc_tvt"), n)
    dwt = _candidate(features, ("kg_dwt_tvt",), n)
    structural = _candidate(features, ("tvt_structural",), n)
    spatial = _candidate(features, ("kg_form_ancc_tvt", "tvt_form_full"), n)
    dense = _candidate(features, ("kg_dense_ancc_tvt", "dense_tvt_pred"), n)

    out["nbv4_path_vs_ncc"] = _pairwise(path, ncc, n)
    out["nbv4_path_vs_pf"] = _pairwise(path, pf, n)
    out["nbv4_path_vs_beam"] = _pairwise(path, beam, n)
    out["nbv4_path_vs_dwt"] = _pairwise(path, dwt, n)
    out["nbv4_path_vs_structural"] = _pairwise(path, structural, n)
    out["nbv4_ncc_vs_pf"] = _pairwise(ncc, pf, n)
    out["nbv4_ncc_vs_beam"] = _pairwise(ncc, beam, n)
    out["nbv4_ncc_vs_dwt"] = _pairwise(ncc, dwt, n)
    out["nbv4_beam_vs_pf"] = _pairwise(beam, pf, n)
    out["nbv4_beam_vs_dwt"] = _pairwise(beam, dwt, n)
    out["nbv4_pf_vs_dwt"] = _pairwise(pf, dwt, n)
    out["nbv4_pf_vs_dense"] = _pairwise(pf, dense, n)
    out["nbv4_spatial_vs_dense"] = _pairwise(spatial, dense, n)

    stats = _candidate_stats(
        [c for c in (path, ncc, beam, pf, dwt, structural, spatial, dense) if c is not None],
        last_tvt,
        n,
    )
    out["nbv4_estimator_tvt_mean"] = stats["mean"]
    out["nbv4_estimator_tvt_std"] = stats["std"]
    out["nbv4_estimator_tvt_range"] = stats["range"]
    out["nbv4_estimator_tvt_min"] = stats["min"]
    out["nbv4_estimator_tvt_max"] = stats["max"]
    out["nbv4_estimator_drift_mean"] = stats["drift_mean"]
    out["nbv4_estimator_drift_std"] = stats["drift_std"]
    out["nbv4_estimator_drift_range"] = stats["drift_range"]
    out["nbv4_estimator_count"] = stats["count"]
    return out


__all__ = ["NOTEBOOK_V4_FEATURE_NAMES", "build_notebook_v4_features"]
