from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .heatmap import fill_nan
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class TracebackConfig:
    rows_per_step: int = 32
    patch_radii: tuple[int, ...] = (3, 5, 9, 15)
    min_patch_points: int = 3
    min_prominence_z: float = 0.75
    max_events_per_well: int = 256
    vertical_step_ft: float = 5.0
    top_k_matches: int = 10
    max_match_candidates: int = 2000
    use_train_dictionary: bool = False
    location_weight: float = 1.0
    location_sigma_ft: float = 120.0
    anchor_min_score_quantile: float = 0.8
    anchor_min_top1_gap: float = 0.05
    anchor_max_bridge_delta_ft: float = 160.0
    progress_every: int = 250
    seed: int = 42


def assert_traceback_feature_schema_safe(
    columns: list[str] | tuple[str, ...] | set[str],
) -> None:
    assert_schema_safe_columns(columns, context="TraceBack feature builder")


def _robust_z(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    med = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - med)))
    scale = 1.4826 * mad if mad > 1e-6 else float(np.nanstd(arr[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return np.where(np.isfinite(arr), (arr - med) / scale, 0.0).astype(np.float32)


def _mean_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(values[finite]))


def _safe_gradient(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return np.zeros_like(arr, dtype=np.float32)
    return np.gradient(arr).astype(np.float32)


def compress_well_rows(frame: pd.DataFrame, *, rows_per_step: int) -> pd.DataFrame:
    assert_traceback_feature_schema_safe(["MD", "X", "Y", "Z", "GR", "TVT_input"])
    well = frame.sort_values("row_idx").reset_index(drop=True).copy()
    gr_raw = pd.to_numeric(well["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, finite_mask = fill_nan(gr_raw)
    well["GR_filled_row"] = gr_filled
    well["GR_finite_row"] = finite_mask

    rows: list[dict[str, Any]] = []
    for step, start in enumerate(range(0, len(well), rows_per_step)):
        chunk = well.iloc[start : start + rows_per_step]
        tvt_input = pd.to_numeric(chunk["TVT_input"], errors="coerce")
        known = tvt_input.notna()
        target_series = chunk.get("TVT", pd.Series(np.nan, index=chunk.index))
        rows.append(
            {
                "well_id": str(chunk["well_id"].iloc[0]),
                "step": int(step),
                "row_start": int(chunk["row_idx"].min()),
                "row_end": int(chunk["row_idx"].max()),
                "MD": _mean_or_nan(
                    pd.to_numeric(chunk["MD"], errors="coerce").to_numpy(dtype=np.float64)
                ),
                "X": _mean_or_nan(
                    pd.to_numeric(chunk["X"], errors="coerce").to_numpy(dtype=np.float64)
                ),
                "Y": _mean_or_nan(
                    pd.to_numeric(chunk["Y"], errors="coerce").to_numpy(dtype=np.float64)
                ),
                "Z": _mean_or_nan(
                    pd.to_numeric(chunk["Z"], errors="coerce").to_numpy(dtype=np.float64)
                ),
                "GR_filled": _mean_or_nan(chunk["GR_filled_row"].to_numpy(dtype=np.float64)),
                "GR_finite_frac": float(np.mean(chunk["GR_finite_row"].to_numpy(dtype=bool))),
                "TVT_input": _mean_or_nan(tvt_input.to_numpy(dtype=np.float64)),
                "known_mask": bool(known.any()),
                "true_TVT": _mean_or_nan(
                    pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=np.float64)
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["GR_z"] = _robust_z(out["GR_filled"].to_numpy(dtype=np.float64))
    out["dGR_z"] = _safe_gradient(out["GR_z"].to_numpy(dtype=np.float32))
    return out


def _patch(values: np.ndarray, center: int, radius: int) -> np.ndarray:
    out = np.full(radius * 2 + 1, np.nan, dtype=np.float32)
    lo = max(0, center - radius)
    hi = min(len(values), center + radius + 1)
    dst_lo = lo - (center - radius)
    out[dst_lo : dst_lo + (hi - lo)] = values[lo:hi]
    return out


def extract_traceback_events(comp: pd.DataFrame, cfg: TracebackConfig) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    if "GR_z" in comp.columns:
        gr = comp["GR_z"].to_numpy(dtype=np.float32)
    else:
        gr = _robust_z(comp["GR_filled"].to_numpy(dtype=np.float64))
    dgr = _safe_gradient(gr)
    rows: list[dict[str, Any]] = []
    for i in range(1, len(comp) - 1):
        left, cur, right = float(gr[i - 1]), float(gr[i]), float(gr[i + 1])
        event_type = ""
        prominence = 0.0
        if cur > left and cur > right:
            event_type = "peak"
            prominence = cur - max(left, right)
        elif cur < left and cur < right:
            event_type = "trough"
            prominence = min(left, right) - cur
        if not event_type or prominence < cfg.min_prominence_z:
            continue
        for radius in cfg.patch_radii:
            patch_gr = _patch(gr, i, radius)
            if np.isfinite(patch_gr).sum() < cfg.min_patch_points:
                continue
            tvt_input = comp["TVT_input"].iloc[i] if "TVT_input" in comp.columns else np.nan
            true_tvt = comp["true_TVT"].iloc[i] if "true_TVT" in comp.columns else np.nan
            rows.append(
                {
                    "well_id": str(comp["well_id"].iloc[i]),
                    "step": int(comp["step"].iloc[i]),
                    "row_start": int(comp["row_start"].iloc[i]),
                    "row_end": int(comp["row_end"].iloc[i]),
                    "MD": float(comp["MD"].iloc[i]),
                    "X": float(comp["X"].iloc[i]),
                    "Y": float(comp["Y"].iloc[i]),
                    "Z": float(comp["Z"].iloc[i]),
                    "known_mask": bool(comp["known_mask"].iloc[i]),
                    "TVT_input": float(tvt_input) if np.isfinite(tvt_input) else np.nan,
                    "true_TVT": float(true_tvt) if np.isfinite(true_tvt) else np.nan,
                    "event_type": event_type,
                    "prominence": float(prominence),
                    "finite_frac": float(comp["GR_finite_frac"].iloc[i]),
                    "patch_radius": int(radius),
                    "patch_gr": patch_gr.astype(np.float32),
                    "patch_dgr": _patch(dgr, i, radius).astype(np.float32),
                }
            )
    events = pd.DataFrame(rows)
    if len(events) > cfg.max_events_per_well:
        events = events.sort_values("prominence", ascending=False).head(cfg.max_events_per_well)
    return events.reset_index(drop=True)
