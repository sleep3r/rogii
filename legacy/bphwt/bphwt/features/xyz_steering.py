from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class XYZSteeringConfig:
    horizons_ft: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0, 500.0)


def build_xyz_steering_features(
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    known_mask: np.ndarray,
    cfg: XYZSteeringConfig | None = None,
    mode: str = "full",
) -> pd.DataFrame:
    """Build trajectory-action features.

    ``mode="past"`` uses only current and previous trajectory. ``mode="full"``
    also uses future trajectory response, which is the post-drilling signal from
    the Kaggle discussion screenshot.
    """
    c = cfg or XYZSteeringConfig()
    mode = str(mode)
    if mode not in {"past", "full"}:
        raise ValueError("mode must be 'past' or 'full'")

    md_arr, x_arr, y_arr, z_arr = _clean_inputs(md, x, y, z)
    known = np.asarray(known_mask, dtype=bool)
    if known.shape != md_arr.shape:
        known = np.zeros_like(md_arr, dtype=bool)

    dx = _gradient(x_arr, md_arr)
    dy = _gradient(y_arr, md_arr)
    dz = _gradient(z_arr, md_arr)
    d2x = _gradient(dx, md_arr)
    d2y = _gradient(dy, md_arr)
    d2z = _gradient(dz, md_arr)
    xy_speed = np.sqrt(dx * dx + dy * dy)

    data: dict[str, np.ndarray] = {
        "md_rel": _rel01(md_arr),
        "x_rel": _rel01(x_arr),
        "y_rel": _rel01(y_arr),
        "z_rel": _rel01(z_arr),
        "dx_dmd": dx,
        "dy_dmd": dy,
        "dz_dmd": dz,
        "d2x_dmd2": d2x,
        "d2y_dmd2": d2y,
        "d2z_dmd2": d2z,
        "xy_speed": xy_speed,
        "curvature_xyz": np.sqrt(d2x * d2x + d2y * d2y + d2z * d2z),
    }

    anchor = _anchor_context(md_arr, known)
    data.update(anchor)

    for horizon in c.horizons_ft:
        h = float(horizon)
        suffix = _horizon_suffix(h)
        past_x = _interp_at(md_arr, x_arr, md_arr - h)
        past_y = _interp_at(md_arr, y_arr, md_arr - h)
        past_z = _interp_at(md_arr, z_arr, md_arr - h)

        past_dx = x_arr - past_x
        past_dy = y_arr - past_y
        past_dz = z_arr - past_z
        data[f"past_x_delta_{suffix}"] = past_dx
        data[f"past_y_delta_{suffix}"] = past_dy
        data[f"past_z_delta_{suffix}"] = past_dz
        data[f"past_z_slope_{suffix}"] = past_dz / max(h, 1.0)
        data[f"past_xy_step_{suffix}"] = np.sqrt(past_dx * past_dx + past_dy * past_dy)
        data[f"past_z_deviation_{suffix}"] = z_arr - (past_z + dz * h)

        if mode == "full":
            future_x = _interp_at(md_arr, x_arr, md_arr + h)
            future_y = _interp_at(md_arr, y_arr, md_arr + h)
            future_z = _interp_at(md_arr, z_arr, md_arr + h)
            future_dx = future_x - x_arr
            future_dy = future_y - y_arr
            future_dz = future_z - z_arr
            cross = past_dx * future_dy - past_dy * future_dx
            dot = past_dx * future_dx + past_dy * future_dy
            data[f"future_x_delta_{suffix}"] = future_dx
            data[f"future_y_delta_{suffix}"] = future_dy
            data[f"future_z_delta_{suffix}"] = future_dz
            data[f"future_z_slope_{suffix}"] = future_dz / max(h, 1.0)
            data[f"future_xy_step_{suffix}"] = np.sqrt(future_dx * future_dx + future_dy * future_dy)
            data[f"future_z_deviation_{suffix}"] = future_z - (z_arr + dz * h)
            data[f"future_turn_cross_{suffix}"] = cross / max(h * h, 1.0)
            data[f"future_turn_dot_{suffix}"] = dot / max(h * h, 1.0)
            data[f"future_available_{suffix}"] = (md_arr + h <= md_arr[-1]).astype(np.float64)

    frame = pd.DataFrame(data)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.astype(np.float32)


def _clean_inputs(
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.asarray(v, dtype=np.float64) for v in (md, x, y, z)]
    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays):
        raise ValueError("md, x, y, and z must have the same shape")
    if arrays[0].ndim != 1:
        raise ValueError("trajectory inputs must be one-dimensional")
    order = np.argsort(arrays[0])
    arrays = [arr[order] for arr in arrays]
    md_arr = arrays[0]
    if md_arr.size == 0:
        raise ValueError("trajectory inputs must be non-empty")
    for i in range(1, 4):
        arrays[i] = _fill_finite(arrays[i])
    if not np.isfinite(md_arr).all() or np.any(np.diff(md_arr) <= 0.0):
        md_arr = np.arange(md_arr.size, dtype=np.float64)
    return md_arr, arrays[1], arrays[2], arrays[3]


def _fill_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros_like(arr)
    axis = np.arange(arr.size, dtype=np.float64)
    arr[~valid] = np.interp(axis[~valid], axis[valid], arr[valid])
    return arr


def _gradient(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values)
    return np.gradient(values, md, edge_order=1).astype(np.float64)


def _interp_at(md: np.ndarray, values: np.ndarray, query_md: np.ndarray) -> np.ndarray:
    return np.interp(query_md, md, values, left=values[0], right=values[-1]).astype(np.float64)


def _rel01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    span = float(np.nanmax(arr) - np.nanmin(arr)) if arr.size else 0.0
    if not np.isfinite(span) or span < 1.0e-9:
        return np.zeros_like(arr)
    return (arr - float(np.nanmin(arr))) / span


def _anchor_context(md: np.ndarray, known: np.ndarray) -> dict[str, np.ndarray]:
    n = md.size
    if not known.any():
        inf_dist = np.full(n, 1.0e6, dtype=np.float64)
        return {
            "dist_prev_anchor": inf_dist,
            "dist_next_anchor": inf_dist,
            "anchor_gap": inf_dist,
            "anchor_frac": np.zeros(n, dtype=np.float64),
            "known_mask": known.astype(np.float64),
        }

    idx = np.arange(n)
    known_idx = np.flatnonzero(known)
    prev_pos = np.searchsorted(known_idx, idx, side="right") - 1
    next_pos = np.searchsorted(known_idx, idx, side="left")
    prev_idx = known_idx[np.clip(prev_pos, 0, known_idx.size - 1)]
    next_idx = known_idx[np.clip(next_pos, 0, known_idx.size - 1)]
    has_prev = prev_pos >= 0
    has_next = next_pos < known_idx.size
    prev_md = md[prev_idx]
    next_md = md[next_idx]
    dist_prev = np.where(has_prev, md - prev_md, 1.0e6)
    dist_next = np.where(has_next, next_md - md, 1.0e6)
    gap = np.where(has_prev & has_next, np.maximum(next_md - prev_md, 1.0), dist_prev + dist_next)
    frac = np.where(has_prev & has_next, np.clip((md - prev_md) / gap, 0.0, 1.0), 0.0)
    return {
        "dist_prev_anchor": dist_prev,
        "dist_next_anchor": dist_next,
        "anchor_gap": gap,
        "anchor_frac": frac,
        "known_mask": known.astype(np.float64),
    }


def _horizon_suffix(horizon: float) -> str:
    if abs(horizon - round(horizon)) < 1.0e-9:
        return str(int(round(horizon)))
    return str(horizon).replace(".", "p")


__all__ = ["XYZSteeringConfig", "build_xyz_steering_features"]
