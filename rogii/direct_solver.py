from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from .path_solver_extras import (
    EnergyContext,
    cem_path_search,
    collect_well_signatures,
    matched_triples,
    stage12_path,
)
from .cross_well_prior import (
    TrainWellPath,
    collect_train_paths,
    compute_signature,
    cross_well_typewell_path,
)

FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


@dataclass(frozen=True)
class SolverVariant:
    """Energy weight container used by ``score_candidate_path``.

    Kept as a dataclass so the scorer signature stays stable, but the old
    safe/bold gated families that previously lived here have been removed
    along with the rest of the anchor-pinned safety net.
    """

    name: str
    gr_weight: float
    geo_weight: float
    anchor_weight: float
    slope_weight: float
    endpoint_weight: float
    max_gate: float
    min_improvement: float


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:05.2f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{sec:02d}"


def log_progress(
    *,
    mode: str,
    current: int,
    total: int,
    rows_done: int,
    total_rows: int,
    well: str,
    started_at: float,
) -> None:
    elapsed = perf_counter() - started_at
    eta = elapsed * (total - current) / max(current, 1)
    rows_per_sec = rows_done / max(elapsed, 1e-9)
    print(
        "Direct solver progress | "
        f"mode={mode} current={current} total={total} well={well} "
        f"rows={rows_done}/{total_rows} elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)} rows_per_sec={rows_per_sec:.1f}",
        flush=True,
    )


def should_log_progress(current: int, total: int, progress_interval: int) -> bool:
    return current == 1 or current % progress_interval == 0 or current == total


def robust_sigma(values: np.ndarray, default: float = 1.0) -> float:
    finite = _finite(np.asarray(values, dtype=float))
    if len(finite) < 3:
        return float(default)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = float(np.nanstd(finite))
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = float(default)
    return max(sigma, 1e-6)


def nanmedian(values: np.ndarray, default: float = 0.0) -> float:
    finite = _finite(np.asarray(values, dtype=float))
    if len(finite) == 0:
        return float(default)
    return float(np.median(finite))


def smooth(values: np.ndarray, window: int = 21) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    if window <= 1 or len(values) < 3:
        return pd.Series(values, dtype=float).interpolate(limit_direction="both").bfill().ffill().to_numpy(float)
    window = int(max(1, min(window, len(values))))
    if window % 2 == 0:
        window -= 1
    return (
        pd.Series(values, dtype=float)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(float)
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if int(mask.sum()) < x.shape[1] + 2:
        return None
    xs = x[mask]
    ys = y[mask]
    scale = np.nanstd(xs, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    xs_scaled = xs / scale
    xtx = xs_scaled.T @ xs_scaled
    penalty = np.eye(xtx.shape[0]) * ridge
    penalty[0, 0] = 0.0
    try:
        beta_scaled = np.linalg.solve(xtx + penalty, xs_scaled.T @ ys)
    except np.linalg.LinAlgError:
        beta_scaled = np.linalg.lstsq(xtx + penalty, xs_scaled.T @ ys, rcond=None)[0]
    beta = beta_scaled / scale
    return beta.astype(float)


def robust_line_slope(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float(default)
    x = x[mask]
    y = y[mask]
    dx = x - np.median(x)
    dy = y - np.median(y)
    denom = float(np.dot(dx, dx))
    if denom <= 1e-9:
        return float(default)
    slope = float(np.dot(dx, dy) / denom)
    if not np.isfinite(slope):
        return float(default)
    return slope


def clip_path_steps(path: np.ndarray, md: np.ndarray, hidden_indices: np.ndarray, last_idx: int, last_tvt: float, tail_slope_abs: float) -> np.ndarray:
    path = np.asarray(path, dtype=float).copy()
    if len(hidden_indices) == 0:
        return path
    max_slope = max(0.015, min(0.35, 4.0 * float(tail_slope_abs)))
    prev_idx = int(last_idx)
    prev_value = float(last_tvt)
    for idx in hidden_indices:
        step_md = abs(float(md[idx] - md[prev_idx]))
        max_step = max(0.75, max_slope * max(step_md, 1.0))
        current = float(path[idx])
        if not np.isfinite(current):
            current = prev_value
        current = float(np.clip(current, prev_value - max_step, prev_value + max_step))
        path[idx] = current
        prev_idx = int(idx)
        prev_value = current
    return path


def read_anchor_submission(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    if "id" not in frame.columns or "tvt" not in frame.columns:
        raise ValueError(f"Anchor submission must have id,tvt columns: {path}")
    return {str(row_id): float(tvt) for row_id, tvt in zip(frame["id"], frame["tvt"])}


def sample_rows(sample_submission: pd.DataFrame) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {}
    for row_id in sample_submission["id"].astype(str):
        well, idx_text = row_id.rsplit("_", 1)
        rows.setdefault(well, []).append(int(idx_text))
    return {well: sorted(indices) for well, indices in rows.items()}


def horizontal_path(data_dir: Path, well: str, train: bool) -> Path:
    sub = "train" if train else "test"
    return data_dir / sub / f"{well}__horizontal_well.csv"


def typewell_path_for(horizontal: Path) -> Path | None:
    candidate = horizontal.with_name(horizontal.name.replace("__horizontal_well.csv", "__typewell.csv"))
    return candidate if candidate.exists() else None


def read_typewell(horizontal: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = typewell_path_for(horizontal)
    if path is None:
        return None
    try:
        frame = pd.read_csv(path, usecols=["TVT", "GR"])
    except Exception:
        return None
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)
    mask = np.isfinite(tvt) & np.isfinite(gr)
    if int(mask.sum()) < 16:
        return None
    tvt = tvt[mask]
    gr = gr[mask]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    unique_tvt, unique_idx = np.unique(tvt, return_index=True)
    return unique_tvt.astype(float), gr[unique_idx].astype(float)


def interpolated_typewell_gr(typewell: tuple[np.ndarray, np.ndarray] | None, tvt_path: np.ndarray) -> np.ndarray | None:
    if typewell is None:
        return None
    tvt_grid, gr_grid = typewell
    if len(tvt_grid) < 2:
        return None
    return np.interp(tvt_path, tvt_grid, gr_grid, left=gr_grid[0], right=gr_grid[-1]).astype(float)


def known_tail_mask(tvt_input: np.ndarray, last_idx: int, tail_rows: int) -> np.ndarray:
    known = np.isfinite(tvt_input)
    start = max(0, int(last_idx) - int(tail_rows) + 1)
    mask = np.zeros(len(tvt_input), dtype=bool)
    mask[start : int(last_idx) + 1] = True
    return mask & known


def fit_linear_candidate(df: pd.DataFrame, md: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray, tvt_input: np.ndarray, hidden_indices: np.ndarray, last_idx: int, last_tvt: float, tail_rows: int) -> np.ndarray:
    n = len(df)
    dmd = md - float(md[last_idx])
    dz = z - float(z[last_idx])
    dxy = np.sqrt((x - float(x[last_idx])) ** 2 + (y - float(y[last_idx])) ** 2)
    span = max(float(np.nanmax(np.abs(dmd))) if np.isfinite(dmd).any() else 1.0, 1.0)
    xmat = np.column_stack([
        np.ones(n),
        dmd / span,
        dz / max(robust_sigma(dz, 1.0), 1.0),
        dxy / max(robust_sigma(dxy, 1.0), 1.0),
        (dmd / span) ** 2,
    ])
    mask = known_tail_mask(tvt_input, last_idx, tail_rows)
    beta = ridge_fit(xmat[mask], tvt_input[mask], ridge=1e-2)
    if beta is None:
        slope = robust_line_slope(md[mask], tvt_input[mask], default=0.0)
        path = last_tvt + slope * dmd
    else:
        path = xmat @ beta
        if np.isfinite(path[last_idx]):
            path = path + (last_tvt - float(path[last_idx]))
        else:
            path = last_tvt + robust_line_slope(md[mask], tvt_input[mask], default=0.0) * dmd
    tail_slope_abs = abs(robust_line_slope(md[mask], tvt_input[mask], default=0.02))
    return clip_path_steps(path, md, hidden_indices, last_idx, last_tvt, tail_slope_abs)


def fit_geo_candidate(df: pd.DataFrame, md: np.ndarray, z: np.ndarray, tvt_input: np.ndarray, hidden_indices: np.ndarray, last_idx: int, last_tvt: float, tail_rows: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Per-formation linear TVT fit.

    Returns ``(best_path, consensus_path, diagnostics)``. ``best_path`` is the
    single best surface (by tail RMSE) preserved for backward compatibility.
    ``consensus_path`` is the inverse-RMSE-weighted median across all
    formations that produced a finite fit. Tier-1 plan item 5: do not lock
    onto one formation when several agree, and do not ignore disagreement.
    """
    n = len(df)
    mask = known_tail_mask(tvt_input, last_idx, tail_rows)
    best_path: np.ndarray | None = None
    best_rmse = np.inf
    best_surface = "none"
    diagnostics: dict[str, float] = {}
    dmd = md - float(md[last_idx])
    per_surface_paths: list[np.ndarray] = []
    per_surface_rmse: list[float] = []
    per_surface_names: list[str] = []
    for surface in FORMATIONS:
        if surface not in df.columns:
            continue
        surf = pd.to_numeric(df[surface], errors="coerce").to_numpy(float)
        rel = z - surf
        xmat = np.column_stack([
            np.ones(n),
            rel,
            dmd / max(float(np.nanmax(np.abs(dmd))) if np.isfinite(dmd).any() else 1.0, 1.0),
        ])
        valid = mask & np.isfinite(rel)
        if int(valid.sum()) < 16:
            continue
        beta = ridge_fit(xmat[valid], tvt_input[valid], ridge=1e-2)
        if beta is None:
            continue
        pred = xmat @ beta
        rmse = float(np.sqrt(np.nanmean((pred[valid] - tvt_input[valid]) ** 2)))
        diagnostics[f"geo_{surface}_tail_rmse"] = rmse
        if np.isfinite(pred).any():
            if np.isfinite(pred[last_idx]):
                pred = pred + (last_tvt - float(pred[last_idx]))
            per_surface_paths.append(pred)
            per_surface_rmse.append(rmse)
            per_surface_names.append(surface)
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_surface = surface
            best_path = pred
    fallback_path = fit_linear_candidate(df, md, z, np.zeros(n), np.zeros(n), tvt_input, hidden_indices, last_idx, last_tvt, tail_rows)
    if best_path is None:
        best_path = fallback_path.copy()
        best_rmse = float("nan")
    if np.isfinite(best_path[last_idx]):
        best_path = best_path + (last_tvt - float(best_path[last_idx]))
    tail_slope_abs = abs(robust_line_slope(md[mask], tvt_input[mask], default=0.02))
    best_path = clip_path_steps(best_path, md, hidden_indices, last_idx, last_tvt, tail_slope_abs)
    if per_surface_paths:
        stack = np.vstack(per_surface_paths)
        rmse_arr = np.asarray(per_surface_rmse, dtype=float)
        finite_rmse = np.where(np.isfinite(rmse_arr) & (rmse_arr > 1e-6), rmse_arr, np.inf)
        weights = 1.0 / (finite_rmse + 1e-3)
        if not np.isfinite(weights).any() or float(weights.sum()) <= 0.0:
            consensus_path = np.nanmedian(stack, axis=0)
        else:
            normalized = weights / float(weights.sum())
            # Weighted median: replicate each surface roughly proportional
            # to its inverse-RMSE weight, then take a plain median. Using a
            # 100-step replication grid keeps this deterministic and avoids
            # an external dependency.
            scaled = np.round(normalized * 100).astype(int)
            scaled = np.maximum(scaled, 1)
            replicated = np.vstack([stack[i] for i, count in enumerate(scaled) for _ in range(int(count))])
            consensus_path = np.nanmedian(replicated, axis=0)
        if np.isfinite(consensus_path[last_idx]):
            consensus_path = consensus_path + (last_tvt - float(consensus_path[last_idx]))
        consensus_path = clip_path_steps(consensus_path, md, hidden_indices, last_idx, last_tvt, tail_slope_abs)
        diagnostics["geo_consensus_surfaces"] = float(len(per_surface_paths))
        diagnostics["geo_consensus_rmse_min"] = float(np.nanmin(rmse_arr)) if rmse_arr.size else float("nan")
        diagnostics["geo_consensus_rmse_max"] = float(np.nanmax(rmse_arr)) if rmse_arr.size else float("nan")
        diagnostics["geo_consensus_rmse_spread"] = (
            float(np.nanmax(rmse_arr) - np.nanmin(rmse_arr)) if rmse_arr.size else float("nan")
        )
    else:
        consensus_path = best_path.copy()
        diagnostics["geo_consensus_surfaces"] = 0.0
    diagnostics["geo_best_tail_rmse"] = float(best_rmse) if np.isfinite(best_rmse) else np.nan
    diagnostics["geo_best_surface_id"] = float(FORMATIONS.index(best_surface)) if best_surface in FORMATIONS else -1.0
    return best_path, consensus_path, diagnostics


def fit_gr_calibration(typewell: tuple[np.ndarray, np.ndarray] | None, tvt_input: np.ndarray, gr: np.ndarray, tail_mask: np.ndarray) -> tuple[float, float, float]:
    tw_gr = interpolated_typewell_gr(typewell, tvt_input)
    if tw_gr is None:
        return 1.0, 0.0, np.inf
    mask = tail_mask & np.isfinite(tw_gr) & np.isfinite(gr)
    if int(mask.sum()) < 12:
        return 1.0, 0.0, np.inf
    x = np.column_stack([tw_gr[mask], np.ones(int(mask.sum()))])
    y = gr[mask]
    beta = ridge_fit(x, y, ridge=1e-3)
    if beta is None:
        return 1.0, 0.0, np.inf
    a = float(beta[0])
    b = float(beta[1])
    pred = a * tw_gr[mask] + b
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    if not np.isfinite(a):
        a = 1.0
    if not np.isfinite(b):
        b = 0.0
    return a, b, rmse


def path_slope(path: np.ndarray, md: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) < 2:
        return 0.0
    return robust_line_slope(md[indices], path[indices], default=0.0)


def gr_path_score(typewell: tuple[np.ndarray, np.ndarray] | None, path: np.ndarray, gr: np.ndarray, hidden_indices: np.ndarray, cal_a: float, cal_b: float) -> float:
    if typewell is None or len(hidden_indices) == 0:
        return 1.0
    tw_gr = interpolated_typewell_gr(typewell, path)
    if tw_gr is None:
        return 1.0
    h = hidden_indices
    observed = smooth(gr[h], window=min(31, max(3, len(h) // 10 * 2 + 1)))
    predicted = smooth(cal_a * tw_gr[h] + cal_b, window=min(31, max(3, len(h) // 10 * 2 + 1)))
    diff = observed - predicted
    sigma = robust_sigma(np.r_[observed, predicted], default=15.0)
    return float(np.nanmean(np.minimum((diff / sigma) ** 2, 9.0)))


def normalized_mad_distance(a: np.ndarray | None, b: np.ndarray | None, hidden_indices: np.ndarray, default: float = 0.0) -> float:
    if a is None or b is None or len(hidden_indices) == 0:
        return float(default)
    diff = np.asarray(a, dtype=float)[hidden_indices] - np.asarray(b, dtype=float)[hidden_indices]
    finite = _finite(diff)
    if len(finite) == 0:
        return float(default)
    scale = max(robust_sigma(np.asarray(b, dtype=float)[hidden_indices], default=10.0), 3.0)
    return float(np.median(np.abs(finite)) / scale)


def score_candidate_path(name: str, path: np.ndarray, variant: SolverVariant, typewell: tuple[np.ndarray, np.ndarray] | None, gr: np.ndarray, hidden_indices: np.ndarray, md: np.ndarray, linear_path: np.ndarray, geo_path: np.ndarray, anchor_path: np.ndarray | None, cal_a: float, cal_b: float, tail_slope: float) -> float:
    gr_score = gr_path_score(typewell, path, gr, hidden_indices, cal_a, cal_b)
    geo_score = normalized_mad_distance(path, geo_path, hidden_indices, default=0.0)
    anchor_score = normalized_mad_distance(path, anchor_path, hidden_indices, default=0.0)
    candidate_slope = path_slope(path, md, hidden_indices)
    slope_score = min(((candidate_slope - tail_slope) / max(abs(tail_slope), 0.03)) ** 2, 16.0)
    endpoint_score = 0.0
    if anchor_path is not None and len(hidden_indices):
        endpoint_score = abs(float(path[hidden_indices[-1]] - anchor_path[hidden_indices[-1]])) / 15.0
    range_penalty = 0.0
    if len(hidden_indices):
        span = float(np.nanmax(path[hidden_indices]) - np.nanmin(path[hidden_indices]))
        if span > 250.0:
            range_penalty = (span - 250.0) / 50.0
    return float(
        variant.gr_weight * gr_score
        + variant.geo_weight * geo_score
        + variant.anchor_weight * anchor_score
        + variant.slope_weight * slope_score
        + variant.endpoint_weight * endpoint_score
        + range_penalty
    )


_ENERGY_VARIANT = SolverVariant(
    name="_energy",
    gr_weight=1.00,
    geo_weight=0.50,
    anchor_weight=0.00,
    slope_weight=0.20,
    endpoint_weight=0.00,
    max_gate=1.0,
    min_improvement=0.0,
)


def solve_well(
    well: str,
    df: pd.DataFrame,
    row_indices: list[int],
    anchor_by_id: Mapping[str, float],
    train_eval: bool,
    tail_rows: int,
    train_paths: list["TrainWellPath"] | None = None,
    cross_well_k: int = 8,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Build direct test-time TVT path candidates for one well.

    Anchor-free by design. The submission anchor (if provided via
    ``anchor_by_id``) is used **only** for optional output blends and a
    diagnostic shift report, never inside the energy function or as the
    base of a correction search. The CEM correction family operates around
    the geological tailfit path, not around the submission anchor.
    """
    n = len(df)
    md = pd.to_numeric(df["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df["Z"], errors="coerce").to_numpy(float)
    x = pd.to_numeric(df.get("X", pd.Series(np.zeros(n))), errors="coerce").to_numpy(float)
    y = pd.to_numeric(df.get("Y", pd.Series(np.zeros(n))), errors="coerce").to_numpy(float)
    gr = pd.to_numeric(df.get("GR", pd.Series(np.zeros(n))), errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(df.get("TVT_input", pd.Series(np.full(n, np.nan))), errors="coerce").to_numpy(float)
    hidden_indices = np.asarray(row_indices, dtype=int)
    hidden_indices = hidden_indices[(hidden_indices >= 0) & (hidden_indices < n)]
    if len(hidden_indices) == 0:
        return {}, []
    known_before = np.flatnonzero(np.isfinite(tvt_input) & (np.arange(n) < int(hidden_indices[0])))
    if len(known_before) == 0:
        known_before = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_before) == 0:
        fallback_value = float(np.nanmedian(z)) if np.isfinite(z).any() else 0.0
        fallback = np.full(n, fallback_value, dtype=float)
        return {
            "linear_tailfit": fallback.copy(),
            "geo_tailfit": fallback.copy(),
            "stage1_raw": fallback.copy(),
            "stage12_raw": fallback.copy(),
            "cem_raw": fallback.copy(),
            "cem_top_median": fallback.copy(),
        }, []
    last_idx = int(known_before[-1])
    last_tvt = float(tvt_input[last_idx])
    tail_mask = known_tail_mask(tvt_input, last_idx, tail_rows)
    tail_slope = robust_line_slope(md[tail_mask], tvt_input[tail_mask], default=0.0)

    linear_path = fit_linear_candidate(df, md, z, x, y, tvt_input, hidden_indices, last_idx, last_tvt, tail_rows)
    geo_path, geo_consensus_path, geo_diag = fit_geo_candidate(df, md, z, tvt_input, hidden_indices, last_idx, last_tvt, tail_rows)

    # Optional submission anchor. Never falls back to a self-fabricated path:
    # if the submission anchor is missing or invalid we just return None and
    # skip the optional anchor-blend outputs.
    anchor_path: np.ndarray | None = None
    if anchor_by_id:
        anchor_buf = np.full(n, np.nan, dtype=float)
        for idx in hidden_indices:
            row_id = f"{well}_{int(idx)}"
            if row_id in anchor_by_id:
                anchor_buf[idx] = float(anchor_by_id[row_id])
        if np.isfinite(anchor_buf[hidden_indices]).any():
            anchor_path = (
                pd.Series(anchor_buf)
                .interpolate(limit_direction="both")
                .bfill()
                .ffill()
                .to_numpy(float)
            )

    horizontal = Path("dummy")
    if "horizontal_path" in df.attrs:
        horizontal = Path(str(df.attrs["horizontal_path"]))
    typewell = read_typewell(horizontal) if horizontal.name != "dummy" else None
    cal_a, cal_b, cal_rmse = fit_gr_calibration(typewell, tvt_input, gr, tail_mask)

    # CEM applies (offset + slope_offset * centered + curvature * shape) to a
    # base path. Use the geological tailfit, not the submission anchor — the
    # whole point of the simplification is to stop pinning the search to a
    # known-suboptimal anchor.
    energy_ctx = EnergyContext(
        md=md,
        gr=gr,
        z=z,
        typewell=typewell,
        hidden_indices=hidden_indices,
        last_idx=int(last_idx),
        last_tvt=float(last_tvt),
        tail_slope=float(tail_slope),
        cal_a=float(cal_a),
        cal_b=float(cal_b),
        linear_path=linear_path,
        geo_path=geo_path,
        anchor_path=geo_path,
    )

    def _energy_fn(path: np.ndarray) -> float:
        if len(hidden_indices) == 0:
            return float("inf")
        clipped = clip_path_steps(path, md, hidden_indices, last_idx, last_tvt, abs(tail_slope))
        return score_candidate_path(
            "_energy",
            clipped,
            _ENERGY_VARIANT,
            typewell,
            gr,
            hidden_indices,
            md,
            linear_path,
            geo_path,
            None,
            cal_a,
            cal_b,
            tail_slope,
        )

    cem_outputs, cem_diag = cem_path_search(energy_ctx, _energy_fn, seed=17)
    stage12_outputs, stage12_diag = stage12_path(energy_ctx, _energy_fn)

    def _clip(path: np.ndarray) -> np.ndarray:
        return clip_path_steps(path, md, hidden_indices, last_idx, last_tvt, abs(tail_slope))

    outputs: dict[str, np.ndarray] = {
        "linear_tailfit": linear_path,
        "geo_tailfit": geo_path,
        "geo_consensus": _clip(geo_consensus_path),
        "cem_raw": _clip(cem_outputs["cem_raw"]),
        "cem_top_median": _clip(cem_outputs["cem_top_median"]),
        "stage1_raw": _clip(stage12_outputs["stage1_path"]),
        "stage12_raw": _clip(stage12_outputs["stage12_path"]),
    }

    crosswell_diag: dict[str, object] = {}
    if train_paths:
        test_signature = compute_signature(df, last_idx=int(last_idx), hidden_indices=hidden_indices)
        crosswell_md, crosswell_z, crosswell_diag_raw = cross_well_typewell_path(
            test_md=md,
            test_z=z,
            hidden_indices=hidden_indices,
            last_idx=int(last_idx),
            last_tvt=float(last_tvt),
            last_md=float(md[int(last_idx)]),
            last_z=float(z[int(last_idx)]),
            test_signature=test_signature,
            train_paths=train_paths,
            k=int(cross_well_k),
            self_well=well,
        )
        crosswell_md = _clip(crosswell_md)
        crosswell_z = _clip(crosswell_z)
        crosswell_median = _clip(np.nanmedian(np.vstack([crosswell_md, crosswell_z]), axis=0))
        outputs["crosswell_md_raw"] = crosswell_md
        outputs["crosswell_z_raw"] = crosswell_z
        outputs["crosswell_median"] = crosswell_median
        crosswell_diag = dict(crosswell_diag_raw)

        # CEM corrections on top of the cross-well prior. Same energy as the
        # main CEM run; only the base path differs. The hypothesis is that
        # the cross-well prior is a *better* base path than geo_path for the
        # test wells where the official typewell is weak.
        xw_ctx = EnergyContext(
            md=md,
            gr=gr,
            z=z,
            typewell=typewell,
            hidden_indices=hidden_indices,
            last_idx=int(last_idx),
            last_tvt=float(last_tvt),
            tail_slope=float(tail_slope),
            cal_a=float(cal_a),
            cal_b=float(cal_b),
            linear_path=linear_path,
            geo_path=geo_path,
            anchor_path=crosswell_median,
        )
        cem_over_xw_outputs, cem_over_xw_diag = cem_path_search(
            xw_ctx, _energy_fn, seed=23
        )
        outputs["cem_over_crosswell_raw"] = _clip(cem_over_xw_outputs["cem_raw"])
        outputs["cem_over_crosswell_top_median"] = _clip(cem_over_xw_outputs["cem_top_median"])
        # Stash the over-crosswell CEM diagnostics so we can attribute the
        # contribution of each base path in the report.
        crosswell_diag = {
            **crosswell_diag,
            "cem_over_crosswell_best_score": cem_over_xw_diag.get("cem_best_score", float("nan")),
            "cem_over_crosswell_best_offset": cem_over_xw_diag.get("cem_best_offset", float("nan")),
            "cem_over_crosswell_best_slope_offset": cem_over_xw_diag.get("cem_best_slope_offset", float("nan")),
            "cem_over_crosswell_best_curvature": cem_over_xw_diag.get("cem_best_curvature", float("nan")),
        }

    # Optional anchor blends. Only emitted when a real submission anchor was
    # provided. These are a controlled regression safety net, not the primary
    # candidate — pick them when you specifically want to soften a bold path
    # toward the known anchor.
    if anchor_path is not None:
        blend_sources = ["cem_top_median", "stage12_raw"]
        if "crosswell_median" in outputs:
            blend_sources.append("crosswell_median")
        for source_name in blend_sources:
            raw = outputs[source_name]
            for weight in (0.40, 0.60):
                blended = anchor_path.copy()
                blended[hidden_indices] = anchor_path[hidden_indices] + float(weight) * (
                    raw[hidden_indices] - anchor_path[hidden_indices]
                )
                outputs[f"{source_name}_anchor_blend{int(round(weight * 100)):02d}"] = blended

    diagnostics: list[dict[str, object]] = []
    diag_extras: dict[str, dict[str, object]] = {
        "linear_tailfit": {},
        "geo_tailfit": {},
        "geo_consensus": {
            "geo_consensus_surfaces": geo_diag.get("geo_consensus_surfaces", float("nan")),
            "geo_consensus_rmse_min": geo_diag.get("geo_consensus_rmse_min", float("nan")),
            "geo_consensus_rmse_max": geo_diag.get("geo_consensus_rmse_max", float("nan")),
            "geo_consensus_rmse_spread": geo_diag.get("geo_consensus_rmse_spread", float("nan")),
        },
        "cem_raw": {k: v for k, v in cem_diag.items() if k != "cem_history"},
        "cem_top_median": {},
        "stage1_raw": {
            "stage1_best_a": stage12_diag.get("stage1_best_a", np.nan),
            "stage1_best_b": stage12_diag.get("stage1_best_b", np.nan),
            "stage1_best_score": stage12_diag.get("stage1_best_score", np.nan),
        },
        "stage12_raw": {
            "stage2_knots": stage12_diag.get("stage2_knots", np.nan),
            "stage2_passes": stage12_diag.get("stage2_passes", np.nan),
            "stage2_accepted": stage12_diag.get("stage2_accepted", np.nan),
            "stage2_score": stage12_diag.get("stage2_score", np.nan),
            "stage2_max_offset_used": stage12_diag.get("stage2_max_offset_used", np.nan),
        },
    }
    if crosswell_diag:
        crosswell_extra = {
            key: value
            for key, value in crosswell_diag.items()
            if key != "crosswell_neighbors"
        }
        diag_extras["crosswell_md_raw"] = crosswell_extra
        diag_extras["crosswell_z_raw"] = crosswell_extra
        diag_extras["crosswell_median"] = crosswell_extra
        diag_extras["cem_over_crosswell_raw"] = crosswell_extra
        diag_extras["cem_over_crosswell_top_median"] = crosswell_extra
    anchor_scale = (
        max(robust_sigma(anchor_path[hidden_indices], default=10.0), 3.0)
        if anchor_path is not None
        else float("nan")
    )
    for name, path_values in outputs.items():
        extra = diag_extras.get(name, {})
        if anchor_path is not None:
            shift_norm = normalized_mad_distance(path_values, anchor_path, hidden_indices, default=0.0)
            median_abs_shift = float(shift_norm * anchor_scale)
            endpoint_shift = (
                float(path_values[hidden_indices[-1]] - anchor_path[hidden_indices[-1]])
                if len(hidden_indices)
                else 0.0
            )
        else:
            median_abs_shift = float("nan")
            endpoint_shift = float("nan")
        diagnostics.append(
            {
                "well": well,
                "variant": name,
                "best_candidate": name,
                "solver_score": float(_energy_fn(path_values)),
                "tail_slope": float(tail_slope),
                "gr_cal_rmse": float(cal_rmse) if np.isfinite(cal_rmse) else np.nan,
                "geo_best_tail_rmse": geo_diag.get("geo_best_tail_rmse", np.nan),
                "geo_best_surface_id": geo_diag.get("geo_best_surface_id", -1.0),
                "median_abs_shift_vs_anchor": median_abs_shift,
                "endpoint_shift_vs_anchor": endpoint_shift,
                "train_eval": bool(train_eval),
                **extra,
            }
        )
    return outputs, diagnostics


def build_submission_frames(
    data_dir: Path,
    anchor_submission: Path | None,
    output_dir: Path,
    train_eval: bool,
    tail_rows: int,
    progress_interval: int = 25,
    max_wells: int | None = None,
    sample_seed: int | None = None,
    cross_well_enabled: bool = False,
    cross_well_k: int = 8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame | None]:
    anchor = read_anchor_submission(anchor_submission)
    train_paths: list[TrainWellPath] | None = None
    if cross_well_enabled:
        train_paths = collect_train_paths(data_dir)
        print(
            "Direct solver | cross-well prior loaded "
            f"{len(train_paths)} train wells with full TVT curves",
            flush=True,
        )
    if train_eval:
        # Train eval uses real hidden rows from train wells, not sample_submission.
        well_rows: dict[str, list[int]] = {}
        for path in sorted((data_dir / "train").glob("*__horizontal_well.csv")):
            well = path.name.replace("__horizontal_well.csv", "")
            frame = pd.read_csv(path)
            tvt_input = pd.to_numeric(frame.get("TVT_input", pd.Series(np.nan, index=frame.index)), errors="coerce")
            tvt = pd.to_numeric(frame.get("TVT", pd.Series(np.nan, index=frame.index)), errors="coerce")
            hidden = np.flatnonzero(tvt_input.isna().to_numpy() & tvt.notna().to_numpy())
            if len(hidden):
                well_rows[well] = hidden.astype(int).tolist()
    else:
        sample_path = data_dir / "sample_submission.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample_submission.csv under {data_dir}")
        sample = pd.read_csv(sample_path)
        well_rows = sample_rows(sample)

    well_items = list(well_rows.items())
    if sample_seed is not None:
        rng = np.random.default_rng(int(sample_seed))
        order = rng.permutation(len(well_items))
        well_items = [well_items[int(idx)] for idx in order]
    if max_wells is not None and max_wells > 0:
        well_items = well_items[: int(max_wells)]
    well_rows = dict(well_items)

    frames: dict[str, list[tuple[str, float]]] = {}
    diagnostics: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    total_wells = len(well_rows)
    total_rows = sum(len(rows) for rows in well_rows.values())
    progress_interval = max(1, int(progress_interval))
    started_at = perf_counter()
    mode = "train_eval" if train_eval else "test"
    print(
        "Direct solver start | "
        f"mode={mode} wells={total_wells} rows={total_rows} "
        f"tail_rows={tail_rows} anchor={'yes' if anchor else 'no'} "
        f"max_wells={max_wells if max_wells is not None else 'all'} "
        f"sample_seed={sample_seed if sample_seed is not None else 'none'} "
        f"output_dir={output_dir}",
        flush=True,
    )
    rows_done = 0
    for current, (well, rows) in enumerate(well_rows.items(), start=1):
        if should_log_progress(current, total_wells, progress_interval):
            print(
                "Direct solver well start | "
                f"mode={mode} current={current} total={total_wells} "
                f"well={well} rows={len(rows)}",
                flush=True,
            )
        path = horizontal_path(data_dir, well, train=train_eval)
        if not path.exists():
            rows_done += len(rows)
            if should_log_progress(current, total_wells, progress_interval):
                log_progress(
                    mode=mode,
                    current=current,
                    total=total_wells,
                    rows_done=rows_done,
                    total_rows=total_rows,
                    well=well,
                    started_at=started_at,
                )
            continue
        df = pd.read_csv(path)
        df.attrs["horizontal_path"] = str(path)
        outputs, diag = solve_well(
            well, df, rows, anchor,
            train_eval=train_eval, tail_rows=tail_rows,
            train_paths=train_paths, cross_well_k=cross_well_k,
        )
        diagnostics.extend(diag)
        for name, path_values in outputs.items():
            frames.setdefault(name, [])
            for idx in rows:
                row_id = f"{well}_{int(idx)}"
                value = float(path_values[int(idx)])
                if not np.isfinite(value):
                    value = float(anchor.get(row_id, np.nan)) if anchor else np.nan
                frames[name].append((row_id, value))
        if train_eval and "TVT" in df.columns:
            y_true = pd.to_numeric(df["TVT"], errors="coerce").to_numpy(float)
            for name, path_values in outputs.items():
                diff = path_values[np.asarray(rows, dtype=int)] - y_true[np.asarray(rows, dtype=int)]
                if np.isfinite(diff).any():
                    eval_rows.append(
                        {
                            "well": well,
                            "variant": name,
                            "rows": int(np.isfinite(diff).sum()),
                            "rmse": float(np.sqrt(np.nanmean(diff**2))),
                            "mae": float(np.nanmean(np.abs(diff))),
                        }
                    )
        rows_done += len(rows)
        if should_log_progress(current, total_wells, progress_interval):
            log_progress(
                mode=mode,
                current=current,
                total=total_wells,
                rows_done=rows_done,
                total_rows=total_rows,
                well=well,
                started_at=started_at,
            )
    submissions = {
        name: pd.DataFrame(rows, columns=["id", "tvt"]).sort_values("id").reset_index(drop=True)
        for name, rows in frames.items()
    }
    diag_frame = pd.DataFrame(diagnostics)
    eval_frame = pd.DataFrame(eval_rows) if train_eval else None
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in submissions.items():
        frame.to_csv(output_dir / f"submission_direct_{name}.csv", index=False)
    diag_frame.to_csv(output_dir / "direct_solver_diagnostics.csv", index=False)
    if eval_frame is not None:
        eval_frame.to_csv(output_dir / "direct_solver_train_eval_by_well.csv", index=False)
    elapsed = perf_counter() - started_at
    print(
        "Direct solver done | "
        f"mode={mode} wells={total_wells} rows={total_rows} "
        f"duration={format_duration(elapsed)} "
        f"rows_per_sec={total_rows / max(elapsed, 1e-9):.1f}",
        flush=True,
    )
    return submissions, diag_frame, eval_frame


def summarize_eval(eval_frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if eval_frame is None or eval_frame.empty:
        return None
    rows = []
    for variant, group in eval_frame.groupby("variant"):
        weights = group["rows"].to_numpy(float)
        rmse_values = group["rmse"].to_numpy(float)
        global_rmse_proxy = math.sqrt(float(np.nansum(weights * rmse_values**2) / max(np.nansum(weights), 1.0)))
        rows.append(
            {
                "variant": variant,
                "weighted_rmse": global_rmse_proxy,
                "mean_well_rmse": float(np.nanmean(rmse_values)),
                "median_well_rmse": float(np.nanmedian(rmse_values)),
                "p90_well_rmse": float(np.nanpercentile(rmse_values, 90)),
                "rows": int(np.nansum(weights)),
                "wells": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_rmse").reset_index(drop=True)


def build_pseudo_public_trials(
    eval_frame: pd.DataFrame | None,
    *,
    trials: int,
    triple_size: int,
    seed: int,
    anchor_variant: str,
    matched_triples_list: list[list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap public-like 3-well leaderboards from train hidden-row solver scores.

    When ``matched_triples_list`` is provided the harness uses those train
    triples (matched by signature to the test wells) instead of uniform-random
    triples. Any leftover trials fall back to random sampling so the total trial
    count is honored.
    """
    if eval_frame is None or eval_frame.empty or trials <= 0:
        return pd.DataFrame(), pd.DataFrame()
    required = {"well", "variant", "rows", "rmse"}
    missing = required - set(eval_frame.columns)
    if missing:
        raise ValueError(f"Missing pseudo-public eval columns: {sorted(missing)}")
    wells = sorted(str(well) for well in eval_frame["well"].dropna().unique())
    variants = sorted(str(variant) for variant in eval_frame["variant"].dropna().unique())
    if len(wells) == 0 or len(variants) == 0:
        return pd.DataFrame(), pd.DataFrame()
    triple_size = max(1, min(int(triple_size), len(wells)))
    rng = np.random.default_rng(int(seed))
    eval_index = {
        (str(row["well"]), str(row["variant"])): (
            int(row["rows"]),
            float(row["rmse"]),
        )
        for row in eval_frame.to_dict(orient="records")
    }
    available_wells = set(wells)
    matched_iter = iter(matched_triples_list or [])
    trial_rows: list[dict[str, object]] = []
    for trial in range(1, int(trials) + 1):
        triple_source = "matched"
        next_matched = next(matched_iter, None)
        if next_matched is not None:
            chosen = [str(w) for w in next_matched if str(w) in available_wells]
            if len(chosen) < triple_size:
                triple_source = "matched_padded"
                pad = [
                    wells[int(idx)]
                    for idx in rng.choice(len(wells), size=triple_size, replace=False)
                ]
                for candidate in pad:
                    if candidate not in chosen:
                        chosen.append(candidate)
                    if len(chosen) >= triple_size:
                        break
            chosen = chosen[:triple_size]
        else:
            triple_source = "random"
            chosen = [
                wells[int(idx)]
                for idx in rng.choice(len(wells), size=triple_size, replace=False)
            ]
        trial_scores: dict[str, float] = {}
        trial_counts: dict[str, int] = {}
        for variant in variants:
            rows_total = 0
            sse_total = 0.0
            for well in chosen:
                item = eval_index.get((well, variant))
                if item is None:
                    continue
                row_count, rmse = item
                if row_count <= 0 or not np.isfinite(rmse):
                    continue
                rows_total += int(row_count)
                sse_total += float(row_count) * float(rmse) ** 2
            if rows_total <= 0:
                continue
            trial_scores[variant] = math.sqrt(sse_total / rows_total)
            trial_counts[variant] = rows_total
        if not trial_scores:
            continue
        anchor_rmse = trial_scores.get(anchor_variant)
        if anchor_rmse is None:
            available = ", ".join(sorted(trial_scores))
            raise ValueError(
                f"Pseudo-public anchor_variant={anchor_variant!r} is missing from trial scores. "
                f"Available variants: {available}"
            )
        for variant, rmse in trial_scores.items():
            trial_rows.append(
                {
                    "trial": trial,
                    "wells": ",".join(chosen),
                    "variant": variant,
                    "rows": trial_counts[variant],
                    "triple_rmse": rmse,
                    "triple_source": triple_source,
                    "anchor_variant": anchor_variant,
                    "anchor_rmse": float(anchor_rmse) if anchor_rmse is not None else np.nan,
                    "delta_vs_anchor": float(rmse - anchor_rmse) if anchor_rmse is not None else np.nan,
                    "beats_anchor": bool(anchor_rmse is not None and rmse < anchor_rmse),
                }
            )
    trial_frame = pd.DataFrame(trial_rows)
    if trial_frame.empty:
        return trial_frame, pd.DataFrame()
    summary_rows = []
    for variant, group in trial_frame.groupby("variant"):
        rmse = group["triple_rmse"].to_numpy(float)
        delta = group["delta_vs_anchor"].to_numpy(float)
        summary_rows.append(
            {
                "variant": variant,
                "median_triple_rmse": float(np.nanmedian(rmse)),
                "mean_triple_rmse": float(np.nanmean(rmse)),
                "p90_triple_rmse": float(np.nanpercentile(rmse, 90)),
                "p95_triple_rmse": float(np.nanpercentile(rmse, 95)),
                "median_delta_vs_anchor": float(np.nanmedian(delta)),
                "p90_delta_vs_anchor": float(np.nanpercentile(delta, 90)),
                "win_rate_vs_anchor": float(np.nanmean(group["beats_anchor"].astype(float))),
                "trials": int(len(group)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("median_triple_rmse").reset_index(drop=True)
    return trial_frame, summary


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        rendered = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_markdown(
    output_dir: Path,
    submissions: Mapping[str, pd.DataFrame],
    diag_frame: pd.DataFrame,
    eval_summary: pd.DataFrame | None,
    pseudo_summary: pd.DataFrame | None = None,
    train_eval: bool = False,
    anchor_present: bool = True,
) -> None:
    lines: list[str] = []
    lines.append("# Direct TVT Solver Report")
    lines.append("")
    lines.append("This report is intentionally independent of fold-safe GBM CV. It creates fixed test-time path candidates and gated blends against an anchor submission.")
    lines.append("")
    lines.append("## Generated submissions")
    lines.append("")
    for name, frame in submissions.items():
        tvt = frame["tvt"].to_numpy(float)
        lines.append(f"- `{name}`: rows={len(frame)}, tvt_min={np.nanmin(tvt):.3f}, tvt_max={np.nanmax(tvt):.3f}, tvt_std={np.nanstd(tvt):.3f}")
    if train_eval and not anchor_present:
        lines.append("")
        lines.append("## Train-Eval Caveat")
        lines.append("")
        lines.append("This run did not receive a real OOF/schema anchor. Gated variants are therefore blended against the internal fallback anchor, not against the production model. Treat gated train-eval ranks as diagnostic only; prefer `*_raw` variants and the pseudo-public summary.")
    if eval_summary is not None and not eval_summary.empty:
        lines.append("")
        lines.append("## Train hidden-row sanity ranking")
        lines.append("")
        lines.append(dataframe_to_markdown(eval_summary))
    if pseudo_summary is not None and not pseudo_summary.empty:
        lines.append("")
        lines.append("## Pseudo-public triple ranking")
        lines.append("")
        lines.append("Random 3-well train-hidden leaderboards. This is a solver-family stress test, not GBM CV.")
        lines.append("")
        lines.append(dataframe_to_markdown(pseudo_summary))
    if not diag_frame.empty:
        lines.append("")
        lines.append("## Solver diagnostics by variant")
        lines.append("")
        summary = diag_frame.groupby("variant").agg(
            wells=("well", "count"),
            median_solver_score=("solver_score", "median"),
            median_shift=("median_abs_shift_vs_anchor", "median"),
            max_endpoint_shift=(
                "endpoint_shift_vs_anchor",
                lambda x: float(np.nanmax(np.abs(x))) if np.isfinite(x.to_numpy(float)).any() else float("nan"),
            ),
        ).reset_index()
        lines.append(dataframe_to_markdown(summary))
    lines.append("")
    lines.append("## Submit discipline")
    lines.append("")
    lines.append("Anchor is only a guard reference and an optional blend partner, never an energy term. Submit at most two variants: a stage12_raw-style primary and one optional anchor-blend safety net. Do not tune weights from public LB.")
    (output_dir / "direct_solver_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate direct test-time TVT path solver submissions.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Competition data directory containing train/test/sample_submission.csv")
    parser.add_argument("--anchor-submission", type=Path, default=None, help="Anchor submission, e.g. schema10/schema15, used only for gated blending and guardrails")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/direct_solver"))
    parser.add_argument("--train-eval", action="store_true", help="Evaluate fixed solver variants on train hidden rows instead of test sample_submission")
    parser.add_argument("--tail-rows", type=int, default=384)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None, help="Shuffle wells with this seed before applying --max-wells")
    parser.add_argument(
        "--cross-well-prior",
        action="store_true",
        help="Load train wells once and produce crosswell_md_raw / crosswell_z_raw / crosswell_median variants for each test well",
    )
    parser.add_argument(
        "--cross-well-k",
        type=int,
        default=8,
        help="Number of nearest train wells to aggregate for the cross-well prior",
    )
    parser.add_argument("--pseudo-public-trials", type=int, default=0, help="After --train-eval, bootstrap this many 3-well pseudo-public trials")
    parser.add_argument("--pseudo-public-triple-size", type=int, default=3)
    parser.add_argument("--pseudo-public-anchor-variant", default="stage12_raw")
    parser.add_argument(
        "--pseudo-public-matched",
        action="store_true",
        help="Pick pseudo-public triples matched to the public test wells by hidden_len/GR/MD/Z signature instead of uniform-random sampling",
    )
    parser.add_argument(
        "--pseudo-public-candidate-k",
        type=int,
        default=80,
        help="For matched triples, draw each well from the nearest-K signature neighbors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submissions, diag_frame, eval_frame = build_submission_frames(
        data_dir=args.data_dir,
        anchor_submission=args.anchor_submission,
        output_dir=args.output_dir,
        train_eval=bool(args.train_eval),
        tail_rows=int(args.tail_rows),
        progress_interval=int(args.progress_interval),
        max_wells=args.max_wells,
        sample_seed=args.sample_seed,
        cross_well_enabled=bool(args.cross_well_prior),
        cross_well_k=int(args.cross_well_k),
    )
    eval_summary = summarize_eval(eval_frame)
    if eval_summary is not None:
        eval_summary.to_csv(args.output_dir / "direct_solver_train_eval_summary.csv", index=False)
    matched_triples_list: list[list[str]] | None = None
    if bool(args.pseudo_public_matched) and bool(args.train_eval) and int(args.pseudo_public_trials) > 0:
        train_signatures = collect_well_signatures(args.data_dir, subset="train")
        test_signatures = collect_well_signatures(args.data_dir, subset="test")
        if not train_signatures.empty and not test_signatures.empty:
            train_signatures.to_csv(
                args.output_dir / "direct_solver_train_signatures.csv", index=False
            )
            test_signatures.to_csv(
                args.output_dir / "direct_solver_test_signatures.csv", index=False
            )
            matched_triples_list = matched_triples(
                train_signatures,
                test_signatures,
                trials=int(args.pseudo_public_trials),
                triple_size=int(args.pseudo_public_triple_size),
                seed=int(args.sample_seed) if args.sample_seed is not None else 42,
                candidate_k=int(args.pseudo_public_candidate_k),
            )
    pseudo_trials, pseudo_summary = build_pseudo_public_trials(
        eval_frame,
        trials=int(args.pseudo_public_trials),
        triple_size=int(args.pseudo_public_triple_size),
        seed=int(args.sample_seed) if args.sample_seed is not None else 42,
        anchor_variant=str(args.pseudo_public_anchor_variant),
        matched_triples_list=matched_triples_list,
    )
    if not pseudo_trials.empty:
        pseudo_trials.to_csv(args.output_dir / "direct_solver_pseudo_public_trials.csv", index=False)
    if not pseudo_summary.empty:
        pseudo_summary.to_csv(args.output_dir / "direct_solver_pseudo_public_summary.csv", index=False)
    write_markdown(
        args.output_dir,
        submissions,
        diag_frame,
        eval_summary,
        pseudo_summary,
        train_eval=bool(args.train_eval),
        anchor_present=bool(args.anchor_submission),
    )
    metadata = {
        "data_dir": str(args.data_dir),
        "anchor_submission": str(args.anchor_submission) if args.anchor_submission else None,
        "output_dir": str(args.output_dir),
        "train_eval": bool(args.train_eval),
        "tail_rows": int(args.tail_rows),
        "max_wells": int(args.max_wells) if args.max_wells is not None else None,
        "sample_seed": int(args.sample_seed) if args.sample_seed is not None else None,
        "pseudo_public_trials": int(args.pseudo_public_trials),
        "pseudo_public_triple_size": int(args.pseudo_public_triple_size),
        "pseudo_public_anchor_variant": str(args.pseudo_public_anchor_variant),
        "pseudo_public_matched": bool(args.pseudo_public_matched),
        "pseudo_public_candidate_k": int(args.pseudo_public_candidate_k),
        "cross_well_prior": bool(args.cross_well_prior),
        "cross_well_k": int(args.cross_well_k),
        "variants": [_ENERGY_VARIANT.__dict__],
        "submission_files": [f"submission_direct_{name}.csv" for name in submissions],
    }
    (args.output_dir / "direct_solver_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(submissions)} direct-solver submissions to {args.output_dir}")
    if eval_summary is not None:
        print(eval_summary.head(12).to_string(index=False))
    if not pseudo_summary.empty:
        print("Pseudo-public triple summary:")
        print(pseudo_summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
