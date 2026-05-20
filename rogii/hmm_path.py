from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .constants import FORMATIONS
from .io import typewell_path

HMM_FEATURE_COLUMNS: tuple[str, ...] = (
    "kg_hmm_tvt",
    "kg_hmm_delta",
    "kg_hmm_minus_flat",
    "kg_hmm_path_cost",
    "kg_hmm_emit_cost",
    "kg_hmm_transition_cost",
    "kg_hmm_confidence_gap",
    "kg_hmm_slope",
    "kg_hmm_curvature",
    "kg_hmm_vs_pf",
    "kg_hmm_vs_dtw",
    "kg_hmm_pf_abs_gap",
    "kg_hmm_pf_gated_tvt",
    "kg_hmm_pf_gated_delta",
    "kg_hmm_pf_gated_minus_flat",
    "kg_hmm_state_index",
    "kg_hmm_candidate_prior",
    "kg_hmm_geo_prior",
)

ABSOLUTE_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "kg_pf_ancc_tvt",
    "kg_pf_z_tvt",
    "pf_ancc",
    "pf_z",
    "kg_signal_robust_tvt",
    "kg_signal_mean_tvt",
    "kg_signal_median_tvt",
    "kg_dense_ancc_tvt",
    "tvt_dense",
)
PF_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "kg_pf_ancc_tvt",
    "kg_pf_z_tvt",
    "pf_ancc",
    "pf_z",
    "pf_ancc_delta",
    "pf_z_delta",
)
PF_ANCC_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "kg_pf_ancc_tvt",
    "pf_ancc",
    "pf_ancc_delta",
    "kg_pf_ancc_minus_last",
    "kg_pf_ancc_minus_flat",
)
DTW_CANDIDATE_TOKENS: tuple[str, ...] = ("dtw", "dwt", "beam", "ncc", "sc_")


def empty_hmm_path_features(n: int) -> dict[str, np.ndarray]:
    return {column: np.full(n, np.nan, dtype=float) for column in HMM_FEATURE_COLUMNS}


def _as_array(value: Any, n: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array), dtype=float)
    if len(array) != n:
        return None
    return array


def _robust_sigma(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return float(default)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = float(np.nanstd(finite))
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = float(default)
    return max(float(sigma), 1e-6)


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(window, 1))
    if window <= 1 or len(values) <= 2:
        return np.asarray(values, dtype=float)
    series = pd.Series(values, dtype=float).interpolate(limit_direction="both")
    return (
        series.rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _read_typewell(horizontal_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = typewell_path(horizontal_path)
    if path is None or not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=["TVT", "GR"])
    except Exception:
        return None
    tvt = frame["TVT"].to_numpy(dtype=float)
    gr = frame["GR"].to_numpy(dtype=float)
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if int(valid.sum()) < 5:
        return None
    tvt = tvt[valid]
    gr = gr[valid]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    unique_tvt, unique_idx = np.unique(tvt, return_index=True)
    return unique_tvt.astype(float), gr[unique_idx].astype(float)


def _candidate_to_tvt(
    name: str,
    value: Any,
    n: int,
    flat_pred: np.ndarray,
    last_known: np.ndarray | None,
) -> np.ndarray | None:
    array = _as_array(value, n)
    if array is None:
        return None
    if name.endswith("_tvt") or name in ABSOLUTE_CANDIDATE_COLUMNS:
        return array
    if name.endswith("_minus_flat"):
        return flat_pred + array
    if name.endswith("_minus_last") or name.endswith("_delta"):
        if last_known is None:
            return None
        return last_known + array
    if name.endswith("_d") and last_known is not None:
        return last_known + array
    return None


def _collect_candidates(
    candidate_features: Mapping[str, Any],
    n: int,
    flat_pred: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    last_known = _as_array(candidate_features.get("last_known_tvt"), n)
    all_candidates: list[np.ndarray] = []
    pf_candidates: list[np.ndarray] = []
    pf_ancc_candidates: list[np.ndarray] = []
    dtw_candidates: list[np.ndarray] = []

    for name, value in candidate_features.items():
        lower = str(name).lower()
        tvt = _candidate_to_tvt(lower, value, n, flat_pred, last_known)
        if tvt is None or not np.isfinite(tvt).any():
            continue

        # Keep the generic prior broad; exclude scalar/statistic-like helper columns.
        if (
            lower in ABSOLUTE_CANDIDATE_COLUMNS
            or lower.endswith("_tvt")
            or lower.endswith("_minus_flat")
            or lower.endswith("_minus_last")
            or lower.endswith("_delta")
            or lower in {"sig_mean_d", "beam_cons_d", "sc_cons_d", "dtw_ens_d"}
        ):
            all_candidates.append(tvt)

        if lower in PF_CANDIDATE_COLUMNS or "pf_" in lower:
            pf_candidates.append(tvt)
        if lower in PF_ANCC_CANDIDATE_COLUMNS:
            pf_ancc_candidates.append(tvt)
        if any(token in lower for token in DTW_CANDIDATE_TOKENS):
            dtw_candidates.append(tvt)

    pf_anchor = _nanmedian_stack(pf_ancc_candidates)
    if pf_anchor is None:
        pf_anchor = _nanmedian_stack(pf_candidates)
    return (
        _nanmedian_stack(all_candidates),
        pf_anchor,
        _nanmedian_stack(dtw_candidates),
    )


def _nanmedian_stack(arrays: list[np.ndarray]) -> np.ndarray | None:
    if not arrays:
        return None
    with np.errstate(all="ignore"):
        stacked = np.vstack(arrays)
        if not np.isfinite(stacked).any():
            return None
        return np.nanmedian(stacked, axis=0)


def _formation_prior(
    horizontal_df: pd.DataFrame,
    z: np.ndarray,
    tvt_input: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    known = np.isfinite(tvt_input) & np.isfinite(z)
    priors: list[np.ndarray] = []
    for formation in FORMATIONS:
        if formation not in horizontal_df.columns:
            continue
        surface = horizontal_df[formation].to_numpy(dtype=float)
        raw = surface - z
        valid = known & np.isfinite(raw)
        if int(valid.sum()) < 5:
            continue
        # Calibrate within-well on visible TVT_input only; this is train/test safe.
        bias = float(np.nanmedian(raw[valid] - tvt_input[valid]))
        prior = raw - bias
        if np.isfinite(prior).any():
            priors.append(prior)
    if not priors:
        return None, None
    stacked = np.vstack(priors)
    with np.errstate(all="ignore"):
        center = np.nanmedian(stacked, axis=0)
        spread = np.nanstd(stacked, axis=0)
    return center, spread


def _tail_tvt_slope(md: np.ndarray, tvt_input: np.ndarray, window: int) -> float:
    known = np.flatnonzero(np.isfinite(md) & np.isfinite(tvt_input))
    if len(known) < 2:
        return 0.0
    tail = known[-min(len(known), max(int(window), 2)) :]
    x = md[tail]
    y = tvt_input[tail]
    if np.ptp(x) < 1e-6:
        return 0.0
    try:
        slope = float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0
    if not np.isfinite(slope):
        return 0.0
    return float(np.clip(slope, -2.0, 2.0))


def _candidate_range(
    values: list[np.ndarray | None],
    fallback_lo: float,
    fallback_hi: float,
) -> tuple[float, float]:
    finite_parts = [array[np.isfinite(array)] for array in values if array is not None]
    finite = np.concatenate([part for part in finite_parts if len(part)]) if finite_parts else np.array([])
    if len(finite) < 3:
        return fallback_lo, fallback_hi
    lo = float(np.nanpercentile(finite, 1))
    hi = float(np.nanpercentile(finite, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return fallback_lo, fallback_hi
    return lo, hi


def _state_bounds(
    candidate_values: list[np.ndarray | None],
    fallback_lo: float,
    fallback_hi: float,
    last_known_tvt: float | None,
    top_cfg: Mapping[str, Any],
) -> tuple[float, float] | None:
    state_pad = float(top_cfg.get("hmm_state_pad", 120.0))
    min_span = max(float(top_cfg.get("hmm_min_state_span", 180.0)), 1.0)
    include_typewell_range = bool(top_cfg.get("hmm_include_typewell_range", False))

    lo, hi = _candidate_range(candidate_values, fallback_lo, fallback_hi)
    if include_typewell_range or (lo == fallback_lo and hi == fallback_hi):
        lo = min(lo, fallback_lo)
        hi = max(hi, fallback_hi)
    else:
        lo = max(lo - state_pad, fallback_lo)
        hi = min(hi + state_pad, fallback_hi)

    if last_known_tvt is not None and np.isfinite(last_known_tvt):
        lo = min(lo, float(last_known_tvt) - state_pad)
        hi = max(hi, float(last_known_tvt) + state_pad)
        lo = max(lo, fallback_lo)
        hi = min(hi, fallback_hi)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        return None

    if hi - lo < min_span:
        finite_parts = [
            array[np.isfinite(array)]
            for array in candidate_values
            if array is not None and np.isfinite(array).any()
        ]
        if finite_parts:
            center = float(np.nanmedian(np.concatenate(finite_parts)))
        elif last_known_tvt is not None and np.isfinite(last_known_tvt):
            center = float(last_known_tvt)
        else:
            center = 0.5 * (lo + hi)
        half = 0.5 * min_span
        lo = max(center - half, fallback_lo)
        hi = min(center + half, fallback_hi)
        if hi - lo < min_span:
            if lo <= fallback_lo:
                hi = min(fallback_hi, lo + min_span)
            elif hi >= fallback_hi:
                lo = max(fallback_lo, hi - min_span)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return float(lo), float(hi)


def _build_emission(
    work_idx: np.ndarray,
    state_tvt: np.ndarray,
    gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    flat_pred: np.ndarray,
    generic_prior: np.ndarray | None,
    pf_prior: np.ndarray | None,
    dtw_prior: np.ndarray | None,
    geo_prior: np.ndarray | None,
    top_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = len(work_idx)
    s = len(state_tvt)
    emission = np.zeros((t, s), dtype=float)

    tw_gr_at_state = np.interp(state_tvt, tw_tvt, tw_gr)
    gr_sigma = float(top_cfg.get("hmm_gr_sigma", 0.0) or 0.0)
    if gr_sigma <= 0.0:
        gr_sigma = _robust_sigma(tw_gr, default=30.0)
    gr_sigma = max(float(gr_sigma), 5.0)

    gr_weight = float(top_cfg.get("hmm_gr_weight", 1.0))
    prior_weight = float(top_cfg.get("hmm_prior_weight", 0.35))
    pf_weight = float(top_cfg.get("hmm_pf_weight", 0.20))
    dtw_weight = float(top_cfg.get("hmm_dtw_weight", 0.20))
    geo_weight = float(top_cfg.get("hmm_geo_weight", 0.25))
    prior_sigma = max(float(top_cfg.get("hmm_prior_sigma", 35.0)), 1.0)
    pf_sigma = max(float(top_cfg.get("hmm_pf_sigma", 25.0)), 1.0)
    dtw_sigma = max(float(top_cfg.get("hmm_dtw_sigma", 30.0)), 1.0)
    geo_sigma = max(float(top_cfg.get("hmm_geo_sigma", 40.0)), 1.0)

    center_prior = np.asarray(flat_pred, dtype=float).copy()
    if generic_prior is not None:
        valid = np.isfinite(generic_prior)
        center_prior[valid] = generic_prior[valid]

    def add_prior_cost(prior: np.ndarray | None, weight: float, sigma: float) -> None:
        if prior is None or weight == 0.0:
            return
        prior_work = prior[work_idx]
        valid_rows = np.isfinite(prior_work)
        if not valid_rows.any():
            return
        diff = (state_tvt[None, :] - prior_work[:, None]) / sigma
        emission[valid_rows] += weight * diff[valid_rows] ** 2

    gr_work = gr[work_idx]
    valid_gr = np.isfinite(gr_work)
    if valid_gr.any() and gr_weight != 0.0:
        diff = (gr_work[:, None] - tw_gr_at_state[None, :]) / gr_sigma
        emission[valid_gr] += gr_weight * diff[valid_gr] ** 2

    add_prior_cost(center_prior, prior_weight, prior_sigma)
    add_prior_cost(pf_prior, pf_weight, pf_sigma)
    add_prior_cost(dtw_prior, dtw_weight, dtw_sigma)
    add_prior_cost(geo_prior, geo_weight, geo_sigma)
    np.clip(emission, 0.0, 1e8, out=emission)
    return emission, center_prior, pf_prior, dtw_prior, geo_prior


def _viterbi_banded(
    emission: np.ndarray,
    state_tvt: np.ndarray,
    md_work: np.ndarray,
    expected_slope: float,
    transition_sigma: float,
    jump_cost: float,
    max_step_states: int,
    start_tvt: float | None,
    start_sigma: float,
    start_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, s = emission.shape
    parent = np.zeros((t, s), dtype=np.int32)
    gap_history = np.full(t, np.nan, dtype=float)
    dp_prev = emission[0].copy()
    if start_tvt is not None and np.isfinite(start_tvt) and start_weight > 0.0:
        dp_prev += start_weight * ((state_tvt - float(start_tvt)) / max(start_sigma, 1.0)) ** 2
    gap_history[0] = _cost_gap(dp_prev)

    transition_sigma = max(float(transition_sigma), 0.25)
    max_step_states = max(1, int(max_step_states))
    state_idx = np.arange(s, dtype=np.int32)

    for row in range(1, t):
        best = np.full(s, np.inf, dtype=float)
        parent_row = np.zeros(s, dtype=np.int32)
        delta_md = float(md_work[row] - md_work[row - 1])
        if not np.isfinite(delta_md) or delta_md <= 0.0:
            delta_md = 1.0
        expected_step = expected_slope * delta_md
        for offset in range(-max_step_states, max_step_states + 1):
            if offset >= 0:
                prev_slice = slice(0, s - offset)
                cur_slice = slice(offset, s)
            else:
                prev_slice = slice(-offset, s)
                cur_slice = slice(0, s + offset)
            if cur_slice.stop <= cur_slice.start:
                continue
            step = state_tvt[cur_slice] - state_tvt[prev_slice]
            transition = ((step - expected_step) / transition_sigma) ** 2
            if jump_cost:
                transition = transition + float(jump_cost) * abs(offset)
            candidate = dp_prev[prev_slice] + transition
            current_best = best[cur_slice]
            update = candidate < current_best
            if update.any():
                best_view = current_best.copy()
                parent_view = parent_row[cur_slice].copy()
                best_view[update] = candidate[update]
                parent_view[update] = state_idx[prev_slice][update]
                best[cur_slice] = best_view
                parent_row[cur_slice] = parent_view
        dp_prev = best + emission[row]
        parent[row] = parent_row
        gap_history[row] = _cost_gap(dp_prev)

    path_idx = np.zeros(t, dtype=np.int32)
    path_idx[-1] = int(np.nanargmin(dp_prev))
    for row in range(t - 1, 0, -1):
        path_idx[row - 1] = parent[row, path_idx[row]]
    path_tvt = state_tvt[path_idx]
    return path_idx, path_tvt, gap_history, emission[np.arange(t), path_idx]


def _cost_gap(cost: np.ndarray) -> float:
    finite = cost[np.isfinite(cost)]
    if len(finite) < 2:
        return np.nan
    best_two = np.partition(finite, 1)[:2]
    return float(best_two[1] - best_two[0])


def _safe_gradient(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    try:
        return np.gradient(values, x)
    except Exception:
        return np.gradient(values)


def build_hmm_path_features(
    horizontal_df: pd.DataFrame,
    horizontal_path: Path,
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    flat_pred: np.ndarray,
    candidate_features: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Build a deterministic Viterbi/HMM TVT path expert.

    This expert is intentionally feature-only.  It uses no `TVT` target values,
    calibrates formation priors only from visible `TVT_input`, and returns NaN
    columns when typewell/hidden information is unavailable.  It can therefore
    run in fold-safe OOF and inference with the same code path.
    """
    n = len(md)
    top_cfg = (config.get("features", {}) or {}).get("kaggle_top", {}) or {}
    if not bool(top_cfg.get("hmm_enabled", False)):
        return {}

    result = empty_hmm_path_features(n)
    typewell = _read_typewell(horizontal_path)
    hidden = ~np.isfinite(tvt_input)
    work_idx = np.flatnonzero(hidden)
    if typewell is None or len(work_idx) < 2:
        return result

    tw_tvt, tw_gr = typewell
    gr_smooth_window = int(top_cfg.get("hmm_gr_smooth_window", 5))
    tw_smooth_window = int(top_cfg.get("hmm_typewell_smooth_window", 9))
    gr_smooth = _smooth_1d(gr, gr_smooth_window)
    tw_gr_smooth = _smooth_1d(tw_gr, tw_smooth_window)

    flat_pred = np.asarray(flat_pred, dtype=float)
    generic_prior, pf_prior, dtw_prior = _collect_candidates(candidate_features, n, flat_pred)
    geo_prior, _geo_spread = _formation_prior(horizontal_df, z, tvt_input)

    known = np.flatnonzero(np.isfinite(tvt_input))
    last_known_tvt = float(tvt_input[known[-1]]) if len(known) else None
    expected_slope = _tail_tvt_slope(
        md,
        tvt_input,
        int(top_cfg.get("hmm_tail_slope_window", 150)),
    )

    max_states = int(top_cfg.get("hmm_max_states", 192))
    max_states = int(np.clip(max_states, 32, 512))
    state_pad = float(top_cfg.get("hmm_state_pad", 120.0))
    fallback_lo = float(np.nanmin(tw_tvt)) - state_pad
    fallback_hi = float(np.nanmax(tw_tvt)) + state_pad
    candidate_values = [flat_pred, generic_prior, pf_prior, dtw_prior, geo_prior]
    bounds = _state_bounds(
        candidate_values,
        fallback_lo,
        fallback_hi,
        last_known_tvt,
        top_cfg,
    )
    if bounds is None:
        return result
    lo, hi = bounds
    state_tvt = np.linspace(lo, hi, max_states, dtype=float)

    emission, center_prior, pf_prior, dtw_prior, geo_prior = _build_emission(
        work_idx=work_idx,
        state_tvt=state_tvt,
        gr=gr_smooth,
        tw_tvt=tw_tvt,
        tw_gr=tw_gr_smooth,
        flat_pred=flat_pred,
        generic_prior=generic_prior,
        pf_prior=pf_prior,
        dtw_prior=dtw_prior,
        geo_prior=geo_prior,
        top_cfg=top_cfg,
    )

    path_idx, path_tvt, gap, emit_cost = _viterbi_banded(
        emission=emission,
        state_tvt=state_tvt,
        md_work=md[work_idx],
        expected_slope=expected_slope,
        transition_sigma=float(top_cfg.get("hmm_transition_sigma", 1.25)),
        jump_cost=float(top_cfg.get("hmm_jump_cost", 0.03)),
        max_step_states=int(top_cfg.get("hmm_max_step_states", 9)),
        start_tvt=last_known_tvt,
        start_sigma=float(top_cfg.get("hmm_start_sigma", 12.0)),
        start_weight=float(top_cfg.get("hmm_start_weight", 1.0)),
    )

    transition_cost = np.zeros(len(work_idx), dtype=float)
    if len(work_idx) > 1:
        delta_md = np.diff(md[work_idx])
        delta_md = np.where(np.isfinite(delta_md) & (delta_md > 0.0), delta_md, 1.0)
        transition_sigma = max(float(top_cfg.get("hmm_transition_sigma", 1.25)), 0.25)
        transition_cost[1:] = ((np.diff(path_tvt) - expected_slope * delta_md) / transition_sigma) ** 2
        transition_cost[1:] += float(top_cfg.get("hmm_jump_cost", 0.03)) * np.abs(np.diff(path_idx))

    cumulative_cost = np.cumsum(emit_cost + transition_cost) / np.arange(1, len(work_idx) + 1)
    slope = _safe_gradient(path_tvt, md[work_idx])
    curvature = _safe_gradient(slope, md[work_idx])

    result["kg_hmm_tvt"][work_idx] = path_tvt
    result["kg_hmm_delta"][work_idx] = path_tvt - float(last_known_tvt if last_known_tvt is not None else 0.0)
    result["kg_hmm_minus_flat"][work_idx] = path_tvt - flat_pred[work_idx]
    result["kg_hmm_path_cost"][work_idx] = cumulative_cost
    result["kg_hmm_emit_cost"][work_idx] = emit_cost
    result["kg_hmm_transition_cost"][work_idx] = transition_cost
    result["kg_hmm_confidence_gap"][work_idx] = gap
    result["kg_hmm_slope"][work_idx] = slope
    result["kg_hmm_curvature"][work_idx] = curvature
    result["kg_hmm_state_index"][work_idx] = path_idx.astype(float)
    result["kg_hmm_candidate_prior"][work_idx] = center_prior[work_idx]
    if geo_prior is not None:
        result["kg_hmm_geo_prior"][work_idx] = geo_prior[work_idx]
    if pf_prior is not None:
        pf_work = pf_prior[work_idx]
        pf_gap = path_tvt - pf_work
        result["kg_hmm_vs_pf"][work_idx] = pf_gap
        result["kg_hmm_pf_abs_gap"][work_idx] = np.abs(pf_gap)
        gate_threshold = float(top_cfg.get("hmm_pf_gate_threshold", 6.0))
        gated_tvt = np.where(np.abs(pf_gap) <= gate_threshold, path_tvt, pf_work)
        result["kg_hmm_pf_gated_tvt"][work_idx] = gated_tvt
        result["kg_hmm_pf_gated_delta"][work_idx] = gated_tvt - float(
            last_known_tvt if last_known_tvt is not None else 0.0
        )
        result["kg_hmm_pf_gated_minus_flat"][work_idx] = gated_tvt - flat_pred[work_idx]
    if dtw_prior is not None:
        result["kg_hmm_vs_dtw"][work_idx] = path_tvt - dtw_prior[work_idx]
    return result
