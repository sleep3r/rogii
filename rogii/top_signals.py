from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FORMATIONS
from .io import typewell_path, well_name
from .numeric import as_float_array, fill_numeric, nearest_index, smooth_for_alignment
from .spatial import KaggleTopContext

try:
    import pywt
except Exception:  # pragma: no cover - Kaggle images may vary.
    pywt = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional unless notebook mode is requested.
    njit = None

NUMBA_AVAILABLE = njit is not None
PF_GR_SIG_MIN = 10.0
PF_GR_SIG_MAX = 60.0
PF_GR_SIG_DEF = 30.0
PF_RESAMP = 0.5


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _interp1(grid, value, vmin, step):
        i = int((value - vmin) / step)
        if i < 0:
            return grid[0]
        n = len(grid) - 1
        if i >= n:
            return grid[n]
        t = (value - vmin) / step - i
        return grid[i] * (1.0 - t) + grid[i + 1] * t

    @njit(cache=True)
    def _resample_particles(pos, aux, weights, n_particles, rough_pos, rough_aux):
        cumulative = np.zeros(n_particles + 1)
        for j in range(n_particles):
            cumulative[j + 1] = cumulative[j] + weights[j]
        u0 = np.random.uniform(0.0, 1.0 / n_particles)
        new_pos = np.empty(n_particles)
        new_aux = np.empty(n_particles)
        cursor = 0
        for j in range(n_particles):
            u = u0 + j / n_particles
            while cursor < n_particles - 1 and cumulative[cursor + 1] < u:
                cursor += 1
            new_pos[j] = pos[cursor] + rough_pos * np.random.randn()
            new_aux[j] = aux[cursor] + rough_aux * np.random.randn()
        return new_pos, new_aux

    @njit(cache=True)
    def _beam_jit(smoothed_gr, tw_gr, start_index, beam_width, move_cost, emit_scale):
        n_steps = len(smoothed_gr)
        n_ref = len(tw_gr)
        max_candidates = beam_width * 6
        beam_idx = np.zeros(beam_width, np.int64)
        beam_cost = np.full(beam_width, 1e30)
        beam_idx[0] = start_index
        beam_cost[0] = 0.0
        beam_count = np.int64(1)
        history_idx = np.zeros((n_steps, beam_width), np.int64)
        history_parent = np.zeros((n_steps, beam_width), np.int64)
        cand_idx = np.zeros(max_candidates, np.int64)
        cand_cost = np.full(max_candidates, 1e30)
        cand_parent = np.zeros(max_candidates, np.int64)

        for step in range(n_steps):
            gr_value = smoothed_gr[step]
            cand_count = np.int64(0)
            for beam_i in range(beam_count):
                idx = beam_idx[beam_i]
                cost = beam_cost[beam_i]
                for delta in range(-2, 3):
                    next_idx = idx + delta
                    if next_idx < 0 or next_idx >= n_ref:
                        continue
                    total = (
                        cost
                        + (gr_value - tw_gr[next_idx]) ** 2 / emit_scale
                        + move_cost * (delta if delta >= 0 else -delta)
                    )
                    found = np.int64(-1)
                    for cand_i in range(cand_count):
                        if cand_idx[cand_i] == next_idx:
                            found = cand_i
                            break
                    if found >= 0:
                        if total < cand_cost[found]:
                            cand_cost[found] = total
                            cand_parent[found] = beam_i
                    elif cand_count < max_candidates:
                        cand_idx[cand_count] = next_idx
                        cand_cost[cand_count] = total
                        cand_parent[cand_count] = beam_i
                        cand_count += 1

            kept = min(beam_width, cand_count)
            for i in range(kept):
                best = i
                for j in range(i + 1, cand_count):
                    if cand_cost[j] < cand_cost[best]:
                        best = j
                if best != i:
                    cand_idx[i], cand_idx[best] = cand_idx[best], cand_idx[i]
                    cand_cost[i], cand_cost[best] = cand_cost[best], cand_cost[i]
                    cand_parent[i], cand_parent[best] = (
                        cand_parent[best],
                        cand_parent[i],
                    )
            history_idx[step, :kept] = cand_idx[:kept]
            history_parent[step, :kept] = cand_parent[:kept]
            beam_idx[:kept] = cand_idx[:kept]
            beam_cost[:kept] = cand_cost[:kept]
            beam_count = kept

        best = np.int64(0)
        for beam_i in range(1, beam_count):
            if beam_cost[beam_i] < beam_cost[best]:
                best = beam_i
        path = np.zeros(n_steps, np.int64)
        beam_i = best
        for step in range(n_steps - 1, -1, -1):
            path[step] = history_idx[step, beam_i]
            beam_i = history_parent[step, beam_i]
        return path

    @njit(cache=True)
    def _lowres_dtw_path_jit(q, r, radius):
        n = len(q)
        m = len(r)
        inf = 1e18
        dp = np.full((n, m), inf)
        parent = np.full((n, m), -1, np.int8)
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

                best_cost = inf
                best_code = np.int8(-1)
                if i > 0 and j > 0 and dp[i - 1, j - 1] < best_cost:
                    best_cost = dp[i - 1, j - 1]
                    best_code = np.int8(0)
                if i > 0 and dp[i - 1, j] < best_cost:
                    best_cost = dp[i - 1, j]
                    best_code = np.int8(1)
                if j > 0 and dp[i, j - 1] < best_cost:
                    best_cost = dp[i, j - 1]
                    best_code = np.int8(2)
                dp[i, j] = cost + best_cost
                parent[i, j] = best_code

        j_end = np.int64(0)
        best_end = dp[n - 1, 0]
        for j in range(1, m):
            if dp[n - 1, j] < best_end:
                best_end = dp[n - 1, j]
                j_end = j

        i = n - 1
        j = j_end
        j_for_i = np.zeros(n, np.int64)
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
        return j_for_i

    @njit(cache=True)
    def _pf_ancc_jit(
        md_v,
        z_v,
        gr_v,
        grid_gr,
        vmin,
        step,
        gr_sigma,
        last_pos,
        init_rate,
        n_particles,
        seed,
        alpha,
        rate_noise,
        process_noise,
        init_spread,
        rough_pos,
        rough_rate,
        resample_threshold,
    ):
        np.random.seed(seed)
        pos = np.empty(n_particles)
        rate = np.empty(n_particles)
        weights = np.ones(n_particles) / n_particles
        for j in range(n_particles):
            pos[j] = last_pos + init_spread * np.random.randn()
            rate[j] = init_rate + 0.01 * np.random.randn()
        points = np.empty(len(md_v))
        std = np.empty(len(md_v))
        prev_md = md_v[0] - 1.0
        for i in range(len(md_v)):
            delta_md = md_v[i] - prev_md
            if delta_md < 1.0:
                delta_md = 1.0
            for j in range(n_particles):
                rate[j] = alpha * rate[j] + rate_noise * np.random.randn()
                pos[j] += rate[j] * delta_md + process_noise * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                lo = vmin - 50.0
                hi = vmin + len(grid_gr) * step + 50.0
                if tvt_j < lo:
                    tvt_j = lo
                if tvt_j > hi:
                    tvt_j = hi
                pos[j] = tvt_j + z_v[i]
            if not np.isnan(gr_v[i]):
                weight_sum = 0.0
                for j in range(n_particles):
                    expected = _interp1(grid_gr, pos[j] - z_v[i], vmin, step)
                    diff = (gr_v[i] - expected) / gr_sigma
                    likelihood = (
                        np.exp(-0.5 * diff * diff) if diff * diff < 600.0 else 0.0
                    )
                    weights[j] *= max(likelihood, 1e-300)
                    weight_sum += weights[j]
                if weight_sum > 0.0:
                    for j in range(n_particles):
                        weights[j] /= weight_sum
                else:
                    for j in range(n_particles):
                        weights[j] = 1.0 / n_particles
            eff_denom = 0.0
            for j in range(n_particles):
                eff_denom += weights[j] * weights[j]
            if 1.0 / eff_denom < resample_threshold * n_particles:
                pos, rate = _resample_particles(
                    pos, rate, weights, n_particles, rough_pos, rough_rate
                )
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles
            tvt_mean = 0.0
            for j in range(n_particles):
                tvt_mean += weights[j] * (pos[j] - z_v[i])
            points[i] = tvt_mean
            variance = 0.0
            for j in range(n_particles):
                centered = pos[j] - z_v[i] - tvt_mean
                variance += weights[j] * centered * centered
            std[i] = variance**0.5
            prev_md = md_v[i]
        return points, std

    @njit(cache=True)
    def _pf_z_jit(
        md_v,
        z_v,
        gr_v,
        gr_smooth_v,
        grid_gr,
        grid_smooth,
        vmin,
        step,
        gr_sigma,
        init_pos,
        init_velocity,
        beta,
        intercept,
        z_sigma,
        n_particles,
        seed,
        momentum,
        velocity_noise,
        process_noise,
        gr_weight,
        rough_pos,
        rough_velocity,
        resample_threshold,
    ):
        np.random.seed(seed)
        pos = np.empty(n_particles)
        velocity = np.empty(n_particles)
        weights = np.ones(n_particles) / n_particles
        for j in range(n_particles):
            pos[j] = init_pos + 0.5 * np.random.randn()
            velocity[j] = init_velocity + 0.02 * np.random.randn()
        points = np.empty(len(md_v))
        std = np.empty(len(md_v))
        prev_md = md_v[0] - 1.0
        prev_z = z_v[0] - 1.0
        for i in range(len(md_v)):
            delta_md = md_v[i] - prev_md
            if delta_md < 1.0:
                delta_md = 1.0
            dzdmd = (z_v[i] - prev_z) / delta_md
            expected_velocity = beta * dzdmd + intercept
            for j in range(n_particles):
                velocity[j] = (
                    momentum * velocity[j] + velocity_noise * np.random.randn()
                )
                pos[j] += velocity[j] * delta_md + process_noise * np.random.randn()
                lo = vmin - 50.0
                hi = vmin + len(grid_gr) * step + 50.0
                if pos[j] < lo:
                    pos[j] = lo
                if pos[j] > hi:
                    pos[j] = hi
            if not np.isnan(gr_v[i]):
                weight_sum = 0.0
                for j in range(n_particles):
                    expected = _interp1(grid_gr, pos[j], vmin, step)
                    diff = (gr_v[i] - expected) / gr_sigma
                    likelihood = (
                        np.exp(-0.5 * diff * diff) if diff * diff < 600.0 else 0.0
                    )
                    if not np.isnan(gr_smooth_v[i]):
                        smooth_expected = _interp1(grid_smooth, pos[j], vmin, step)
                        smooth_diff = (gr_smooth_v[i] - smooth_expected) / (
                            gr_sigma * 1.5
                        )
                        smooth_like = (
                            np.exp(-0.5 * smooth_diff * smooth_diff)
                            if smooth_diff * smooth_diff < 600.0
                            else 0.0
                        )
                        likelihood = (
                            1.0 - gr_weight
                        ) * likelihood + gr_weight * smooth_like
                    weights[j] *= max(likelihood, 1e-300)
                    weight_sum += weights[j]
                if weight_sum > 0.0:
                    for j in range(n_particles):
                        weights[j] /= weight_sum
                else:
                    for j in range(n_particles):
                        weights[j] = 1.0 / n_particles
            velocity_weight_sum = 0.0
            for j in range(n_particles):
                diff_velocity = (velocity[j] - expected_velocity) / max(
                    z_sigma * 2.0, 0.005
                )
                likelihood = (
                    np.exp(-0.5 * diff_velocity * diff_velocity)
                    if diff_velocity * diff_velocity < 600.0
                    else 0.0
                )
                weights[j] *= max(likelihood, 1e-300)
                velocity_weight_sum += weights[j]
            if velocity_weight_sum > 0.0:
                for j in range(n_particles):
                    weights[j] /= velocity_weight_sum
            else:
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles
            eff_denom = 0.0
            for j in range(n_particles):
                eff_denom += weights[j] * weights[j]
            if 1.0 / eff_denom < resample_threshold * n_particles:
                pos, velocity = _resample_particles(
                    pos, velocity, weights, n_particles, rough_pos, rough_velocity
                )
                for j in range(n_particles):
                    weights[j] = 1.0 / n_particles
            weighted_mean = 0.0
            for j in range(n_particles):
                weighted_mean += weights[j] * pos[j]
            points[i] = weighted_mean
            variance = 0.0
            for j in range(n_particles):
                centered = pos[j] - weighted_mean
                variance += weights[j] * centered * centered
            std[i] = variance**0.5
            prev_md = md_v[i]
            prev_z = z_v[i]
        return points, std


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
    smoothed_gr = smooth_for_alignment(
        gr_query, smooth_radius, float(np.nanmean(tw_gr))
    )
    idx = nearest_index(tw_tvt, start_tvt)
    path = np.empty(len(smoothed_gr), dtype=int)
    for i, gr_value in enumerate(smoothed_gr):
        candidates = np.arange(max(0, idx - 2), min(len(tw_gr), idx + 3))
        costs = ((gr_value - tw_gr[candidates]) ** 2) / max(float(emit_scale), 1e-6)
        costs += float(move_cost) * np.abs(candidates - idx)
        idx = int(candidates[int(np.argmin(costs))])
        path[i] = idx
    return tw_tvt[path].astype(float)


def require_numba_for_notebook_mode(top_cfg: dict[str, Any]) -> None:
    if str(top_cfg.get("mode", "")).lower() == "notebook" and not NUMBA_AVAILABLE:
        raise ImportError("features.kaggle_top.mode=notebook requires numba.")


def parse_beam_config(
    item: list[Any] | tuple[Any, ...],
) -> tuple[int, float, float, int, str]:
    if len(item) == 5:
        beam_width, move_cost, emit_scale, smooth_radius, tag = item
        return (
            int(beam_width),
            float(move_cost),
            float(emit_scale),
            int(smooth_radius),
            str(tag),
        )
    if len(item) == 4:
        move_cost, emit_scale, smooth_radius, tag = item
        return 10, float(move_cost), float(emit_scale), int(smooth_radius), str(tag)
    raise ValueError("kaggle_top.beam_configs entries must have 4 or 5 values.")


def beam_search_signal(
    gr_query: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    beam_width: int,
    move_cost: float,
    emit_scale: float,
    smooth_radius: int,
    require_numba: bool,
) -> np.ndarray:
    if len(gr_query) == 0:
        return np.array([], dtype=float)
    if NUMBA_AVAILABLE:
        smoothed_gr = smooth_for_alignment(
            gr_query, smooth_radius, float(np.nanmean(tw_gr))
        ).astype(np.float64)
        start_index = nearest_index(tw_tvt, start_tvt)
        path = _beam_jit(
            smoothed_gr,
            tw_gr.astype(np.float64),
            int(start_index),
            max(int(beam_width), 1),
            float(move_cost),
            max(float(emit_scale), 1e-6),
        )
        return tw_tvt[path].astype(float)
    if require_numba:
        raise ImportError("Notebook beam search requires numba.")
    return greedy_beam_signal(
        gr_query, tw_tvt, tw_gr, start_tvt, move_cost, emit_scale, smooth_radius
    )


def gr_sigma_from_known(
    gr: np.ndarray, tvt_input: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray
) -> float:
    valid = np.isfinite(gr) & np.isfinite(tvt_input)
    if valid.sum() < 20:
        return PF_GR_SIG_DEF
    diff = gr[valid] - np.interp(tvt_input[valid], tw_tvt, tw_gr)
    return float(np.clip(np.nanstd(diff), PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def typewell_grid(
    tw_tvt: np.ndarray, tw_gr: np.ndarray, step: float = 0.2
) -> tuple[np.ndarray, float, float]:
    tvt_min = float(np.nanmin(tw_tvt))
    tvt_max = float(np.nanmax(tw_tvt))
    tvt_grid = np.arange(tvt_min, tvt_max + step, step)
    return np.interp(tvt_grid, tw_tvt, tw_gr).astype(np.float64), tvt_min, float(step)


def run_pf_ancc_signal(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    hidden_idx: np.ndarray,
    known_idx: np.ndarray,
    seed: int,
    n_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(hidden_idx) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if not NUMBA_AVAILABLE:
        raise ImportError("PF_ANCC requires numba.")

    gr_sigma = gr_sigma_from_known(gr, tvt_input, tw_tvt, tw_gr)
    last_known_idx = int(known_idx[-1])
    last_pos = float(tvt_input[last_known_idx] + z[last_known_idx])
    tail_idx = known_idx[-30:]
    delta_tvt = np.diff(tvt_input[tail_idx])
    delta_z = np.diff(z[tail_idx])
    delta_md = np.diff(md[tail_idx])
    valid_delta = delta_md > 0
    init_rate = (
        float(
            np.nanmedian(
                (delta_tvt[valid_delta] + delta_z[valid_delta]) / delta_md[valid_delta]
            )
        )
        if valid_delta.sum() >= 3
        else 0.0
    )
    grid_gr, grid_min, grid_step = typewell_grid(tw_tvt, tw_gr)
    points, std = _pf_ancc_jit(
        md[hidden_idx].astype(np.float64),
        z[hidden_idx].astype(np.float64),
        gr[hidden_idx].astype(np.float64),
        grid_gr,
        grid_min,
        grid_step,
        gr_sigma,
        last_pos,
        init_rate,
        max(int(n_particles), 32),
        int(seed % (2**31 - 1)),
        0.998,
        0.002,
        0.005,
        0.3,
        0.1,
        0.001,
        PF_RESAMP,
    )
    return points.astype(float), std.astype(float)


def run_pf_z_signal(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    hidden_idx: np.ndarray,
    known_idx: np.ndarray,
    seed: int,
    n_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(hidden_idx) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if not NUMBA_AVAILABLE:
        raise ImportError("PF_Z requires numba.")

    gr_sigma = gr_sigma_from_known(gr, tvt_input, tw_tvt, tw_gr)
    known = known_idx
    delta_z = np.diff(z[known])
    delta_tvt = np.diff(tvt_input[known])
    delta_md = np.diff(md[known])
    valid_delta = delta_md > 0
    if valid_delta.sum() >= 10:
        z_velocity = delta_z[valid_delta] / delta_md[valid_delta]
        tvt_velocity = delta_tvt[valid_delta] / delta_md[valid_delta]
        design = np.column_stack([z_velocity, np.ones_like(z_velocity)])
        coef, _, _, _ = np.linalg.lstsq(design, tvt_velocity, rcond=None)
        beta = float(coef[0])
        intercept = float(coef[1])
        z_sigma = max(float(np.nanstd(tvt_velocity - (design @ coef))), 0.001)
    else:
        beta = -1.0
        intercept = 0.0
        z_sigma = 0.1

    tail_idx = known[-20:]
    tail_delta_tvt = np.diff(tvt_input[tail_idx])
    tail_delta_md = np.diff(md[tail_idx])
    valid_tail = tail_delta_md > 0
    init_velocity = (
        float(np.nanmedian(tail_delta_tvt[valid_tail] / tail_delta_md[valid_tail]))
        if valid_tail.sum() >= 3
        else 0.0
    )
    smoothed_tw_gr = (
        pd.Series(tw_gr)
        .rolling(5, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    smoothed_gr = (
        pd.Series(gr)
        .rolling(5, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    grid_gr, grid_min, grid_step = typewell_grid(tw_tvt, tw_gr)
    grid_smooth, _, _ = typewell_grid(tw_tvt, smoothed_tw_gr)
    points, std = _pf_z_jit(
        md[hidden_idx].astype(np.float64),
        z[hidden_idx].astype(np.float64),
        gr[hidden_idx].astype(np.float64),
        smoothed_gr[hidden_idx].astype(np.float64),
        grid_gr,
        grid_smooth,
        grid_min,
        grid_step,
        gr_sigma,
        float(tvt_input[known[-1]]),
        init_velocity,
        beta,
        intercept,
        z_sigma,
        max(int(n_particles), 32),
        int(seed % (2**31 - 1)),
        0.993,
        0.005,
        0.01,
        0.3,
        0.2,
        0.003,
        PF_RESAMP,
    )
    return points.astype(float), std.astype(float)


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
            result[f"ncc_{half_window}_tvt"] = np.full(
                len(query_filled), known_tvt[-1], dtype=float
            )
            result[f"ncc_{half_window}_score"] = np.zeros(
                len(query_filled), dtype=float
            )
            continue

        starts = np.arange(
            0, len(known_filled) - win + 1, max(int(stride), 1), dtype=int
        )
        window_offsets = np.arange(win, dtype=int)
        known_windows = known_filled[starts[:, None] + window_offsets[None, :]]
        known_norm = (known_windows - known_windows.mean(axis=1, keepdims=True)) / (
            known_windows.std(axis=1, keepdims=True) + 1e-6
        )

        padded_query = np.pad(query_filled, half_window, mode="edge")
        query_windows = padded_query[
            np.arange(len(query_filled))[:, None] + window_offsets[None, :]
        ]
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


def lowres_dtw_path_python(q: np.ndarray, r: np.ndarray, radius: int) -> np.ndarray:
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

            best_cost = inf
            best_code = -1
            if i > 0 and j > 0 and dp[i - 1, j - 1] < best_cost:
                best_cost = dp[i - 1, j - 1]
                best_code = 0
            if i > 0 and dp[i - 1, j] < best_cost:
                best_cost = dp[i - 1, j]
                best_code = 1
            if j > 0 and dp[i, j - 1] < best_cost:
                best_cost = dp[i, j - 1]
                best_code = 2
            dp[i, j] = cost + best_cost
            parent[i, j] = best_code

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
    return j_for_i


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

    if NUMBA_AVAILABLE:
        j_for_i = _lowres_dtw_path_jit(
            q.astype(np.float64),
            r.astype(np.float64),
            int(radius),
        )
    else:
        j_for_i = lowres_dtw_path_python(q, r, radius)
    coarse_tvt = tw_tvt[r_idx[j_for_i]]
    return np.interp(np.arange(len(full_gr)), q_idx, coarse_tvt).astype(float)


def wavelet_lowpass(
    values: np.ndarray,
    wavelet: str,
    level: int,
    fallback: float,
) -> np.ndarray:
    filled = fill_numeric(values, fallback)
    if pywt is None or len(filled) < 8 or level <= 0:
        return smooth_for_alignment(filled, 3, fallback)
    try:
        max_level = pywt.dwt_max_level(len(filled), pywt.Wavelet(wavelet).dec_len)
        effective_level = max(1, min(int(level), int(max_level)))
        coeffs = pywt.wavedec(filled, wavelet, mode="symmetric", level=effective_level)
        coeffs[1:] = [np.zeros_like(coeff) for coeff in coeffs[1:]]
        reconstructed = pywt.waverec(coeffs, wavelet, mode="symmetric")
    except Exception:
        return smooth_for_alignment(filled, 3, fallback)
    return np.asarray(reconstructed[: len(filled)], dtype=float)


def deterministic_seed(text: str) -> int:
    seed = 2166136261
    for char in text:
        seed ^= ord(char)
        seed *= 16777619
        seed &= 0xFFFFFFFF
    return int(seed)


def particle_filter_signal(
    candidate_matrix: np.ndarray,
    seed: int,
    n_particles: int,
    process_noise: float,
    observation_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidate_matrix, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    row_median = np.nanmedian(candidates, axis=1)
    global_median = (
        float(np.nanmedian(row_median)) if np.isfinite(row_median).any() else 0.0
    )
    row_median = np.where(np.isfinite(row_median), row_median, global_median)
    candidates = np.where(np.isfinite(candidates), candidates, row_median[:, None])

    n_steps = candidates.shape[0]
    n_particles = max(32, int(n_particles))
    process_noise = max(float(process_noise), 1e-3)
    observation_scale = max(float(observation_scale), 1e-3)
    rng = np.random.default_rng(seed)

    first_candidates = candidates[0]
    base = rng.choice(first_candidates, size=n_particles, replace=True)
    particles = base + rng.normal(0.0, process_noise, size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    means = np.empty(n_steps, dtype=float)
    stds = np.empty(n_steps, dtype=float)

    for step in range(n_steps):
        if step > 0:
            delta = float(np.nanmedian(candidates[step] - candidates[step - 1]))
            particles = (
                particles + delta + rng.normal(0.0, process_noise, size=n_particles)
            )

        residual = particles[:, None] - candidates[step][None, :]
        likelihood = np.exp(
            -0.5 * np.nanmin((residual / observation_scale) ** 2, axis=1)
        )
        weights *= likelihood + 1e-12
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            weights.fill(1.0 / n_particles)
        else:
            weights /= weight_sum

        mean = float(weights @ particles)
        variance = float(weights @ ((particles - mean) ** 2))
        means[step] = mean
        stds[step] = np.sqrt(max(variance, 0.0))

        effective_size = 1.0 / float(np.sum(weights**2))
        if effective_size < n_particles * 0.5:
            positions = (rng.random() + np.arange(n_particles)) / n_particles
            cumulative = np.cumsum(weights)
            indexes = np.searchsorted(cumulative, positions, side="left")
            particles = particles[np.clip(indexes, 0, n_particles - 1)]
            weights.fill(1.0 / n_particles)

    return means, stds


def empty_top_signal_features(n: int) -> dict[str, np.ndarray | float]:
    keys = [
        "kg_hidden_row",
        "kg_beam_mean_tvt",
        "kg_beam_mean_minus_flat",
        "kg_beam_mean_minus_last",
        "kg_beam_std",
        "kg_ncc_mean_tvt",
        "kg_ncc_mean_minus_flat",
        "kg_ncc_mean_minus_last",
        "kg_ncc_score_mean",
        "kg_dtw_tvt",
        "kg_dtw_minus_flat",
        "kg_dtw_minus_last",
        "kg_dtw_std",
        "kg_dtw_vs_beam",
        "kg_dwt_tvt",
        "kg_dwt_minus_flat",
        "kg_dwt_minus_last",
        "kg_dwt_std",
        "kg_dwt_vs_dtw",
        "kg_dwt_vs_beam",
        "kg_signal_mean_tvt",
        "kg_signal_mean_minus_flat",
        "kg_signal_mean_minus_last",
        "kg_signal_std",
        "kg_form_ancc_tvt",
        "kg_form_ancc_minus_flat",
        "kg_form_ancc_minus_last",
        "kg_form_mean_tvt",
        "kg_form_mean_minus_flat",
        "kg_form_mean_minus_last",
        "kg_form_std",
        "kg_form_range",
        "kg_form_knn_dist",
        "kg_dense_ancc_tvt",
        "kg_dense_ancc_minus_flat",
        "kg_dense_ancc_minus_last",
        "kg_dense_ancc_std",
        "kg_dense_ancc_dist",
        "kg_dense_vs_form",
        "kg_pf_z_tvt",
        "kg_pf_z_minus_flat",
        "kg_pf_z_minus_last",
        "kg_pf_z_std",
        "kg_pf_z_velocity",
        "kg_pf_ancc_tvt",
        "kg_pf_ancc_minus_flat",
        "kg_pf_ancc_minus_last",
        "kg_pf_ancc_std",
        "kg_pf_ancc_vs_dense",
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
    require_numba_for_notebook_mode(top_cfg)
    notebook_mode = str(top_cfg.get("mode", "")).lower() == "notebook"
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
    dwt_signal = None
    dense_signal = None
    form_ancc_signal = None

    beam_signals: list[np.ndarray] = []
    for beam_cfg in top_cfg.get("beam_configs", []):
        beam_width, move_cost, emit_scale, smooth_radius, tag = parse_beam_config(
            beam_cfg
        )
        signal = beam_search_signal(
            hidden_gr,
            tw_tvt,
            tw_gr,
            last_tvt,
            beam_width,
            float(move_cost),
            float(emit_scale),
            int(smooth_radius),
            notebook_mode,
        )
        beam_signals.append(signal)
        features[f"kg_beam_{tag}_tvt"] = np.zeros(n, dtype=float)
        features[f"kg_beam_{tag}_tvt"][hidden_idx] = signal
        features[f"kg_beam_{tag}_minus_flat"] = np.zeros(n, dtype=float)
        features[f"kg_beam_{tag}_minus_flat"][hidden_idx] = (
            signal - flat_pred[hidden_idx]
        )
        features[f"kg_beam_{tag}_minus_last"] = np.zeros(n, dtype=float)
        features[f"kg_beam_{tag}_minus_last"][hidden_idx] = signal - last_tvt

    if beam_signals:
        beam_matrix = np.vstack(beam_signals).T
        beam_mean = np.nanmean(beam_matrix, axis=1)
        features["kg_beam_mean_tvt"][hidden_idx] = beam_mean
        features["kg_beam_mean_minus_flat"][hidden_idx] = (
            beam_mean - flat_pred[hidden_idx]
        )
        features["kg_beam_mean_minus_last"][hidden_idx] = beam_mean - last_tvt
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
        if key.endswith("_tvt"):
            tvt_full = np.zeros(n, dtype=float)
            tvt_full[hidden_idx] = value
            features[f"kg_{key}"] = tvt_full
            diff_full = np.zeros(n, dtype=float)
            diff_full[hidden_idx] = value - flat_pred[hidden_idx]
            features[f"kg_{key}_minus_flat"] = diff_full
            ncc_signals.append(value)
        else:
            full = np.zeros(n, dtype=float)
            full[hidden_idx] = value
            features[f"kg_{key}"] = full
            ncc_scores.append(value)
    if ncc_signals:
        ncc_matrix = np.vstack(ncc_signals).T
        features["kg_ncc_mean_tvt"][hidden_idx] = np.nanmean(ncc_matrix, axis=1)
        features["kg_ncc_mean_minus_flat"][hidden_idx] = (
            np.nanmean(ncc_matrix, axis=1) - flat_pred[hidden_idx]
        )
        features["kg_ncc_mean_minus_last"][hidden_idx] = (
            np.nanmean(ncc_matrix, axis=1) - last_tvt
        )
    if ncc_scores:
        features["kg_ncc_score_mean"][hidden_idx] = np.nanmean(
            np.vstack(ncc_scores).T, axis=1
        )

    if top_cfg.get("dtw_enabled", True):
        dtw_hidden_signals = []
        dtw_radii = top_cfg.get("dtw_radii") or [top_cfg.get("dtw_radius", 35)]
        for radius in [int(item) for item in dtw_radii]:
            signal = lowres_dtw_signal(
                gr_full,
                tw_tvt,
                tw_gr,
                int(top_cfg.get("dtw_max_query_points", 700)),
                int(top_cfg.get("dtw_max_ref_points", 700)),
                radius,
            )
            dtw_hidden_signals.append(signal[hidden_idx])
            features[f"kg_dtw_r{radius}_tvt"] = np.zeros(n, dtype=float)
            features[f"kg_dtw_r{radius}_tvt"][hidden_idx] = signal[hidden_idx]
            features[f"kg_dtw_r{radius}_minus_flat"] = np.zeros(n, dtype=float)
            features[f"kg_dtw_r{radius}_minus_flat"][hidden_idx] = (
                signal[hidden_idx] - flat_pred[hidden_idx]
            )
            features[f"kg_dtw_r{radius}_minus_last"] = np.zeros(n, dtype=float)
            features[f"kg_dtw_r{radius}_minus_last"][hidden_idx] = (
                signal[hidden_idx] - last_tvt
            )
        dtw_matrix = np.vstack(dtw_hidden_signals).T
        dtw_hidden = np.nanmean(dtw_matrix, axis=1)
        dtw_signal = np.full(n, np.nan, dtype=float)
        dtw_signal[hidden_idx] = dtw_hidden
        features["kg_dtw_tvt"][hidden_idx] = dtw_signal[hidden_idx]
        features["kg_dtw_minus_flat"][hidden_idx] = (
            dtw_signal[hidden_idx] - flat_pred[hidden_idx]
        )
        features["kg_dtw_minus_last"][hidden_idx] = dtw_signal[hidden_idx] - last_tvt
        features["kg_dtw_std"][hidden_idx] = np.nanstd(dtw_matrix, axis=1)
        features["kg_dtw_vs_beam"][hidden_idx] = dtw_signal[hidden_idx] - beam_mean
        signal_stack = [beam_mean, dtw_signal[hidden_idx]]
    else:
        dtw_signal = None
        signal_stack = [beam_mean]

    if top_cfg.get("dwt_enabled", False):
        dwt_full = wavelet_lowpass(
            gr_full,
            str(top_cfg.get("dwt_wavelet", "db4")),
            int(top_cfg.get("dwt_level", 3)),
            float(np.nanmean(tw_gr)),
        )
        dwt_tw = wavelet_lowpass(
            tw_gr,
            str(top_cfg.get("dwt_wavelet", "db4")),
            int(top_cfg.get("dwt_level", 3)),
            float(np.nanmean(tw_gr)),
        )
        dwt_hidden_signals = []
        dwt_radii = top_cfg.get("dwt_radii") or [
            top_cfg.get("dwt_radius", top_cfg.get("dtw_radius", 35))
        ]
        for radius in [int(item) for item in dwt_radii]:
            signal = lowres_dtw_signal(
                dwt_full,
                tw_tvt,
                dwt_tw,
                int(
                    top_cfg.get(
                        "dwt_max_query_points",
                        top_cfg.get("dtw_max_query_points", 700),
                    )
                ),
                int(
                    top_cfg.get(
                        "dwt_max_ref_points", top_cfg.get("dtw_max_ref_points", 700)
                    )
                ),
                radius,
            )
            dwt_hidden_signals.append(signal[hidden_idx])
            features[f"kg_dwt_r{radius}_tvt"] = np.zeros(n, dtype=float)
            features[f"kg_dwt_r{radius}_tvt"][hidden_idx] = signal[hidden_idx]
            features[f"kg_dwt_r{radius}_minus_flat"] = np.zeros(n, dtype=float)
            features[f"kg_dwt_r{radius}_minus_flat"][hidden_idx] = (
                signal[hidden_idx] - flat_pred[hidden_idx]
            )
            features[f"kg_dwt_r{radius}_minus_last"] = np.zeros(n, dtype=float)
            features[f"kg_dwt_r{radius}_minus_last"][hidden_idx] = (
                signal[hidden_idx] - last_tvt
            )
        dwt_matrix = np.vstack(dwt_hidden_signals).T
        dwt_hidden = np.nanmean(dwt_matrix, axis=1)
        dwt_signal = np.full(n, np.nan, dtype=float)
        dwt_signal[hidden_idx] = dwt_hidden
        features["kg_dwt_tvt"][hidden_idx] = dwt_signal[hidden_idx]
        features["kg_dwt_minus_flat"][hidden_idx] = (
            dwt_signal[hidden_idx] - flat_pred[hidden_idx]
        )
        features["kg_dwt_minus_last"][hidden_idx] = dwt_signal[hidden_idx] - last_tvt
        features["kg_dwt_std"][hidden_idx] = np.nanstd(dwt_matrix, axis=1)
        if dtw_signal is not None:
            features["kg_dwt_vs_dtw"][hidden_idx] = (
                dwt_signal[hidden_idx] - dtw_signal[hidden_idx]
            )
        features["kg_dwt_vs_beam"][hidden_idx] = dwt_signal[hidden_idx] - beam_mean
        signal_stack.append(dwt_signal[hidden_idx])

    if context is not None:
        xy_hidden = np.column_stack([x[hidden_idx], y[hidden_idx]])
        form_hidden, form_dist = context.impute_formations(xy_hidden, self_well)
        xy_known = np.column_stack([x[known_idx], y[known_idx]])
        form_known, _ = context.impute_formations(xy_known, self_well)
        if form_hidden.shape[1] == len(FORMATIONS) and np.isfinite(form_hidden).any():
            form_signals = []
            for formation_idx, formation in enumerate(FORMATIONS):
                residual_base = known_tvt + z[known_idx] - form_known[:, formation_idx]
                b = (
                    float(np.nanmedian(residual_base))
                    if np.isfinite(residual_base).any()
                    else 0.0
                )
                signal = -z[hidden_idx] + form_hidden[:, formation_idx] + b
                form_signals.append(signal)
                tvt_col = f"kg_form_{formation}_tvt"
                features[tvt_col] = np.zeros(n, dtype=float)
                features[tvt_col][hidden_idx] = signal
                diff_col = f"kg_form_{formation}_minus_flat"
                features[diff_col] = np.zeros(n, dtype=float)
                features[diff_col][hidden_idx] = signal - flat_pred[hidden_idx]
            form_matrix = np.vstack(form_signals).T
            form_mean = np.nanmean(form_matrix, axis=1)
            form_ancc_signal = form_matrix[:, 0]
            features["kg_form_ancc_tvt"][hidden_idx] = form_ancc_signal
            features["kg_form_ancc_minus_flat"][hidden_idx] = (
                form_ancc_signal - flat_pred[hidden_idx]
            )
            features["kg_form_ancc_minus_last"][hidden_idx] = (
                form_ancc_signal - last_tvt
            )
            features["kg_form_mean_tvt"][hidden_idx] = form_mean
            features["kg_form_mean_minus_flat"][hidden_idx] = (
                form_mean - flat_pred[hidden_idx]
            )
            features["kg_form_mean_minus_last"][hidden_idx] = form_mean - last_tvt
            features["kg_form_std"][hidden_idx] = np.nanstd(form_matrix, axis=1)
            features["kg_form_range"][hidden_idx] = np.nanmax(
                form_matrix, axis=1
            ) - np.nanmin(form_matrix, axis=1)
            features["kg_form_knn_dist"][hidden_idx] = form_dist
            signal_stack.append(form_mean)
        else:
            form_mean = flat_pred[hidden_idx]

        dense_ancc, dense_std, dense_dist = context.impute_dense_ancc(
            xy_hidden, self_well
        )
        dense_known, _, _ = context.impute_dense_ancc(xy_known, self_well)
        dense_residual = known_tvt + z[known_idx] - dense_known
        dense_b = (
            float(np.nanmedian(dense_residual))
            if np.isfinite(dense_residual).any()
            else 0.0
        )
        dense_signal = -z[hidden_idx] + dense_ancc + dense_b
        features["kg_dense_ancc_tvt"][hidden_idx] = dense_signal
        features["kg_dense_ancc_minus_flat"][hidden_idx] = (
            dense_signal - flat_pred[hidden_idx]
        )
        features["kg_dense_ancc_minus_last"][hidden_idx] = dense_signal - last_tvt
        features["kg_dense_ancc_std"][hidden_idx] = dense_std
        features["kg_dense_ancc_dist"][hidden_idx] = dense_dist
        features["kg_dense_vs_form"][hidden_idx] = dense_signal - form_mean
        signal_stack.append(dense_signal)

    if top_cfg.get("particle_enabled", False):
        if notebook_mode:
            pf_z, pf_z_std = run_pf_z_signal(
                md,
                z,
                gr,
                tvt_input,
                tw_tvt,
                tw_gr,
                hidden_idx,
                known_idx,
                deterministic_seed(f"{well}:pf_z"),
                int(top_cfg.get("particle_count", 600)),
            )
            pf_ancc, pf_ancc_std = run_pf_ancc_signal(
                md,
                z,
                gr,
                tvt_input,
                tw_tvt,
                tw_gr,
                hidden_idx,
                known_idx,
                deterministic_seed(f"{well}:pf_ancc"),
                int(
                    top_cfg.get(
                        "ancc_particle_count", top_cfg.get("particle_count", 600)
                    )
                ),
            )
        else:
            base_candidates = [flat_pred[hidden_idx], *signal_stack]
            pf_z, pf_z_std = particle_filter_signal(
                np.vstack(base_candidates).T,
                deterministic_seed(f"{well}:pf_z"),
                int(top_cfg.get("particle_count", 192)),
                float(top_cfg.get("particle_process_noise", 4.0)),
                float(top_cfg.get("particle_observation_scale", 18.0)),
            )

            ancc_candidates = [flat_pred[hidden_idx], beam_mean]
            if dtw_signal is not None:
                ancc_candidates.append(dtw_signal[hidden_idx])
            if dwt_signal is not None:
                ancc_candidates.append(dwt_signal[hidden_idx])
            if form_ancc_signal is not None:
                ancc_candidates.append(form_ancc_signal)
            if dense_signal is not None:
                ancc_candidates.append(dense_signal)
            pf_ancc, pf_ancc_std = particle_filter_signal(
                np.vstack(ancc_candidates).T,
                deterministic_seed(f"{well}:pf_ancc"),
                int(top_cfg.get("particle_count", 192)),
                float(top_cfg.get("particle_process_noise", 4.0)),
                float(top_cfg.get("particle_observation_scale", 18.0)),
            )
        features["kg_pf_z_tvt"][hidden_idx] = pf_z
        features["kg_pf_z_minus_flat"][hidden_idx] = pf_z - flat_pred[hidden_idx]
        features["kg_pf_z_minus_last"][hidden_idx] = pf_z - last_tvt
        features["kg_pf_z_std"][hidden_idx] = pf_z_std
        features["kg_pf_z_velocity"][hidden_idx] = np.gradient(pf_z)
        features["kg_pf_ancc_tvt"][hidden_idx] = pf_ancc
        features["kg_pf_ancc_minus_flat"][hidden_idx] = pf_ancc - flat_pred[hidden_idx]
        features["kg_pf_ancc_minus_last"][hidden_idx] = pf_ancc - last_tvt
        features["kg_pf_ancc_std"][hidden_idx] = pf_ancc_std
        if dense_signal is not None:
            features["kg_pf_ancc_vs_dense"][hidden_idx] = pf_ancc - dense_signal

    signal_matrix = np.vstack(signal_stack).T
    features["kg_signal_mean_tvt"][hidden_idx] = np.nanmean(signal_matrix, axis=1)
    features["kg_signal_mean_minus_flat"][hidden_idx] = (
        np.nanmean(signal_matrix, axis=1) - flat_pred[hidden_idx]
    )
    features["kg_signal_mean_minus_last"][hidden_idx] = (
        np.nanmean(signal_matrix, axis=1) - last_tvt
    )
    features["kg_signal_std"][hidden_idx] = np.nanstd(signal_matrix, axis=1)
    features["kg_hidden_row"][hidden_idx] = 1.0
    return features
