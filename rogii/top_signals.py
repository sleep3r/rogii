from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .constants import FORMATIONS
from .io import typewell_path, well_name
from .numeric import as_float_array, fill_numeric, nearest_index, smooth_for_alignment
from .runlog import RunLogger
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
ANCH_OFFSETS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], dtype=float)
BEAM_OFFSETS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], dtype=float)
NCC_OFFSETS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], dtype=float)
PF_OFFSETS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], dtype=float)
DTW_OFFSETS = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 20], dtype=float)


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
    def _stochastic_dtw_paths_jit(q, r, radius, n_paths, temperature, seed):
        np.random.seed(seed)
        n = len(q)
        m = len(r)
        inf = 1e18
        slope = (m - 1) / max(n - 1, 1)
        radius = max(int(radius), 1)
        paths = np.zeros((n_paths, n), np.int64)
        base_cost = np.full((n, m), inf)

        for i in range(n):
            center = int(round(i * slope))
            lo = max(0, center - radius)
            hi = min(m - 1, center + radius)
            for j in range(lo, hi + 1):
                base_cost[i, j] = (q[i] - r[j]) ** 2

        for path_idx in range(n_paths):
            dp = np.full((n, m), inf)
            parent = np.full((n, m), -1, np.int8)
            for i in range(n):
                center = int(round(i * slope))
                lo = max(0, center - radius)
                hi = min(m - 1, center + radius)
                for j in range(lo, hi + 1):
                    u = np.random.uniform(1e-10, 1.0)
                    noise = -temperature * np.log(-np.log(u))
                    cost = base_cost[i, j] + noise
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
            while i >= 0 and j >= 0:
                paths[path_idx, i] = j
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
        return paths

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


def require_numba_for_top_signals() -> None:
    if not NUMBA_AVAILABLE:
        raise ImportError("ROGII top-solution signals require numba.")


def log_profile_stage(
    logger: RunLogger | None,
    profile_enabled: bool,
    stage: str,
    started_at: float,
    **fields: Any,
) -> None:
    if logger is None or not profile_enabled:
        return
    logger.info(
        "Feature stage",
        stage=stage,
        duration_sec=perf_counter() - started_at,
        **fields,
    )


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
    raise ImportError("ROGII beam search requires numba.")


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


def lowres_dtw_alignment(
    full_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    max_query_points: int,
    max_ref_points: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return TVT, local path slope, and a normalized path cost."""
    if len(full_gr) == 0 or len(tw_gr) == 0:
        empty = np.full(len(full_gr), np.nan, dtype=float)
        return empty, empty, np.nan

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

    ref_idx = r_idx[j_for_i]
    coarse_tvt = tw_tvt[ref_idx]
    coarse_slope = np.gradient(ref_idx.astype(float))
    cost = float(np.nanmean((q - r[j_for_i]) ** 2))
    x_full = np.arange(len(full_gr))
    tvt_signal = np.interp(x_full, q_idx, coarse_tvt).astype(float)
    slope_signal = np.interp(x_full, q_idx, coarse_slope).astype(float)
    return tvt_signal, slope_signal, cost


def run_dtw_multiscale(
    full_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    max_query_points: int,
    max_ref_points: int,
    radii: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float], np.ndarray]:
    tvt_by_radius: dict[int, np.ndarray] = {}
    slope_by_radius: dict[int, np.ndarray] = {}
    costs: dict[int, float] = {}
    weighted_signals: list[np.ndarray] = []
    inv_costs: list[float] = []

    for radius in radii:
        tvt_signal, slope_signal, cost = lowres_dtw_alignment(
            full_gr,
            tw_tvt,
            tw_gr,
            max_query_points,
            max_ref_points,
            int(radius),
        )
        tvt_by_radius[int(radius)] = tvt_signal
        slope_by_radius[int(radius)] = slope_signal
        costs[int(radius)] = cost
        inv_cost = 1.0 / (cost + 1e-6) if np.isfinite(cost) else 0.0
        weighted_signals.append(tvt_signal)
        inv_costs.append(inv_cost)

    if not weighted_signals:
        ensemble = np.full(len(full_gr), np.nan, dtype=float)
    else:
        weights = np.asarray(inv_costs, dtype=float)
        if not np.isfinite(weights).any() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(weighted_signals), dtype=float)
        weights = weights / float(weights.sum())
        ensemble = np.vstack(weighted_signals).T @ weights
    return tvt_by_radius, slope_by_radius, costs, ensemble.astype(float)


def _cost_margin(cost_values: np.ndarray) -> float:
    finite = np.sort(cost_values[np.isfinite(cost_values)])
    if len(finite) == 0:
        return np.nan
    if len(finite) == 1:
        return np.inf
    return float((finite[1] - finite[0]) / (abs(finite[0]) + 1e-6))


def _best_radius_id(radii: list[int], cost_values: np.ndarray) -> float:
    if len(radii) == 0 or not np.isfinite(cost_values).any():
        return np.nan
    return float(radii[int(np.nanargmin(cost_values))])


def _safe_nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def _safe_nanstd(values: np.ndarray) -> float:
    return float(np.nanstd(values)) if np.isfinite(values).any() else np.nan


def run_dtw_stochastic(
    full_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    max_query_points: int,
    max_ref_points: int,
    radius: int,
    n_paths: int,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(full_gr) == 0 or len(tw_gr) == 0:
        empty = np.full(len(full_gr), np.nan, dtype=float)
        return empty, empty, empty
    if not NUMBA_AVAILABLE:
        raise ImportError("Stochastic DTW requires numba.")

    q_idx = downsample_indices(len(full_gr), max_query_points)
    r_idx = downsample_indices(len(tw_gr), max_ref_points)
    q = full_gr[q_idx]
    r = tw_gr[r_idx]
    q = (q - np.nanmean(q)) / (np.nanstd(q) + 1e-6)
    r = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-6)
    paths = _stochastic_dtw_paths_jit(
        q.astype(np.float64),
        r.astype(np.float64),
        int(radius),
        max(int(n_paths), 1),
        float(temperature),
        int(seed % (2**31 - 1)),
    )
    tvt_realizations = tw_tvt[r_idx[paths]]
    mean_coarse = np.nanmean(tvt_realizations, axis=0)
    std_coarse = np.nanstd(tvt_realizations, axis=0)
    cv_coarse = std_coarse / (np.abs(mean_coarse) + 1e-6)
    x_full = np.arange(len(full_gr))
    return (
        np.interp(x_full, q_idx, mean_coarse).astype(float),
        np.interp(x_full, q_idx, std_coarse).astype(float),
        np.interp(x_full, q_idx, cv_coarse).astype(float),
    )


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


def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or float(np.nanstd(x[valid])) < 1e-6:
        return 0.0
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def affine_calibration(
    known_gr: np.ndarray, typewell_gr_at_known: np.ndarray
) -> tuple[float, float]:
    valid = np.isfinite(known_gr) & np.isfinite(typewell_gr_at_known)
    if valid.sum() < 20 or float(np.nanstd(typewell_gr_at_known[valid])) < 1e-6:
        bias = (
            float(np.nanmean(known_gr[valid]) - np.nanmean(typewell_gr_at_known[valid]))
            if valid.any()
            else 0.0
        )
        return 1.0, bias
    scale, bias = np.polyfit(typewell_gr_at_known[valid], known_gr[valid], 1)
    return float(scale), float(bias)


def segment_biases(
    known_tvt: np.ndarray,
    known_z: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float, float, float]:
    bias_values = np.asarray(known_tvt + known_z - reference, dtype=float)
    valid = np.isfinite(bias_values)
    if not valid.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    clean = bias_values[valid]
    n = len(clean)
    full = float(np.nanmedian(clean))
    one_third = n // 3
    two_thirds = 2 * n // 3
    early = float(np.nanmedian(clean[: max(1, one_third)]))
    mid = float(np.nanmedian(clean[one_third : max(one_third + 1, two_thirds)]))
    late = float(np.nanmedian(clean[max(0, n - 50) :]))
    weights = np.exp(0.02 * np.arange(n, dtype=float))
    weights /= float(weights.sum())
    weighted = float(weights @ clean)
    return full, early, mid, late, weighted


def full_feature(n: int, hidden_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    full = np.full(n, np.nan, dtype=float)
    full[hidden_idx] = values
    return full


def full_scalar(n: int, hidden_idx: np.ndarray, value: float) -> np.ndarray:
    full = np.full(n, np.nan, dtype=float)
    full[hidden_idx] = float(value)
    return full


def add_pf_beam_robust_features(
    features: dict[str, np.ndarray | float],
    n: int,
    hidden_idx: np.ndarray,
    flat_pred: np.ndarray,
    last_tvt: float,
    pf_ancc_signal: np.ndarray | None,
    pf_ancc_std: np.ndarray | None,
    beam_mean: np.ndarray,
    beam_ref: np.ndarray,
    beam_matrix: np.ndarray | None,
    dtw_signal: np.ndarray | None = None,
    dwt_signal: np.ndarray | None = None,
) -> None:
    """Add robust PF/beam candidate features.

    PF_ANCC is the anchor because it is the strongest standalone expert in
    fold-safe diagnostics. Beam candidates are allowed to smooth it only when
    they agree locally; disagreement becomes confidence/gating signal.
    """

    hidden_len = len(hidden_idx)
    candidates: list[np.ndarray] = []
    base_weights: list[float] = []

    pf = None
    if pf_ancc_signal is not None:
        pf = np.asarray(pf_ancc_signal, dtype=float)
        candidates.append(pf)
        base_weights.append(4.0)

    beam_values = np.asarray(beam_mean, dtype=float)
    candidates.append(beam_values)
    base_weights.append(1.5)
    candidates.append(np.asarray(beam_ref, dtype=float))
    base_weights.append(1.5)

    if beam_matrix is not None and beam_matrix.size:
        matrix = np.asarray(beam_matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        for column_idx in range(matrix.shape[1]):
            candidates.append(matrix[:, column_idx])
            base_weights.append(0.75)

    candidate_matrix = np.vstack(candidates).T
    base_weight_array = np.asarray(base_weights, dtype=float)
    if pf is not None and np.isfinite(pf).any():
        center = pf
    else:
        center = np.nanmedian(candidate_matrix, axis=1)

    distances = np.abs(candidate_matrix - center[:, None])
    finite = np.isfinite(candidate_matrix) & np.isfinite(center[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        row_median = np.nanmedian(np.where(finite, distances, np.nan), axis=1)
        row_mad = np.nanmedian(
            np.abs(distances - row_median[:, None]),
            axis=1,
        )
    row_median = np.where(np.isfinite(row_median), row_median, 0.0)
    row_mad = np.where(np.isfinite(row_mad), row_mad, 0.0)
    cutoff = np.maximum(30.0, row_median + 2.0 * row_mad + 15.0)
    kept = finite & (distances <= cutoff[:, None])
    if pf is not None:
        kept[:, 0] = np.isfinite(candidate_matrix[:, 0])

    weights = base_weight_array[None, :] / (1.0 + distances / 25.0)
    weights = np.where(kept & np.isfinite(weights), weights, 0.0)
    weight_sum = weights.sum(axis=1)
    weighted_values = np.where(kept, candidate_matrix, 0.0)
    robust = np.divide(
        (weighted_values * weights).sum(axis=1),
        weight_sum,
        out=np.where(np.isfinite(center), center, np.nanmean(candidate_matrix, axis=1)),
        where=weight_sum > 1e-12,
    )

    kept_values = np.where(kept, candidate_matrix, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        robust_range = np.nanmax(kept_values, axis=1) - np.nanmin(kept_values, axis=1)
        robust_std = np.nanstd(kept_values, axis=1)
    robust_range = np.where(np.isfinite(robust_range), robust_range, 0.0)
    robust_std = np.where(np.isfinite(robust_std), robust_std, 0.0)

    if beam_matrix is not None and np.asarray(beam_matrix).size:
        beam_std = np.nanstd(np.asarray(beam_matrix, dtype=float), axis=1)
    else:
        beam_std = np.zeros(hidden_len, dtype=float)
    if pf_ancc_std is not None:
        pf_std = np.asarray(pf_ancc_std, dtype=float)
    else:
        pf_std = np.full(hidden_len, np.nan, dtype=float)

    features["kg_signal_robust_tvt"] = full_feature(n, hidden_idx, robust)
    features["kg_signal_robust_minus_flat"] = full_feature(
        n, hidden_idx, robust - flat_pred[hidden_idx]
    )
    features["kg_signal_robust_minus_last"] = full_feature(
        n, hidden_idx, robust - last_tvt
    )
    features["kg_signal_robust_std"] = full_feature(n, hidden_idx, robust_std)
    features["kg_signal_robust_range"] = full_feature(n, hidden_idx, robust_range)
    if pf is not None:
        features["kg_signal_robust_vs_pf"] = full_feature(n, hidden_idx, robust - pf)
        features["pf_beam_gap"] = full_feature(n, hidden_idx, pf - beam_values)
        features["pf_beam_abs_gap"] = full_feature(
            n, hidden_idx, np.abs(pf - beam_values)
        )
        if dtw_signal is not None:
            features["pf_dtw_gap"] = full_feature(
                n, hidden_idx, pf - np.asarray(dtw_signal, dtype=float)
            )
        if dwt_signal is not None:
            features["pf_dwt_gap"] = full_feature(
                n, hidden_idx, pf - np.asarray(dwt_signal, dtype=float)
            )
    features["kg_signal_robust_vs_beam"] = full_feature(
        n, hidden_idx, robust - beam_values
    )
    features["pf_ancc_conf"] = full_feature(n, hidden_idx, 1.0 / (1.0 + pf_std))
    features["beam_conf"] = full_feature(n, hidden_idx, 1.0 / (1.0 + beam_std))


def add_offset_residuals(
    features: dict[str, np.ndarray | float],
    prefix: str,
    n: int,
    hidden_idx: np.ndarray,
    hidden_gr: np.ndarray,
    reference_tvt: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    offsets: np.ndarray,
) -> None:
    for offset in offsets:
        name = f"{prefix}{int(offset)}"
        expected_gr = np.interp(reference_tvt + offset, tw_tvt, tw_gr)
        features[name] = full_feature(n, hidden_idx, hidden_gr - expected_gr)


def empty_top_signal_features(
    n: int, robust_expert_enabled: bool = False
) -> dict[str, np.ndarray | float]:
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
        "kg_dtw_radii_agreement",
        "kg_dtw_cost_mean",
        "kg_dtw_cost_std",
        "kg_dtw_best_radius_id",
        "kg_dtw_best_radius_margin",
        "kg_dwt_tvt",
        "kg_dwt_minus_flat",
        "kg_dwt_minus_last",
        "kg_dwt_std",
        "kg_dwt_vs_dtw",
        "kg_dwt_vs_beam",
        "kg_dwt_radii_agreement",
        "kg_dwt_best_radius_id",
        "kg_dwt_best_radius_margin",
        "kg_dwt_raw_gap",
        "kg_dwt_vs_dtw_slope_gap",
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
    if robust_expert_enabled:
        keys.extend(
            [
                "kg_signal_robust_tvt",
                "kg_signal_robust_minus_flat",
                "kg_signal_robust_minus_last",
                "kg_signal_robust_std",
                "kg_signal_robust_range",
                "kg_signal_robust_vs_pf",
                "kg_signal_robust_vs_beam",
                "pf_ancc_conf",
                "beam_conf",
                "pf_beam_gap",
                "pf_beam_abs_gap",
                "pf_dtw_gap",
                "pf_dwt_gap",
            ]
        )
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
    logger: RunLogger | None = None,
) -> dict[str, np.ndarray | float]:
    n = len(df)
    top_cfg = config["features"].get("kaggle_top", {})
    features = empty_top_signal_features(
        n, robust_expert_enabled=bool(top_cfg.get("robust_expert_enabled", False))
    )
    profile_stages = bool(config["features"].get("profile_stages", False))
    require_numba_for_top_signals()

    # Pre-initialize all dynamic config-driven features with NaN so the
    # feature schema is stable regardless of typewell presence.  During
    # training, no-typewell wells produce NaN for these columns via
    # pd.concat; using NaN here keeps inference aligned with that
    # training distribution (tree models use their "missing" branch).
    _nan = np.full(n, np.nan, dtype=float)
    for beam_cfg in top_cfg.get("beam_configs", []):
        _, _, _, _, tag = parse_beam_config(beam_cfg)
        features[f"kg_beam_{tag}_tvt"] = _nan.copy()
        features[f"kg_beam_{tag}_minus_flat"] = _nan.copy()
        features[f"kg_beam_{tag}_minus_last"] = _nan.copy()
    for hw in [int(w) for w in top_cfg.get("ncc_windows", [8, 15, 25])]:
        features[f"kg_ncc_{hw}_tvt"] = _nan.copy()
        features[f"kg_ncc_{hw}_tvt_minus_flat"] = _nan.copy()
        features[f"kg_ncc_{hw}_score"] = _nan.copy()
    if top_cfg.get("dtw_enabled", True):
        for radius in [
            int(r)
            for r in (top_cfg.get("dtw_radii") or [top_cfg.get("dtw_radius", 35)])
        ]:
            features[f"kg_dtw_r{radius}_tvt"] = _nan.copy()
            features[f"kg_dtw_r{radius}_minus_flat"] = _nan.copy()
            features[f"kg_dtw_r{radius}_minus_last"] = _nan.copy()
            features[f"kg_dtw_r{radius}_cost_mean"] = _nan.copy()
            features[f"kg_dtw_r{radius}_path_slope_mean"] = _nan.copy()
            features[f"kg_dtw_r{radius}_path_slope_std"] = _nan.copy()
            features[f"kg_dtw_r{radius}_local_stretch"] = _nan.copy()
            features[f"kg_dtw_r{radius}_endpoint_gap"] = _nan.copy()
            features[f"kg_dtw_r{radius}_vs_ensemble"] = _nan.copy()
    if top_cfg.get("dwt_enabled", False):
        for radius in [
            int(r)
            for r in (top_cfg.get("dwt_radii") or [top_cfg.get("dwt_radius", 35)])
        ]:
            features[f"kg_dwt_r{radius}_tvt"] = _nan.copy()
            features[f"kg_dwt_r{radius}_minus_flat"] = _nan.copy()
            features[f"kg_dwt_r{radius}_minus_last"] = _nan.copy()
            features[f"kg_dwt_r{radius}_cost_mean"] = _nan.copy()
            features[f"kg_dwt_r{radius}_path_slope_mean"] = _nan.copy()
            features[f"kg_dwt_r{radius}_path_slope_std"] = _nan.copy()
            features[f"kg_dwt_r{radius}_local_stretch"] = _nan.copy()
            features[f"kg_dwt_r{radius}_endpoint_gap"] = _nan.copy()
            features[f"kg_dwt_r{radius}_vs_ensemble"] = _nan.copy()
    for formation in FORMATIONS:
        features[f"kg_form_{formation}_tvt"] = _nan.copy()
        features[f"kg_form_{formation}_minus_flat"] = _nan.copy()

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
    dwt_hidden_signal = None
    dense_signal = None
    form_ancc_signal = None
    pf_ancc_signal = None
    pf_ancc_std_signal = None

    stage_started_at = perf_counter()
    beam_signals: list[np.ndarray] = []
    beam_by_tag: dict[str, np.ndarray] = {}
    beam_matrix = None
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
        )
        beam_signals.append(signal)
        beam_by_tag[tag] = signal
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
        beam_median = np.nanmedian(beam_matrix, axis=1)
        features["kg_beam_mean_tvt"][hidden_idx] = beam_mean
        features["kg_beam_mean_minus_flat"][hidden_idx] = (
            beam_mean - flat_pred[hidden_idx]
        )
        features["kg_beam_mean_minus_last"][hidden_idx] = beam_mean - last_tvt
        features["kg_beam_std"][hidden_idx] = np.nanstd(beam_matrix, axis=1)
    else:
        beam_mean = flat_pred[hidden_idx]
        beam_median = beam_mean

    if "cons" in beam_by_tag and "sm5" in beam_by_tag:
        beam_ref = (beam_by_tag["cons"] + beam_by_tag["sm5"]) / 2.0
    else:
        beam_ref = beam_mean
    for tag, signal in beam_by_tag.items():
        features[f"beam_{tag}_d"] = full_feature(n, hidden_idx, signal - last_tvt)
    features["beam_mean_d"] = full_feature(n, hidden_idx, beam_mean - last_tvt)
    features["beam_std_d"] = full_feature(
        n,
        hidden_idx,
        np.nanstd(beam_matrix, axis=1) if beam_signals else np.zeros(len(hidden_idx)),
    )
    features["beam_med_d"] = full_feature(n, hidden_idx, beam_median - last_tvt)
    log_profile_stage(
        logger,
        profile_stages,
        "top.beam",
        stage_started_at,
        signals=len(beam_signals),
        hidden_rows=len(hidden_idx),
    )

    stage_started_at = perf_counter()
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

    ncc_windows = [int(item) for item in top_cfg.get("ncc_windows", [8, 15, 25])]
    ncc_tvt_by_window = {
        window: ncc.get(f"ncc_{window}_tvt", np.full(len(hidden_idx), last_tvt))
        for window in ncc_windows
    }
    ncc_score_by_window = {
        window: ncc.get(f"ncc_{window}_score", np.zeros(len(hidden_idx), dtype=float))
        for window in ncc_windows
    }
    if ncc_tvt_by_window:
        ncc_matrix = np.vstack(list(ncc_tvt_by_window.values())).T
        ncc_cons = np.nanmean(ncc_matrix, axis=1)
    else:
        ncc_cons = np.full(len(hidden_idx), last_tvt, dtype=float)
    if ncc_scores:
        score_matrix = np.vstack(list(ncc_score_by_window.values())).T
        score_weights = np.exp(3.0 * score_matrix)
        score_weights /= score_weights.sum(axis=1, keepdims=True) + 1e-9
        ncc_ens = (ncc_matrix * score_weights).sum(axis=1)
    else:
        ncc_ens = ncc_cons
    known_trust = float(np.clip(len(known_idx) / 200.0, 0.0, 0.6))
    hybrid_ref = (1.0 - known_trust) * beam_ref + known_trust * ncc_ens
    for window in ncc_windows:
        features[f"sc{window}_d"] = full_feature(
            n, hidden_idx, ncc_tvt_by_window[window] - last_tvt
        )
        features[f"sc{window}_sc"] = full_feature(
            n, hidden_idx, ncc_score_by_window[window]
        )
    if 8 in ncc_tvt_by_window:
        features["sc8_d"] = full_feature(n, hidden_idx, ncc_tvt_by_window[8] - last_tvt)
        features["sc8_sc"] = full_feature(n, hidden_idx, ncc_score_by_window[8])
    if 15 in ncc_tvt_by_window:
        features["sc15_d"] = full_feature(
            n, hidden_idx, ncc_tvt_by_window[15] - last_tvt
        )
        features["sc15_sc"] = full_feature(n, hidden_idx, ncc_score_by_window[15])
    if 25 in ncc_tvt_by_window:
        features["sc25_d"] = full_feature(
            n, hidden_idx, ncc_tvt_by_window[25] - last_tvt
        )
        features["sc25_sc"] = full_feature(n, hidden_idx, ncc_score_by_window[25])
    features["sc_cons_d"] = full_feature(n, hidden_idx, ncc_cons - last_tvt)
    features["sc_ens_d"] = full_feature(n, hidden_idx, ncc_ens - last_tvt)
    features["sc_trust"] = full_scalar(n, hidden_idx, known_trust)
    features["hyb_d"] = full_feature(n, hidden_idx, hybrid_ref - last_tvt)
    log_profile_stage(
        logger,
        profile_stages,
        "top.ncc",
        stage_started_at,
        windows=len(ncc_windows),
        hidden_rows=len(hidden_idx),
    )

    stage_started_at = perf_counter()
    if top_cfg.get("dtw_enabled", True):
        dtw_radii = [
            int(item)
            for item in (top_cfg.get("dtw_radii") or [top_cfg.get("dtw_radius", 35)])
        ]
        dtw_by_radius, dtw_slope_by_radius, dtw_costs, dtw_ensemble = (
            run_dtw_multiscale(
                gr_full,
                tw_tvt,
                tw_gr,
                int(top_cfg.get("dtw_max_query_points", 700)),
                int(top_cfg.get("dtw_max_ref_points", 700)),
                dtw_radii,
            )
        )
        dtw_hidden_signals = []
        dtw_hidden_slopes = []
        for radius in dtw_radii:
            signal = dtw_by_radius[radius]
            slope = dtw_slope_by_radius[radius]
            dtw_hidden_signals.append(signal[hidden_idx])
            dtw_hidden_slopes.append(slope[hidden_idx])
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
            features[f"dtw_r{radius}_d"] = full_feature(
                n, hidden_idx, signal[hidden_idx] - last_tvt
            )
            features[f"dtw_slope_r{radius}"] = full_feature(
                n, hidden_idx, slope[hidden_idx]
            )
            hidden_slope = slope[hidden_idx]
            slope_center = _safe_nanmean(hidden_slope)
            features[f"kg_dtw_r{radius}_cost_mean"] = full_scalar(
                n, hidden_idx, dtw_costs[radius]
            )
            features[f"kg_dtw_r{radius}_path_slope_mean"] = full_scalar(
                n, hidden_idx, slope_center
            )
            features[f"kg_dtw_r{radius}_path_slope_std"] = full_scalar(
                n, hidden_idx, _safe_nanstd(hidden_slope)
            )
            features[f"kg_dtw_r{radius}_local_stretch"] = full_feature(
                n, hidden_idx, hidden_slope - slope_center
            )
        dtw_matrix = np.vstack(dtw_hidden_signals).T
        dtw_hidden = dtw_ensemble[hidden_idx]
        dtw_slope_matrix = np.vstack(dtw_hidden_slopes).T
        dtw_slope_mean = np.nanmean(dtw_slope_matrix, axis=1)
        dtw_signal = np.full(n, np.nan, dtype=float)
        dtw_signal[hidden_idx] = dtw_hidden
        features["kg_dtw_tvt"][hidden_idx] = dtw_signal[hidden_idx]
        features["kg_dtw_minus_flat"][hidden_idx] = (
            dtw_signal[hidden_idx] - flat_pred[hidden_idx]
        )
        features["kg_dtw_minus_last"][hidden_idx] = dtw_signal[hidden_idx] - last_tvt
        dtw_radius_std = np.nanstd(dtw_matrix, axis=1)
        features["kg_dtw_std"][hidden_idx] = dtw_radius_std
        features["kg_dtw_radii_agreement"][hidden_idx] = 1.0 / (
            1.0 + dtw_radius_std
        )
        features["kg_dtw_vs_beam"][hidden_idx] = dtw_signal[hidden_idx] - beam_mean
        cost_values = np.array([dtw_costs[radius] for radius in dtw_radii], dtype=float)
        features["dtw_ens_d"] = full_feature(n, hidden_idx, dtw_hidden - last_tvt)
        features["dtw_slope_mean"] = full_feature(n, hidden_idx, dtw_slope_mean)
        features["dtw_cost_min"] = full_scalar(
            n, hidden_idx, float(np.nanmin(cost_values))
        )
        features["dtw_cost_range"] = full_scalar(
            n, hidden_idx, float(np.nanmax(cost_values) - np.nanmin(cost_values))
        )
        features["kg_dtw_cost_mean"] = full_scalar(
            n, hidden_idx, _safe_nanmean(cost_values)
        )
        features["kg_dtw_cost_std"] = full_scalar(
            n, hidden_idx, _safe_nanstd(cost_values)
        )
        features["kg_dtw_best_radius_id"] = full_scalar(
            n, hidden_idx, _best_radius_id(dtw_radii, cost_values)
        )
        features["kg_dtw_best_radius_margin"] = full_scalar(
            n, hidden_idx, _cost_margin(cost_values)
        )
        for radius, radius_signal in zip(dtw_radii, dtw_hidden_signals, strict=False):
            features[f"kg_dtw_r{radius}_vs_ensemble"] = full_feature(
                n, hidden_idx, radius_signal - dtw_hidden
            )
            endpoint_gap = (
                float(radius_signal[-1] - dtw_hidden[-1])
                if len(radius_signal) and np.isfinite(radius_signal[-1])
                else np.nan
            )
            features[f"kg_dtw_r{radius}_endpoint_gap"] = full_scalar(
                n, hidden_idx, endpoint_gap
            )
        features["dtw_vs_beam"] = full_feature(n, hidden_idx, dtw_hidden - beam_ref)
        features["dtw_vs_sc"] = full_feature(n, hidden_idx, dtw_hidden - ncc_ens)
        if top_cfg.get("dtw_stochastic_enabled", True):
            stoch_mean, stoch_std, stoch_cv = run_dtw_stochastic(
                gr_full,
                tw_tvt,
                tw_gr,
                int(top_cfg.get("dtw_max_query_points", 700)),
                int(top_cfg.get("dtw_max_ref_points", 700)),
                int(top_cfg.get("dtw_stochastic_radius", 50)),
                int(top_cfg.get("dtw_stochastic_k", 12)),
                float(top_cfg.get("dtw_stochastic_temperature", 3.0)),
                deterministic_seed(f"{well}:dtw_stochastic"),
            )
            features["dtw_stoch_mean_d"] = full_feature(
                n, hidden_idx, stoch_mean[hidden_idx] - last_tvt
            )
            features["dtw_stoch_std"] = full_feature(
                n, hidden_idx, stoch_std[hidden_idx]
            )
            features["dtw_stoch_cv"] = full_feature(n, hidden_idx, stoch_cv[hidden_idx])
    else:
        dtw_signal = None
    signal_stack: list[np.ndarray] = []
    if beam_by_tag:
        signal_stack.extend(beam_by_tag.values())
    else:
        signal_stack.append(beam_mean)
    signal_stack.extend(ncc_tvt_by_window.values())
    signal_stack.append(ncc_ens)
    if dtw_signal is not None:
        signal_stack.append(dtw_signal[hidden_idx])
    log_profile_stage(
        logger,
        profile_stages,
        "top.dtw",
        stage_started_at,
        enabled=bool(top_cfg.get("dtw_enabled", True)),
        hidden_rows=len(hidden_idx),
    )

    stage_started_at = perf_counter()
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
        dwt_hidden_slopes = []
        dwt_costs: dict[int, float] = {}
        dwt_radii = [
            int(item)
            for item in (
                top_cfg.get("dwt_radii")
                or [top_cfg.get("dwt_radius", top_cfg.get("dtw_radius", 35))]
            )
        ]
        for radius in dwt_radii:
            signal, slope, cost = lowres_dtw_alignment(
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
            dwt_costs[int(radius)] = cost
            dwt_hidden_signals.append(signal[hidden_idx])
            dwt_hidden_slopes.append(slope[hidden_idx])
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
            hidden_slope = slope[hidden_idx]
            slope_center = _safe_nanmean(hidden_slope)
            features[f"kg_dwt_r{radius}_cost_mean"] = full_scalar(
                n, hidden_idx, cost
            )
            features[f"kg_dwt_r{radius}_path_slope_mean"] = full_scalar(
                n, hidden_idx, slope_center
            )
            features[f"kg_dwt_r{radius}_path_slope_std"] = full_scalar(
                n, hidden_idx, _safe_nanstd(hidden_slope)
            )
            features[f"kg_dwt_r{radius}_local_stretch"] = full_feature(
                n, hidden_idx, hidden_slope - slope_center
            )
        dwt_matrix = np.vstack(dwt_hidden_signals).T
        dwt_hidden = np.nanmean(dwt_matrix, axis=1)
        dwt_hidden_signal = dwt_hidden
        dwt_slope_matrix = np.vstack(dwt_hidden_slopes).T
        dwt_slope_mean = np.nanmean(dwt_slope_matrix, axis=1)
        dwt_signal = np.full(n, np.nan, dtype=float)
        dwt_signal[hidden_idx] = dwt_hidden
        features["kg_dwt_tvt"][hidden_idx] = dwt_signal[hidden_idx]
        features["kg_dwt_minus_flat"][hidden_idx] = (
            dwt_signal[hidden_idx] - flat_pred[hidden_idx]
        )
        features["kg_dwt_minus_last"][hidden_idx] = dwt_signal[hidden_idx] - last_tvt
        dwt_radius_std = np.nanstd(dwt_matrix, axis=1)
        features["kg_dwt_std"][hidden_idx] = dwt_radius_std
        features["kg_dwt_radii_agreement"][hidden_idx] = 1.0 / (
            1.0 + dwt_radius_std
        )
        features["dwt_ens_d"] = full_feature(n, hidden_idx, dwt_hidden - last_tvt)
        dwt_cost_values = np.array(
            [dwt_costs[radius] for radius in dwt_radii], dtype=float
        )
        features["kg_dwt_best_radius_id"] = full_scalar(
            n, hidden_idx, _best_radius_id(dwt_radii, dwt_cost_values)
        )
        features["kg_dwt_best_radius_margin"] = full_scalar(
            n, hidden_idx, _cost_margin(dwt_cost_values)
        )
        for radius, radius_signal in zip(dwt_radii, dwt_hidden_signals, strict=False):
            features[f"kg_dwt_r{radius}_vs_ensemble"] = full_feature(
                n, hidden_idx, radius_signal - dwt_hidden
            )
            endpoint_gap = (
                float(radius_signal[-1] - dwt_hidden[-1])
                if len(radius_signal) and np.isfinite(radius_signal[-1])
                else np.nan
            )
            features[f"kg_dwt_r{radius}_endpoint_gap"] = full_scalar(
                n, hidden_idx, endpoint_gap
            )
        if dtw_signal is not None:
            features["kg_dwt_vs_dtw"][hidden_idx] = (
                dwt_signal[hidden_idx] - dtw_signal[hidden_idx]
            )
            features["kg_dwt_raw_gap"][hidden_idx] = np.abs(
                dwt_signal[hidden_idx] - dtw_signal[hidden_idx]
            )
            features["kg_dwt_vs_dtw_slope_gap"][hidden_idx] = (
                dwt_slope_mean - dtw_slope_mean
            )
        features["kg_dwt_vs_beam"][hidden_idx] = dwt_signal[hidden_idx] - beam_mean
        signal_stack.append(dwt_signal[hidden_idx])
    log_profile_stage(
        logger,
        profile_stages,
        "top.dwt",
        stage_started_at,
        enabled=bool(top_cfg.get("dwt_enabled", False)),
        hidden_rows=len(hidden_idx),
    )

    stage_started_at = perf_counter()
    if context is not None:
        formation_started_at = perf_counter()
        xy_hidden = np.column_stack([x[hidden_idx], y[hidden_idx]])
        form_hidden, form_dist = context.impute_formations(xy_hidden, self_well)
        xy_known = np.column_stack([x[known_idx], y[known_idx]])
        form_known, _ = context.impute_formations(xy_known, self_well)
        if form_hidden.shape[1] == len(FORMATIONS) and np.isfinite(form_hidden).any():
            form_signals = []
            form_rmse: dict[str, float] = {}
            for formation_idx, formation in enumerate(FORMATIONS):
                b_full, b_early, b_mid, b_late, b_weighted = segment_biases(
                    known_tvt, z[known_idx], form_known[:, formation_idx]
                )
                signal = -z[hidden_idx] + form_hidden[:, formation_idx] + b_full
                signal_weighted = (
                    -z[hidden_idx] + form_hidden[:, formation_idx] + b_weighted
                )
                signal_late = -z[hidden_idx] + form_hidden[:, formation_idx] + b_late
                form_signals.append(signal)
                tvt_col = f"kg_form_{formation}_tvt"
                features[tvt_col] = np.zeros(n, dtype=float)
                features[tvt_col][hidden_idx] = signal
                diff_col = f"kg_form_{formation}_minus_flat"
                features[diff_col] = np.zeros(n, dtype=float)
                features[diff_col][hidden_idx] = signal - flat_pred[hidden_idx]
                features[f"tvtF_{formation}"] = full_feature(n, hidden_idx, signal)
                features[f"tvtFw_{formation}"] = full_feature(
                    n, hidden_idx, signal_weighted
                )
                features[f"tvtF50_{formation}"] = full_feature(
                    n, hidden_idx, signal_late
                )
                features[f"bw_{formation}"] = full_scalar(n, hidden_idx, b_full)
                features[f"bww_{formation}"] = full_scalar(n, hidden_idx, b_weighted)
                features[f"bw50_{formation}"] = full_scalar(n, hidden_idx, b_late)
                features[f"bw_early_{formation}"] = full_scalar(n, hidden_idx, b_early)
                features[f"bw_mid_{formation}"] = full_scalar(n, hidden_idx, b_mid)
                known_signal = -z[known_idx] + form_known[:, formation_idx] + b_full
                form_rmse[formation] = float(
                    np.sqrt(np.nanmean((known_tvt - known_signal) ** 2))
                )
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
            features["form_mean_d"] = full_feature(n, hidden_idx, form_mean - last_tvt)
            features["form_std_d"] = full_feature(
                n, hidden_idx, np.nanstd(form_matrix, axis=1)
            )
            features["form_rng_d"] = full_feature(
                n,
                hidden_idx,
                np.nanmax(form_matrix, axis=1) - np.nanmin(form_matrix, axis=1),
            )
            features["spatial_ancc_d"] = full_feature(
                n,
                hidden_idx,
                form_hidden[:, 0] - np.interp(last_tvt, tw_tvt, tw_gr),
            )
            features["spatial_knn_dist"] = full_feature(n, hidden_idx, form_dist)
            for formation, score in form_rmse.items():
                features[f"frm_rmse_{formation}"] = full_scalar(n, hidden_idx, score)
            signal_stack.append(form_ancc_signal)
        else:
            form_mean = flat_pred[hidden_idx]
        log_profile_stage(
            logger,
            profile_stages,
            "top.spatial.formations",
            formation_started_at,
            hidden_rows=len(hidden_idx),
        )

        dense_started_at = perf_counter()
        dense_ancc, dense_std, dense_dist = context.impute_dense_ancc(
            xy_hidden, self_well
        )
        dense_known, _, _ = context.impute_dense_ancc(xy_known, self_well)
        dense_b, _, _, dense_b_late, dense_b_weighted = segment_biases(
            known_tvt, z[known_idx], dense_known
        )
        dense_signal = -z[hidden_idx] + dense_ancc + dense_b
        dense_signal_weighted = -z[hidden_idx] + dense_ancc + dense_b_weighted
        dense_signal_late = -z[hidden_idx] + dense_ancc + dense_b_late
        dense_known_residual = known_tvt + z[known_idx] - dense_known
        dense_rmse = (
            float(np.sqrt(np.nanmean(dense_known_residual**2)))
            if np.isfinite(dense_known_residual).any()
            else np.nan
        )
        dense_bias = (
            float(np.nanmean(dense_known_residual))
            if np.isfinite(dense_known_residual).any()
            else np.nan
        )
        dense_nb_std = (
            float(np.nanmean(dense_std)) if np.isfinite(dense_std).any() else np.nan
        )
        features["kg_dense_ancc_tvt"][hidden_idx] = dense_signal
        features["kg_dense_ancc_minus_flat"][hidden_idx] = (
            dense_signal - flat_pred[hidden_idx]
        )
        features["kg_dense_ancc_minus_last"][hidden_idx] = dense_signal - last_tvt
        features["kg_dense_ancc_std"][hidden_idx] = dense_std
        features["kg_dense_ancc_dist"][hidden_idx] = dense_dist
        features["kg_dense_vs_form"][hidden_idx] = dense_signal - form_mean
        features["dense_ancc"] = full_feature(n, hidden_idx, dense_ancc)
        features["dense_std"] = full_feature(n, hidden_idx, dense_std)
        features["dense_dist"] = full_feature(n, hidden_idx, dense_dist)
        features["tvt_dense_d"] = full_feature(n, hidden_idx, dense_signal - last_tvt)
        features["tvt_densew_d"] = full_feature(
            n, hidden_idx, dense_signal_weighted - last_tvt
        )
        features["tvt_dense50_d"] = full_feature(
            n, hidden_idx, dense_signal_late - last_tvt
        )
        features["dense_rmse"] = full_scalar(n, hidden_idx, dense_rmse)
        features["dense_bias"] = full_scalar(n, hidden_idx, dense_bias)
        features["dense_nb_std"] = full_scalar(n, hidden_idx, dense_nb_std)
        if form_ancc_signal is not None:
            features["spatial_vs_dense"] = full_feature(
                n, hidden_idx, form_ancc_signal - dense_signal
            )
            features["beam_vs_spatial"] = full_feature(
                n, hidden_idx, beam_ref - form_ancc_signal
            )
        signal_stack.append(dense_signal)
        log_profile_stage(
            logger,
            profile_stages,
            "top.spatial.dense",
            dense_started_at,
            hidden_rows=len(hidden_idx),
        )
    log_profile_stage(
        logger,
        profile_stages,
        "top.spatial",
        stage_started_at,
        enabled=context is not None,
        hidden_rows=len(hidden_idx),
    )

    stage_started_at = perf_counter()
    if top_cfg.get("particle_enabled", False):
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
            int(top_cfg.get("ancc_particle_count", top_cfg.get("particle_count", 600))),
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
        pf_ancc_signal = pf_ancc
        pf_ancc_std_signal = pf_ancc_std
        features["pf_ancc"] = full_feature(n, hidden_idx, pf_ancc)
        features["pf_ancc_std"] = full_feature(n, hidden_idx, pf_ancc_std)
        features["pf_ancc_delta"] = full_feature(n, hidden_idx, pf_ancc - last_tvt)
        features["pf_z"] = full_feature(n, hidden_idx, pf_z)
        features["pf_z_delta"] = full_feature(n, hidden_idx, pf_z - last_tvt)
        features["pf_vs_z"] = full_feature(n, hidden_idx, pf_ancc - pf_z)
        if dense_signal is not None:
            features["kg_pf_ancc_vs_dense"][hidden_idx] = pf_ancc - dense_signal
            features["pf_vs_dense"] = full_feature(
                n, hidden_idx, pf_ancc - dense_signal
            )
        if form_ancc_signal is not None:
            features["pf_vs_spatial"] = full_feature(
                n, hidden_idx, pf_ancc - form_ancc_signal
            )
        if dtw_signal is not None:
            features["dtw_vs_pf"] = full_feature(
                n, hidden_idx, dtw_signal[hidden_idx] - pf_ancc
            )
    log_profile_stage(
        logger,
        profile_stages,
        "top.particle",
        stage_started_at,
        enabled=bool(top_cfg.get("particle_enabled", False)),
        hidden_rows=len(hidden_idx),
    )

    stage_started_at = perf_counter()
    if pf_ancc_signal is not None:
        signal_stack.append(pf_ancc_signal)
    typewell_gr_at_known = np.interp(known_tvt, tw_tvt, tw_gr)
    cal_a, cal_b = affine_calibration(known_gr, typewell_gr_at_known)
    pfx_rmse = float(np.sqrt(np.nanmean((known_gr - typewell_gr_at_known) ** 2)))
    slope_all = robust_slope(md[known_idx], known_tvt)
    slope_50 = robust_slope(md[known_idx][-50:], known_tvt[-50:])
    slope_z = robust_slope(z[known_idx], known_tvt)
    md_hidden_from_last = md[hidden_idx] - md[last_known_idx]
    slope_baseline_all = last_tvt + slope_all * md_hidden_from_last
    slope_baseline_50 = last_tvt + slope_50 * md_hidden_from_last

    features["cal_a"] = full_scalar(n, hidden_idx, cal_a)
    features["cal_b"] = full_scalar(n, hidden_idx, cal_b)
    features["pfx_rmse"] = full_scalar(n, hidden_idx, pfx_rmse)
    features["known_len"] = full_scalar(n, hidden_idx, float(len(known_idx)))
    features["eval_len"] = full_scalar(n, hidden_idx, float(len(hidden_idx)))
    features["slp_all"] = full_scalar(n, hidden_idx, slope_all)
    features["slp_50"] = full_scalar(n, hidden_idx, slope_50)
    features["slp_z"] = full_scalar(n, hidden_idx, slope_z)
    features["slp_b_d_all"] = full_feature(n, hidden_idx, slope_baseline_all - last_tvt)
    features["slp_b_d_50"] = full_feature(n, hidden_idx, slope_baseline_50 - last_tvt)
    features["ktvt_range"] = full_scalar(
        n, hidden_idx, float(np.nanmax(known_tvt) - np.nanmin(known_tvt))
    )
    features["ktvt_std"] = full_scalar(n, hidden_idx, float(np.nanstd(known_tvt)))
    features["tw_range"] = full_scalar(
        n, hidden_idx, float(np.nanmax(tw_tvt) - np.nanmin(tw_tvt))
    )
    features["tw_gr_mean"] = full_scalar(n, hidden_idx, float(np.nanmean(tw_gr)))
    features["gr_vs_tw_anc"] = full_feature(
        n, hidden_idx, hidden_gr - np.interp(last_tvt, tw_tvt, tw_gr)
    )
    features["gr_vs_slp_all"] = full_feature(
        n,
        hidden_idx,
        hidden_gr - np.interp(slope_baseline_all, tw_tvt, tw_gr),
    )
    add_offset_residuals(
        features,
        "tda",
        n,
        hidden_idx,
        hidden_gr,
        np.full(len(hidden_idx), last_tvt, dtype=float),
        tw_tvt,
        tw_gr,
        ANCH_OFFSETS,
    )
    add_offset_residuals(
        features,
        "tdbc",
        n,
        hidden_idx,
        hidden_gr,
        beam_ref,
        tw_tvt,
        tw_gr,
        BEAM_OFFSETS,
    )
    add_offset_residuals(
        features,
        "tdsc",
        n,
        hidden_idx,
        hidden_gr,
        ncc_ens,
        tw_tvt,
        tw_gr,
        NCC_OFFSETS,
    )
    if pf_ancc_signal is not None:
        add_offset_residuals(
            features,
            "tdpf",
            n,
            hidden_idx,
            hidden_gr,
            pf_ancc_signal,
            tw_tvt,
            tw_gr,
            PF_OFFSETS,
        )
    if dtw_signal is not None:
        add_offset_residuals(
            features,
            "tddtw",
            n,
            hidden_idx,
            hidden_gr,
            dtw_signal[hidden_idx],
            tw_tvt,
            tw_gr,
            DTW_OFFSETS,
        )

    if top_cfg.get("robust_expert_enabled", False):
        add_pf_beam_robust_features(
            features,
            n,
            hidden_idx,
            flat_pred,
            last_tvt,
            pf_ancc_signal,
            pf_ancc_std_signal,
            beam_mean,
            beam_ref,
            beam_matrix,
            dtw_signal[hidden_idx] if dtw_signal is not None else None,
            dwt_hidden_signal,
        )

    signal_matrix = np.vstack(signal_stack).T
    features["sig_std"] = full_feature(n, hidden_idx, np.nanstd(signal_matrix, axis=1))
    features["sig_mean_d"] = full_feature(
        n, hidden_idx, np.nanmean(signal_matrix, axis=1) - last_tvt
    )
    features["kg_signal_mean_tvt"][hidden_idx] = np.nanmean(signal_matrix, axis=1)
    features["kg_signal_mean_minus_flat"][hidden_idx] = (
        np.nanmean(signal_matrix, axis=1) - flat_pred[hidden_idx]
    )
    features["kg_signal_mean_minus_last"][hidden_idx] = (
        np.nanmean(signal_matrix, axis=1) - last_tvt
    )
    features["kg_signal_std"][hidden_idx] = np.nanstd(signal_matrix, axis=1)
    features["kg_hidden_row"][hidden_idx] = 1.0
    log_profile_stage(
        logger,
        profile_stages,
        "top.offsets",
        stage_started_at,
        hidden_rows=len(hidden_idx),
    )
    return features


def _hidden_column(
    base_features: pd.DataFrame,
    column: str,
    hidden_idx: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    if column not in base_features.columns:
        return np.asarray(default, dtype=float)
    values = base_features[column].to_numpy(dtype=float)[hidden_idx]
    if not np.isfinite(values).any():
        return np.asarray(default, dtype=float)
    return values


def build_kaggle_context_signal_features(
    df: pd.DataFrame,
    horizontal_path: Path,
    context: KaggleTopContext,
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    flat_pred: np.ndarray,
    config: dict[str, Any],
    train: bool,
    base_features: pd.DataFrame,
    logger: RunLogger | None = None,
) -> dict[str, np.ndarray | float]:
    """Build only context-dependent top-solution features.

    The heavy alignment/PF block is cached context-free. This function rebuilds
    the spatial/dense ANCC block and aggregate signal columns that change when
    the fold-safe spatial context changes.
    """

    n = len(df)
    features: dict[str, np.ndarray | float] = {}
    top_cfg = config["features"].get("kaggle_top", {})
    profile_stages = bool(config["features"].get("profile_stages", False))

    known = np.isfinite(tvt_input)
    hidden_idx = np.flatnonzero(~known)
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
    known_tvt = tvt_input[known_idx]

    beam_mean = _hidden_column(
        base_features,
        "kg_beam_mean_tvt",
        hidden_idx,
        flat_pred[hidden_idx],
    )
    beam_cons = _hidden_column(base_features, "kg_beam_cons_tvt", hidden_idx, beam_mean)
    beam_sm5 = _hidden_column(base_features, "kg_beam_sm5_tvt", hidden_idx, beam_mean)
    if (
        "kg_beam_cons_tvt" in base_features.columns
        and "kg_beam_sm5_tvt" in base_features.columns
    ):
        beam_ref = (beam_cons + beam_sm5) / 2.0
    else:
        beam_ref = beam_mean

    signal_stack: list[np.ndarray] = []
    beam_signal_stack: list[np.ndarray] = []
    beam_candidates_found = False
    for beam_cfg in top_cfg.get("beam_configs", []):
        _, _, _, _, tag = parse_beam_config(beam_cfg)
        column = f"beam_{tag}_d"
        if column in base_features.columns:
            beam_candidates_found = True
            signal = (
                _hidden_column(
                    base_features,
                    column,
                    hidden_idx,
                    np.zeros(len(hidden_idx), dtype=float),
                )
                + last_tvt
            )
            signal_stack.append(signal)
            beam_signal_stack.append(signal)
    if not signal_stack:
        signal_stack.append(beam_mean)
        beam_signal_stack.append(beam_mean)

    ncc_windows = [int(item) for item in top_cfg.get("ncc_windows", [8, 15, 25])]
    for window in ncc_windows:
        column = f"sc{window}_d"
        if column in base_features.columns:
            signal_stack.append(
                _hidden_column(
                    base_features,
                    column,
                    hidden_idx,
                    np.zeros(len(hidden_idx), dtype=float),
                )
                + last_tvt
            )
    ncc_ens = (
        _hidden_column(
            base_features,
            "sc_ens_d",
            hidden_idx,
            np.zeros(len(hidden_idx), dtype=float),
        )
        + last_tvt
    )
    signal_stack.append(ncc_ens)

    dtw_hidden = None
    if top_cfg.get("dtw_enabled", True) and "kg_dtw_tvt" in base_features.columns:
        dtw_hidden = _hidden_column(
            base_features,
            "kg_dtw_tvt",
            hidden_idx,
            np.full(len(hidden_idx), last_tvt, dtype=float),
        )
        signal_stack.append(dtw_hidden)
    dwt_hidden = None
    if top_cfg.get("dwt_enabled", False) and "kg_dwt_tvt" in base_features.columns:
        dwt_hidden = _hidden_column(
            base_features,
            "kg_dwt_tvt",
            hidden_idx,
            np.full(len(hidden_idx), last_tvt, dtype=float),
        )
        signal_stack.append(dwt_hidden)

    form_ancc_signal = None
    form_mean = flat_pred[hidden_idx]
    dense_signal = None

    stage_started_at = perf_counter()
    formation_started_at = perf_counter()
    xy_hidden = np.column_stack([x[hidden_idx], y[hidden_idx]])
    form_hidden, form_dist = context.impute_formations(xy_hidden, self_well)
    xy_known = np.column_stack([x[known_idx], y[known_idx]])
    form_known, _ = context.impute_formations(xy_known, self_well)
    if form_hidden.shape[1] == len(FORMATIONS) and np.isfinite(form_hidden).any():
        form_signals = []
        form_rmse: dict[str, float] = {}
        for formation_idx, formation in enumerate(FORMATIONS):
            b_full, b_early, b_mid, b_late, b_weighted = segment_biases(
                known_tvt, z[known_idx], form_known[:, formation_idx]
            )
            signal = -z[hidden_idx] + form_hidden[:, formation_idx] + b_full
            signal_weighted = (
                -z[hidden_idx] + form_hidden[:, formation_idx] + b_weighted
            )
            signal_late = -z[hidden_idx] + form_hidden[:, formation_idx] + b_late
            form_signals.append(signal)
            full = np.zeros(n, dtype=float)
            full[hidden_idx] = signal
            features[f"kg_form_{formation}_tvt"] = full
            diff = np.zeros(n, dtype=float)
            diff[hidden_idx] = signal - flat_pred[hidden_idx]
            features[f"kg_form_{formation}_minus_flat"] = diff
            features[f"tvtF_{formation}"] = full_feature(n, hidden_idx, signal)
            features[f"tvtFw_{formation}"] = full_feature(
                n, hidden_idx, signal_weighted
            )
            features[f"tvtF50_{formation}"] = full_feature(n, hidden_idx, signal_late)
            features[f"bw_{formation}"] = full_scalar(n, hidden_idx, b_full)
            features[f"bww_{formation}"] = full_scalar(n, hidden_idx, b_weighted)
            features[f"bw50_{formation}"] = full_scalar(n, hidden_idx, b_late)
            features[f"bw_early_{formation}"] = full_scalar(n, hidden_idx, b_early)
            features[f"bw_mid_{formation}"] = full_scalar(n, hidden_idx, b_mid)
            known_signal = -z[known_idx] + form_known[:, formation_idx] + b_full
            form_rmse[formation] = float(
                np.sqrt(np.nanmean((known_tvt - known_signal) ** 2))
            )
        form_matrix = np.vstack(form_signals).T
        form_mean = np.nanmean(form_matrix, axis=1)
        form_ancc_signal = form_matrix[:, 0]
        features["kg_form_ancc_tvt"] = full_feature(n, hidden_idx, form_ancc_signal)
        features["kg_form_ancc_minus_flat"] = full_feature(
            n, hidden_idx, form_ancc_signal - flat_pred[hidden_idx]
        )
        features["kg_form_ancc_minus_last"] = full_feature(
            n, hidden_idx, form_ancc_signal - last_tvt
        )
        features["kg_form_mean_tvt"] = full_feature(n, hidden_idx, form_mean)
        features["kg_form_mean_minus_flat"] = full_feature(
            n, hidden_idx, form_mean - flat_pred[hidden_idx]
        )
        features["kg_form_mean_minus_last"] = full_feature(
            n, hidden_idx, form_mean - last_tvt
        )
        features["kg_form_std"] = full_feature(
            n, hidden_idx, np.nanstd(form_matrix, axis=1)
        )
        features["kg_form_range"] = full_feature(
            n,
            hidden_idx,
            np.nanmax(form_matrix, axis=1) - np.nanmin(form_matrix, axis=1),
        )
        features["kg_form_knn_dist"] = full_feature(n, hidden_idx, form_dist)
        features["form_mean_d"] = full_feature(n, hidden_idx, form_mean - last_tvt)
        features["form_std_d"] = full_feature(
            n, hidden_idx, np.nanstd(form_matrix, axis=1)
        )
        features["form_rng_d"] = full_feature(
            n,
            hidden_idx,
            np.nanmax(form_matrix, axis=1) - np.nanmin(form_matrix, axis=1),
        )
        features["spatial_ancc_d"] = full_feature(
            n, hidden_idx, form_hidden[:, 0] - np.interp(last_tvt, tw_tvt, tw_gr)
        )
        features["spatial_knn_dist"] = full_feature(n, hidden_idx, form_dist)
        for formation, score in form_rmse.items():
            features[f"frm_rmse_{formation}"] = full_scalar(n, hidden_idx, score)
        signal_stack.append(form_ancc_signal)
    log_profile_stage(
        logger,
        profile_stages,
        "top.spatial.formations",
        formation_started_at,
        hidden_rows=len(hidden_idx),
    )

    dense_started_at = perf_counter()
    dense_ancc, dense_std, dense_dist = context.impute_dense_ancc(xy_hidden, self_well)
    dense_known, _, _ = context.impute_dense_ancc(xy_known, self_well)
    dense_b, _, _, dense_b_late, dense_b_weighted = segment_biases(
        known_tvt, z[known_idx], dense_known
    )
    dense_signal = -z[hidden_idx] + dense_ancc + dense_b
    dense_signal_weighted = -z[hidden_idx] + dense_ancc + dense_b_weighted
    dense_signal_late = -z[hidden_idx] + dense_ancc + dense_b_late
    dense_known_residual = known_tvt + z[known_idx] - dense_known
    dense_rmse = (
        float(np.sqrt(np.nanmean(dense_known_residual**2)))
        if np.isfinite(dense_known_residual).any()
        else np.nan
    )
    dense_bias = (
        float(np.nanmean(dense_known_residual))
        if np.isfinite(dense_known_residual).any()
        else np.nan
    )
    dense_nb_std = (
        float(np.nanmean(dense_std)) if np.isfinite(dense_std).any() else np.nan
    )
    features["kg_dense_ancc_tvt"] = full_feature(n, hidden_idx, dense_signal)
    features["kg_dense_ancc_minus_flat"] = full_feature(
        n, hidden_idx, dense_signal - flat_pred[hidden_idx]
    )
    features["kg_dense_ancc_minus_last"] = full_feature(
        n, hidden_idx, dense_signal - last_tvt
    )
    features["kg_dense_ancc_std"] = full_feature(n, hidden_idx, dense_std)
    features["kg_dense_ancc_dist"] = full_feature(n, hidden_idx, dense_dist)
    features["kg_dense_vs_form"] = full_feature(n, hidden_idx, dense_signal - form_mean)
    features["dense_ancc"] = full_feature(n, hidden_idx, dense_ancc)
    features["dense_std"] = full_feature(n, hidden_idx, dense_std)
    features["dense_dist"] = full_feature(n, hidden_idx, dense_dist)
    features["tvt_dense_d"] = full_feature(n, hidden_idx, dense_signal - last_tvt)
    features["tvt_densew_d"] = full_feature(
        n, hidden_idx, dense_signal_weighted - last_tvt
    )
    features["tvt_dense50_d"] = full_feature(
        n, hidden_idx, dense_signal_late - last_tvt
    )
    features["dense_rmse"] = full_scalar(n, hidden_idx, dense_rmse)
    features["dense_bias"] = full_scalar(n, hidden_idx, dense_bias)
    features["dense_nb_std"] = full_scalar(n, hidden_idx, dense_nb_std)
    if form_ancc_signal is not None:
        features["spatial_vs_dense"] = full_feature(
            n, hidden_idx, form_ancc_signal - dense_signal
        )
        features["beam_vs_spatial"] = full_feature(
            n, hidden_idx, beam_ref - form_ancc_signal
        )
    signal_stack.append(dense_signal)
    log_profile_stage(
        logger,
        profile_stages,
        "top.spatial.dense",
        dense_started_at,
        hidden_rows=len(hidden_idx),
    )
    log_profile_stage(
        logger,
        profile_stages,
        "top.spatial",
        stage_started_at,
        enabled=True,
        hidden_rows=len(hidden_idx),
    )

    pf_ancc_signal = None
    pf_ancc_std_signal = None
    if top_cfg.get("particle_enabled", False) and "pf_ancc" in base_features.columns:
        pf_ancc_signal = _hidden_column(
            base_features,
            "pf_ancc",
            hidden_idx,
            np.full(len(hidden_idx), last_tvt, dtype=float),
        )
        pf_ancc_std_signal = _hidden_column(
            base_features,
            "pf_ancc_std",
            hidden_idx,
            np.full(len(hidden_idx), np.nan, dtype=float),
        )
        features["kg_pf_ancc_vs_dense"] = full_feature(
            n, hidden_idx, pf_ancc_signal - dense_signal
        )
        features["pf_vs_dense"] = full_feature(
            n, hidden_idx, pf_ancc_signal - dense_signal
        )
        if form_ancc_signal is not None:
            features["pf_vs_spatial"] = full_feature(
                n, hidden_idx, pf_ancc_signal - form_ancc_signal
            )
        signal_stack.append(pf_ancc_signal)

    beam_matrix = (
        np.vstack(beam_signal_stack).T
        if beam_signal_stack and beam_candidates_found
        else None
    )
    if top_cfg.get("robust_expert_enabled", False):
        add_pf_beam_robust_features(
            features,
            n,
            hidden_idx,
            flat_pred,
            last_tvt,
            pf_ancc_signal,
            pf_ancc_std_signal,
            beam_mean,
            beam_ref,
            beam_matrix,
            dtw_hidden,
            dwt_hidden,
        )

    signal_matrix = np.vstack(signal_stack).T
    signal_mean = np.nanmean(signal_matrix, axis=1)
    signal_std = np.nanstd(signal_matrix, axis=1)
    features["sig_std"] = full_feature(n, hidden_idx, signal_std)
    features["sig_mean_d"] = full_feature(n, hidden_idx, signal_mean - last_tvt)
    features["kg_signal_mean_tvt"] = full_feature(n, hidden_idx, signal_mean)
    features["kg_signal_mean_minus_flat"] = full_feature(
        n, hidden_idx, signal_mean - flat_pred[hidden_idx]
    )
    features["kg_signal_mean_minus_last"] = full_feature(
        n, hidden_idx, signal_mean - last_tvt
    )
    features["kg_signal_std"] = full_feature(n, hidden_idx, signal_std)
    return features
