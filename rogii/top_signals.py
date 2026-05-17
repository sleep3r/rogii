from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FORMATIONS
from .io import typewell_path, well_name
from .numeric import as_float_array, fill_numeric, nearest_index, smooth_for_alignment
from .spatial import KaggleTopContext


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
        features[f"kg_beam_{tag}_minus_flat"][hidden_idx] = (
            signal - flat_pred[hidden_idx]
        )
        features[f"kg_beam_{tag}_minus_last"] = np.zeros(n, dtype=float)
        features[f"kg_beam_{tag}_minus_last"][hidden_idx] = signal - last_tvt

    if beam_signals:
        beam_matrix = np.vstack(beam_signals).T
        beam_mean = np.nanmean(beam_matrix, axis=1)
        features["kg_beam_mean_minus_flat"][hidden_idx] = (
            beam_mean - flat_pred[hidden_idx]
        )
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
        features["kg_ncc_mean_minus_flat"][hidden_idx] = (
            np.nanmean(ncc_matrix, axis=1) - flat_pred[hidden_idx]
        )
    if ncc_scores:
        features["kg_ncc_score_mean"][hidden_idx] = np.nanmean(
            np.vstack(ncc_scores).T, axis=1
        )

    if top_cfg.get("dtw_enabled", True):
        dtw_signal = lowres_dtw_signal(
            gr_full,
            tw_tvt,
            tw_gr,
            int(top_cfg.get("dtw_max_query_points", 700)),
            int(top_cfg.get("dtw_max_ref_points", 700)),
            int(top_cfg.get("dtw_radius", 35)),
        )
        features["kg_dtw_minus_flat"][hidden_idx] = (
            dtw_signal[hidden_idx] - flat_pred[hidden_idx]
        )
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
                b = (
                    float(np.nanmedian(residual_base))
                    if np.isfinite(residual_base).any()
                    else 0.0
                )
                signal = -z[hidden_idx] + form_hidden[:, formation_idx] + b
                form_signals.append(signal)
                col = f"kg_form_{formation}_minus_flat"
                features[col] = np.zeros(n, dtype=float)
                features[col][hidden_idx] = signal - flat_pred[hidden_idx]
            form_matrix = np.vstack(form_signals).T
            form_mean = np.nanmean(form_matrix, axis=1)
            features["kg_form_ancc_minus_flat"][hidden_idx] = (
                form_matrix[:, 0] - flat_pred[hidden_idx]
            )
            features["kg_form_mean_minus_flat"][hidden_idx] = (
                form_mean - flat_pred[hidden_idx]
            )
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
        features["kg_dense_ancc_minus_flat"][hidden_idx] = (
            dense_signal - flat_pred[hidden_idx]
        )
        features["kg_dense_ancc_std"][hidden_idx] = dense_std
        features["kg_dense_ancc_dist"][hidden_idx] = dense_dist
        features["kg_dense_vs_form"][hidden_idx] = dense_signal - form_mean
        signal_stack.append(dense_signal)

    signal_matrix = np.vstack(signal_stack).T
    features["kg_signal_mean_minus_flat"][hidden_idx] = (
        np.nanmean(signal_matrix, axis=1) - flat_pred[hidden_idx]
    )
    features["kg_signal_std"][hidden_idx] = np.nanstd(signal_matrix, axis=1)
    features["kg_hidden_row"][hidden_idx] = 1.0
    return features
