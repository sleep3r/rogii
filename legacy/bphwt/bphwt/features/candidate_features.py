"""Assemble all per-row features into a single float32 matrix.

Feature vector layout (canonical order, used by the model):

Group 0 — position / depth (4)
    row_idx_norm, md_norm, cum_md_norm, z_norm

Group 1 — relative coordinates (3)
    traj_rel_x_norm, traj_rel_y_norm, traj_rel_z_norm

Group 2 — trajectory direction (7)
    traj_tx, traj_ty, traj_tz
    traj_inclination_norm, traj_azimuth_sin, traj_azimuth_cos
    traj_dogleg_norm

Group 3 — anchor / TVT prior (7)
    tvt_input_norm, known_mask
    last_known_tvt_norm, next_known_tvt_norm
    linear_tvt_norm
    dist_prev_known_norm, dist_next_known_norm

Group 4 — GR features (varies with n_sigma, default 4 sigmas → ~20)
    gr_valid, gr_filled_norm, gr_zscore
    gr_smooth_{5,15,50,200}_norm
    gr_deriv_{5,15,50}
    gr_deriv2_{5,15,50}
    gr_dist_prev, gr_dist_next

Group 5 — HMM prior (5)
    hmm_tvt_mu_norm, hmm_tvt_std_norm
    hmm_viterbi_norm, hmm_velocity
    hmm_entropy

Group 6 — typewell offset samples (len(offsets))
    tw_gr_at_offset_{k} for each offset

Group 7 — formation context (via TopContext, optional)
    per-formation depth + z_minus_top features

Total (default config): ~64 channels
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-8


def build_feature_matrix(
    df: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    hmm_feats: dict[str, np.ndarray],
    traj_feats: dict[str, np.ndarray],
    gr_feats: dict[str, np.ndarray],
    norm_stats: dict[str, tuple],  # key -> (mean, std)
    tw_offsets: list[float],
    dtw_feats: dict[str, np.ndarray] | None = None,
    neighbor_feats: dict[str, np.ndarray] | None = None,
    top_feats: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Assemble all features into a [n_rows, C] float32 matrix.

    norm_stats: dict computed from train statistics, e.g.
        {'TVT': (mean, std), 'MD': (mean, std), ...}
    """
    n = len(df)

    def _norm(v: np.ndarray, key: str) -> np.ndarray:
        if key in norm_stats:
            mu, sigma = norm_stats[key]
            return ((v - mu) / (sigma + _EPS)).astype(np.float32)
        return v.astype(np.float32)

    md = df["MD"].to_numpy(dtype=np.float32)
    z = df["Z"].to_numpy(dtype=np.float32)
    tvt_input = df["TVT_input"].to_numpy(dtype=np.float32)
    known_mask = np.isfinite(tvt_input).astype(np.float32)

    # ---- Anchor / TVT prior features ---------------------------------
    # Fill TVT_input with forward-fill + backward-fill for last/next known
    last_known = _forward_fill(tvt_input)
    next_known = _backward_fill(tvt_input)
    linear_tvt = _linear_interp_tvt(tvt_input)
    head_slope_tvt = _constant_slope_from_head(md, tvt_input)
    head_tail_slope_tvt = _constant_slope_head_tail(md, tvt_input)

    # Distances to known boundaries (in rows, normalised)
    dist_prev = _dist_to_prev_known(known_mask)
    dist_next = _dist_to_next_known(known_mask)

    # Row index (normalised within well)
    row_idx = np.arange(n, dtype=np.float32) / max(n - 1, 1)

    # ---- Typewell offset samples -------------------------------------
    tw_offset_cols = []
    tvt_base = hmm_feats.get("hmm_tvt_mu", linear_tvt)
    for offset in tw_offsets:
        tvt_query = tvt_base + offset
        gr_at_offset = np.interp(tvt_query, tw_tvt, tw_gr).astype(np.float32)
        # normalise with global tw_gr stats
        tw_gr_mean = tw_gr.mean()
        tw_gr_std = tw_gr.std() + _EPS
        gr_at_offset_norm = ((gr_at_offset - tw_gr_mean) / tw_gr_std).astype(np.float32)
        tw_offset_cols.append(gr_at_offset_norm)

    # ---- Assemble ----------------------------------------------------
    cols: list[np.ndarray] = []

    # Group 0: position
    cols.append(row_idx)
    cols.append(_norm(md, "MD"))
    cols.append(_norm(traj_feats["traj_cum_md"], "cum_md"))
    cols.append(_norm(z, "Z"))

    # Group 1: relative coords
    cols.append(_norm(traj_feats["traj_rel_x"], "rel_x"))
    cols.append(_norm(traj_feats["traj_rel_y"], "rel_y"))
    cols.append(_norm(traj_feats["traj_rel_z"], "rel_z"))

    # Group 2: trajectory direction
    cols.append(traj_feats["traj_tx"])
    cols.append(traj_feats["traj_ty"])
    cols.append(traj_feats["traj_tz"])
    cols.append(_norm(traj_feats["traj_inclination"], "inclination"))
    cols.append(traj_feats["traj_azimuth_sin"])
    cols.append(traj_feats["traj_azimuth_cos"])
    cols.append(np.clip(_norm(traj_feats["traj_dogleg"], "dogleg"), -5, 5))

    # Group 3: anchor / TVT prior
    # Fill nan tvt_input with last known for input feature
    tvt_input_filled = last_known.copy()
    cols.append(_norm(tvt_input_filled, "TVT"))
    cols.append(known_mask)
    cols.append(_norm(last_known, "TVT"))
    cols.append(_norm(next_known, "TVT"))
    cols.append(_norm(linear_tvt, "TVT"))
    cols.append(_norm(head_slope_tvt, "TVT"))
    cols.append(_norm(head_tail_slope_tvt, "TVT"))
    cols.append(np.clip(dist_prev / 500.0, 0.0, 1.0))
    cols.append(np.clip(dist_next / 500.0, 0.0, 1.0))

    # Group 4: GR features
    cols.append(gr_feats["gr_valid"])
    gr_mu, gr_sigma = norm_stats.get("GR", (gr_feats["gr_filled"].mean(), gr_feats["gr_filled"].std() + _EPS))
    cols.append(((gr_feats["gr_filled"] - gr_mu) / (gr_sigma + _EPS)).astype(np.float32))
    cols.append(np.clip(gr_feats["gr_zscore"], -4, 4))
    for sigma_key in ["gr_smooth_5", "gr_smooth_15", "gr_smooth_50", "gr_smooth_200"]:
        if sigma_key in gr_feats:
            v = gr_feats[sigma_key]
            cols.append(((v - gr_mu) / (gr_sigma + _EPS)).astype(np.float32))
    for deriv_key in ["gr_deriv_5", "gr_deriv_15", "gr_deriv_50"]:
        if deriv_key in gr_feats:
            cols.append(np.clip(gr_feats[deriv_key] / (gr_sigma + _EPS), -5, 5).astype(np.float32))
    cols.append(gr_feats["gr_dist_prev"])
    cols.append(gr_feats["gr_dist_next"])

    # Group 5: HMM prior
    cols.append(_norm(hmm_feats.get("hmm_tvt_mu", linear_tvt), "TVT"))
    cols.append(np.clip(hmm_feats.get("hmm_tvt_std", np.ones(n, np.float32)) / 20.0, 0, 1))
    cols.append(_norm(hmm_feats.get("hmm_viterbi", linear_tvt), "TVT"))
    cols.append(np.clip(hmm_feats.get("hmm_velocity", np.zeros(n, np.float32)), -0.15, 0.15) / 0.15)
    cols.append(np.clip(hmm_feats.get("hmm_entropy", np.zeros(n, np.float32)) / 5.0, 0, 1))
    cols.append(np.clip(hmm_feats.get("hmm_gr_mismatch", np.zeros(n, np.float32)) / 50.0, 0, 5))

    # Group 5b: segment DTW prior
    if dtw_feats is not None:
        cols.append(_norm(dtw_feats.get("dtw_tvt", linear_tvt), "TVT"))
        cols.append(np.clip(dtw_feats.get("dtw_score", np.full(n, 999.0, np.float32)) / 20.0, 0, 5))
        cols.append(np.clip(dtw_feats.get("dtw_orientation", np.zeros(n, np.float32)), -1, 1))

    # Group 5c: neighbor/template prior
    if neighbor_feats is not None:
        cols.append(_norm(neighbor_feats.get("neighbor_tvt", linear_tvt), "TVT"))
        cols.append(np.clip(neighbor_feats.get("neighbor_std", np.full(n, 20.0, np.float32)) / 40.0, 0, 5))
        cols.append(np.clip(neighbor_feats.get("neighbor_corr", np.zeros(n, np.float32)), -1, 1))
        cols.append(
            np.clip(neighbor_feats.get("neighbor_count_same_typewell", np.zeros(n, np.float32)) / 8.0, 0, 1)
        )

    # Group 6: typewell offset samples
    cols.extend(tw_offset_cols)

    # Group 7: formation context (optional)
    if top_feats is not None:
        from bphwt.features.top_context import FORMATION_COLS

        for col in FORMATION_COLS:
            if col in top_feats:
                cols.append(_norm(top_feats[col], "Z"))  # formations are Z-like values
            if f"z_minus_{col.lower()}" in top_feats:
                cols.append(np.clip(top_feats[f"z_minus_{col.lower()}"] / 200.0, -3, 3))
            if f"{col}_uncertainty" in top_feats:
                cols.append(np.clip(top_feats[f"{col}_uncertainty"] / 200.0, 0, 5))
        for key, arr in top_feats.items():
            if key.startswith("interval_"):
                cols.append(np.clip(arr / 200.0, 0, 5))

    X = np.stack(cols, axis=1).astype(np.float32)  # [n, C]
    return X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forward_fill(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    if not np.all(np.isfinite(out)):
        # Back-fill remaining NaN at head
        last = np.nan
        for i in range(len(out) - 1, -1, -1):
            if np.isfinite(out[i]):
                last = out[i]
            elif np.isfinite(last):
                out[i] = last
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float32)


def _backward_fill(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    nxt = np.nan
    for i in range(len(out) - 1, -1, -1):
        if np.isfinite(out[i]):
            nxt = out[i]
        elif np.isfinite(nxt):
            out[i] = nxt
    if not np.all(np.isfinite(out)):
        # forward-fill remaining
        out = _forward_fill(out)
    return out.astype(np.float32)


def _linear_interp_tvt(tvt_input: np.ndarray) -> np.ndarray:
    """Linear interpolation between known TVT_input anchors."""
    n = len(tvt_input)
    known_idx = np.where(np.isfinite(tvt_input))[0]
    if len(known_idx) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.interp(np.arange(n), known_idx, tvt_input[known_idx]).astype(np.float32)


def _constant_slope_from_head(md: np.ndarray, tvt_input: np.ndarray) -> np.ndarray:
    n = len(tvt_input)
    known_idx = np.where(np.isfinite(tvt_input))[0]
    if len(known_idx) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(known_idx) == 1:
        return np.full(n, tvt_input[known_idx[0]], dtype=np.float32)
    head = known_idx[: min(len(known_idx), 16)]
    denom = md[head[-1]] - md[head[0]]
    if abs(float(denom)) < _EPS:
        slope = 0.0
    else:
        slope = float((tvt_input[head[-1]] - tvt_input[head[0]]) / denom)
    return (tvt_input[known_idx[0]] + slope * (md - md[known_idx[0]])).astype(np.float32)


def _constant_slope_head_tail(md: np.ndarray, tvt_input: np.ndarray) -> np.ndarray:
    known_idx = np.where(np.isfinite(tvt_input))[0]
    if len(known_idx) < 2:
        return _constant_slope_from_head(md, tvt_input)
    denom = md[known_idx[-1]] - md[known_idx[0]]
    if abs(float(denom)) < _EPS:
        slope = 0.0
    else:
        slope = float((tvt_input[known_idx[-1]] - tvt_input[known_idx[0]]) / denom)
    return (tvt_input[known_idx[0]] + slope * (md - md[known_idx[0]])).astype(np.float32)


def _dist_to_prev_known(known_mask: np.ndarray) -> np.ndarray:
    n = len(known_mask)
    d = np.full(n, n, dtype=np.float32)
    last = -n
    for i in range(n):
        if known_mask[i] > 0.5:
            last = i
        d[i] = i - last
    return d


def _dist_to_next_known(known_mask: np.ndarray) -> np.ndarray:
    n = len(known_mask)
    d = np.full(n, n, dtype=np.float32)
    nxt = 2 * n
    for i in range(n - 1, -1, -1):
        if known_mask[i] > 0.5:
            nxt = i
        d[i] = nxt - i
    return d
