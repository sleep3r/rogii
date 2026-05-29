"""GR log feature extraction: filling, smoothing, derivatives, masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

_EPS = 1e-8


def _fill_gr(gr: np.ndarray, method: str = "linear") -> np.ndarray:
    """Fill NaN values in GR trace.

    'linear'  — linear interpolation between valid neighbours, edge fill.
    'forward' — forward fill then backward fill.
    """
    gr = gr.copy()
    valid = np.isfinite(gr)
    if valid.sum() == 0:
        return np.zeros_like(gr)
    if method == "forward":
        # Forward fill
        idx = np.arange(len(gr))
        last_valid = np.where(valid, idx, -1)
        for i in range(1, len(gr)):
            if last_valid[i] < 0:
                last_valid[i] = last_valid[i - 1]
        fwd = np.where(last_valid >= 0, gr[np.maximum(last_valid, 0)], np.nan)
        # Backward fill remaining NaNs
        bwd_idx = np.where(valid, idx, len(gr))
        for i in range(len(gr) - 2, -1, -1):
            if bwd_idx[i] == len(gr):
                bwd_idx[i] = bwd_idx[i + 1]
        bwd = np.where(bwd_idx < len(gr), gr[np.minimum(bwd_idx, len(gr) - 1)], np.nan)
        filled = np.where(np.isfinite(fwd), fwd, bwd)
        filled = np.where(np.isfinite(filled), filled, gr[valid].mean())
    else:
        # Linear interpolation
        x = np.arange(len(gr))
        filled = np.interp(x, x[valid], gr[valid])
    return filled.astype(np.float32)


def compute_gr_features(
    df: pd.DataFrame | np.ndarray,
    smooth_sigmas: list[float] = (5.0, 15.0, 50.0, 200.0),
    fill_method: str = "linear",
) -> dict[str, np.ndarray]:
    """
    Compute GR-based features.

    Returns dict of 1-D float32 arrays.
    """
    if isinstance(df, pd.DataFrame):
        gr_raw = df["GR"].to_numpy(dtype=np.float64)
    else:
        gr_raw = np.asarray(df, dtype=np.float64)
    valid_mask = np.isfinite(gr_raw).astype(np.float32)

    # Fill NaNs
    gr_filled = _fill_gr(gr_raw, method=fill_method)

    # Distance to nearest valid GR (in rows)
    n = len(gr_raw)
    valid_idx = np.where(np.isfinite(gr_raw))[0]
    if len(valid_idx) == 0:
        dist_prev = np.full(n, n, dtype=np.float32)
        dist_next = np.full(n, n, dtype=np.float32)
    else:
        # Distance to previous valid
        dist_prev = np.full(n, n, dtype=np.float32)
        last = -n
        for i in range(n):
            if np.isfinite(gr_raw[i]):
                last = i
            dist_prev[i] = i - last

        # Distance to next valid
        dist_next = np.full(n, n, dtype=np.float32)
        nxt = 2 * n
        for i in range(n - 1, -1, -1):
            if np.isfinite(gr_raw[i]):
                nxt = i
            dist_next[i] = nxt - i

    # Normalise distances
    dist_prev = np.clip(dist_prev / 500.0, 0.0, 1.0).astype(np.float32)
    dist_next = np.clip(dist_next / 500.0, 0.0, 1.0).astype(np.float32)

    # Smoothed GR at multiple scales
    smoothed = {}
    for sigma in smooth_sigmas:
        key = f"gr_smooth_{int(sigma)}"
        smoothed[key] = gaussian_filter1d(gr_filled, sigma=sigma).astype(np.float32)

    # Derivatives of smoothed GR (gradient)
    derivatives = {}
    for sigma in smooth_sigmas[:3]:  # derivatives at 3 coarser scales
        sg = smoothed[f"gr_smooth_{int(sigma)}"]
        dg = np.gradient(sg).astype(np.float32)
        derivatives[f"gr_deriv_{int(sigma)}"] = dg
        d2g = np.gradient(dg).astype(np.float32)
        derivatives[f"gr_deriv2_{int(sigma)}"] = d2g

    # GR z-score (global)
    gr_mean = gr_filled.mean()
    gr_std = gr_filled.std() + _EPS
    gr_zscore = ((gr_filled - gr_mean) / gr_std).astype(np.float32)

    feats: dict[str, np.ndarray] = {
        "gr_filled": gr_filled,
        "gr_valid": valid_mask,
        "gr_dist_prev": dist_prev,
        "gr_dist_next": dist_next,
        "gr_zscore": gr_zscore,
        "gr_mean": np.full(n, float(gr_mean), dtype=np.float32),
        "gr_std": np.full(n, float(gr_std), dtype=np.float32),
    }
    feats.update(smoothed)
    feats.update(derivatives)
    return feats
