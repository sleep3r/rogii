"""
dlmtp/heatmap.py  —  Build 2D GR-alignment heatmap for a hidden-row crop.

Convention
----------
For each hidden row s in the crop, the TVT candidate grid is:
    t_grid[s, j] = prior_tvt[s] + (j - J//2) * bin_ft

Eight channels (N_CHANNELS = 8):
    0  NCC product      h_gr_z[s] * tw_gr_z[s, j]
    1  Abs GR mismatch  |h_gr_z[s] - tw_gr_z[s, j]|
    2  Typewell GR      tw_gr_z[s, j]  (z-scored per typewell)
    3  Horizontal GR    h_gr_z[s]       (z-scored per well, broadcast over j)
    4  Bin offset       (j - J//2) / (J//2)  (broadcast over s)
    5  Valid mask       1 if t_grid in typewell TVT range
    6  Row fraction     s / (crop_len - 1)  (broadcast over j)
    7  TW GR gradient   d(tw_gr_z)/d(tvt) at t_grid[s, j]
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import iqr as scipy_iqr

N_CHANNELS: int = 8
N_CHANNELS_RUN2: int = 9  # +1 greedy-GR prior offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _robust_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score using median / IQR; fallback to mean/std if IQR ~ 0."""
    med = float(np.nanmedian(arr))
    scale = float(scipy_iqr(arr, nan_policy="omit"))
    if scale < 1e-6:
        scale = float(np.nanstd(arr)) + 1e-6
    return (arr - med) / scale


def _smooth_gr(gr: np.ndarray, window: int = 101, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smooth; replaces NaN/inf with interpolation first."""
    gr = gr.astype(np.float64).copy()
    bad = ~np.isfinite(gr)
    if bad.any():
        idx = np.arange(len(gr))
        ok = ~bad
        gr[bad] = np.interp(idx[bad], idx[ok], gr[ok]) if ok.any() else 0.0
    if len(gr) >= window:
        gr = savgol_filter(gr, window_length=window, polyorder=poly)
    return gr


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_heatmap(
    gr_well: np.ndarray,
    z_well: np.ndarray,
    hidden_rows: np.ndarray,
    prior_tvt: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    crop_start: int,
    crop_len: int,
    tvt_bins: int = 128,
    bin_ft: float = 2.0,
    gr_smooth_window: int = 101,
    gr_smooth_poly: int = 3,
    gr_smooth_precomputed: np.ndarray | None = None,
    gr_prior_tvt: np.ndarray | None = None,   # (nh,) greedy-GR TVT, adds channel 8
) -> np.ndarray:
    """Build a (N_CHANNELS[+1], crop_len, tvt_bins) float32 heatmap.

    Parameters
    ----------
    gr_well                : (N,) horizontal GR for the full well
    z_well                 : (N,) TVD for the full well
    hidden_rows            : (nh,) row indices into gr_well / z_well
    prior_tvt              : (nh,) prior TVT for each hidden row (e.g. K3 regression)
    tw_tvt                 : (TW,) typewell TVT (monotone increasing)
    tw_gr                  : (TW,) typewell GR
    crop_start             : start position in [0, nh)
    crop_len               : number of rows in the crop
    tvt_bins               : J — number of TVT candidate bins
    bin_ft                 : ft per bin (corridor half-width = J//2 * bin_ft)
    gr_smooth_precomputed  : (N,) precomputed smoothed GR (skips Savitzky-Golay if provided)
    gr_prior_tvt           : (nh,) greedy-GR predicted TVT; if given, adds channel 8

    Returns
    -------
    heatmap : (N_CHANNELS or N_CHANNELS_RUN2, crop_len, tvt_bins) float32
    """
    nh = len(hidden_rows)
    J = tvt_bins

    # Clamp crop
    s0 = max(0, min(int(crop_start), nh - 1))
    s1 = min(s0 + crop_len, nh)
    actual_len = s1 - s0

    hr_crop = hidden_rows[s0:s1]                          # (actual_len,)
    prior_crop = prior_tvt[s0:s1].astype(np.float64)      # (actual_len,)

    # ---- Horizontal GR (smoothed, z-scored) --------------------------------
    if gr_smooth_precomputed is not None:
        gr_smooth = gr_smooth_precomputed.astype(np.float64)
    else:
        gr_smooth = _smooth_gr(gr_well.astype(np.float64),
                               gr_smooth_window, gr_smooth_poly)
    h_gr_raw = gr_smooth[hr_crop]                          # (actual_len,)
    h_gr_z   = _robust_zscore(h_gr_raw).astype(np.float64)

    # ---- Typewell GR (z-scored) + gradient ---------------------------------
    tw_gr_z_full  = _robust_zscore(tw_gr.astype(np.float64))
    tw_tvt_f      = tw_tvt.astype(np.float64)
    tw_gr_grad    = np.gradient(tw_gr_z_full,
                                np.maximum(tw_tvt_f, tw_tvt_f + 1e-6))
    tw_min, tw_max = tw_tvt_f.min(), tw_tvt_f.max()

    # ---- TVT candidate grid (actual_len × J) --------------------------------
    j_off  = (np.arange(J, dtype=np.float64) - J // 2) * bin_ft   # (J,)
    t_grid = prior_crop[:, None] + j_off[None, :]                  # (actual_len, J)

    # ---- Interpolate typewell at grid points --------------------------------
    flat  = t_grid.ravel()
    tw_at_grid   = np.interp(flat, tw_tvt_f, tw_gr_z_full ).reshape(actual_len, J)
    grad_at_grid = np.interp(flat, tw_tvt_f, tw_gr_grad   ).reshape(actual_len, J)

    # ---- Build channels ----------------------------------------------------
    valid = ((t_grid >= tw_min) & (t_grid <= tw_max)).astype(np.float32)

    h_z_bc    = h_gr_z[:, None] * np.ones((1, J), dtype=np.float64)
    bin_norm  = ((np.arange(J) - J // 2) / max(J // 2, 1)).astype(np.float32)
    bin_bc    = np.ones((actual_len, 1), dtype=np.float32) * bin_norm[None, :]
    row_frac  = (np.arange(actual_len, dtype=np.float32)
                 / max(actual_len - 1, 1))[:, None] * np.ones((1, J), dtype=np.float32)

    heatmap = np.stack([
        (h_z_bc * tw_at_grid   ).astype(np.float32),   # 0 NCC product
        np.abs(h_z_bc - tw_at_grid).astype(np.float32), # 1 abs diff
        tw_at_grid.astype(np.float32),                  # 2 tw GR z
        h_z_bc.astype(np.float32),                      # 3 h GR z
        bin_bc,                                          # 4 bin norm
        valid,                                           # 5 valid mask
        row_frac,                                        # 6 row frac
        grad_at_grid.astype(np.float32),                # 7 tw GR grad
    ], axis=0)  # (8, actual_len, J)

    # Optional channel 8: greedy-GR prior offset (scalar per row, broadcast over J)
    if gr_prior_tvt is not None:
        gr_crop = gr_prior_tvt[s0:s1].astype(np.float64)    # (actual_len,)
        gr_off  = ((gr_crop - prior_crop) / bin_ft            # bins from K3 center
                   ).astype(np.float32)[:, None]              # (actual_len, 1)
        gr_ch   = gr_off * np.ones((1, J), dtype=np.float32) # (actual_len, J)
        heatmap = np.concatenate([heatmap, gr_ch[None]], axis=0)  # (9, actual_len, J)

    # Pad if crop was shorter than requested (end of well)
    if actual_len < crop_len:
        n_ch = heatmap.shape[0]
        pad  = np.zeros((n_ch, crop_len - actual_len, J), dtype=np.float32)
        heatmap = np.concatenate([heatmap, pad], axis=1)

    return heatmap
