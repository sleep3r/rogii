"""Feature engineering for the offset-state decoder models.

Feature philosophy:
- ALL features must be test-safe (computable from known prefix only)
- No future information from hidden rows (except geometry: MD/X/Y/Z/GR)
- GR, X, Y, Z at hidden rows ARE allowed — they are measured, not predicted

Well-level features (one row per well) — used for global offset classifier:
    - C-field drift statistics: c0, std, slope, multi-window medians
    - Anchor geometry: Z_anchor, MD_anchor, anchor_C
    - Trajectory geometry: total dip change, azimuth change, MD/Z/XY ranges
    - GR statistics from known prefix
    - Hidden segment geometry summary

Per-segment features (one row per segment) — used for K-segment classifier:
    - All well-level features
    - Segment position (normalized: 0=first, 1=last)
    - Segment mean Z, dZ, MD delta
    - GR at segment center/start/end

Usage:
    feats = make_well_features(samples)            # shape (N_wells, n_features)
    feats = make_segment_features(samples, K=3)    # shape (N_wells*K, n_features)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature name lists (for reproducibility / feature importance)
# ---------------------------------------------------------------------------

WELL_FEATURE_NAMES: list[str] = []   # populated below
SEG_EXTRA_FEATURE_NAMES: list[str] = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_stat(arr: np.ndarray, fn) -> float:
    """Apply fn to arr, return 0.0 on error or empty."""
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return 0.0
    try:
        return float(fn(v))
    except Exception:
        return 0.0


def _slope(x: np.ndarray) -> float:
    """Linear slope of a 1-D array via polyfit (robust)."""
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=np.float64)
    try:
        return float(np.polyfit(t, x, 1)[0])
    except Exception:
        return 0.0


def _percentile(arr: np.ndarray, q: float) -> float:
    v = arr[np.isfinite(arr)]
    return float(np.percentile(v, q)) if len(v) > 0 else 0.0


# ---------------------------------------------------------------------------
# Well-level features
# ---------------------------------------------------------------------------

def make_well_feature_dict(sample) -> dict[str, float]:
    """Extract all well-level features from an OffsetSample into a flat dict."""
    z = sample.z.astype(np.float64)
    md = sample.md.astype(np.float64)
    gr = sample.gr.astype(np.float64)
    tvt_input = sample.tvt_input.astype(np.float64)
    hidden_rows = sample.hidden_rows
    anchor_row = sample.anchor_row
    anchor_tvt = float(sample.anchor_tvt)
    anchor_z = float(sample.anchor_z)
    n_hidden = sample.n_hidden
    n_known = int(np.sum(np.isfinite(tvt_input)))

    # Known row indices
    known_idx = np.flatnonzero(np.isfinite(tvt_input))

    # --- C-field drift statistics (test-safe) ---
    feats: dict[str, float] = {
        "c0":                   sample.c0,
        "dC_last64_median":     sample.dC_last64_median,
        "dC_last128_median":    sample.dC_last128_median,
        "dC_last256_median":    sample.dC_last256_median,
        "dC_last512_median":    sample.dC_last512_median,
        "dC_last512_mean":      sample.dC_last512_mean,
        "dC_last512_std":       sample.dC_last512_std,
        "dC_last512_slope":     sample.dC_last512_slope,
        "anchor_C":             sample.anchor_C,
    }

    # --- Multi-window c0 differences (trend indicators) ---
    feats["dC_64_vs_512"]  = sample.dC_last64_median  - sample.dC_last512_median
    feats["dC_128_vs_512"] = sample.dC_last128_median - sample.dC_last512_median
    feats["dC_256_vs_512"] = sample.dC_last256_median - sample.dC_last512_median

    # --- Geometry: anchor and well totals ---
    feats["anchor_z"]   = anchor_z
    feats["anchor_tvt"] = anchor_tvt
    feats["anchor_md"]  = float(md[anchor_row]) if np.isfinite(md[anchor_row]) else 0.0

    z_range = float(np.nanmax(z[:anchor_row + 1]) - np.nanmin(z[:anchor_row + 1]))
    md_range = float(np.nanmax(md[:anchor_row + 1]) - np.nanmin(md[:anchor_row + 1]))
    feats["z_range_known"]  = z_range
    feats["md_range_known"] = md_range

    x = sample.x.astype(np.float64)
    y = sample.y.astype(np.float64)
    feats["anchor_x"] = float(x[anchor_row]) if np.isfinite(x[anchor_row]) else 0.0
    feats["anchor_y"] = float(y[anchor_row]) if np.isfinite(y[anchor_row]) else 0.0

    # Horizontal displacement (well reach)
    if len(known_idx) >= 2:
        dx = x[anchor_row] - x[known_idx[0]]
        dy = y[anchor_row] - y[known_idx[0]]
        feats["lateral_reach"] = float(np.sqrt(dx**2 + dy**2))
    else:
        feats["lateral_reach"] = 0.0

    # --- Trajectory: dip and build rates (over last known 64 rows) ---
    win_start = max(0, anchor_row - 64)
    known_win = known_idx[known_idx >= win_start]
    if len(known_win) >= 2:
        dz_win = np.diff(z[known_win])
        dmd_win = np.diff(md[known_win])
        with np.errstate(divide="ignore", invalid="ignore"):
            dip_win = np.where(np.abs(dmd_win) > 0, dz_win / dmd_win, 0.0)
        feats["dip_last64_mean"]   = _safe_stat(dip_win, np.mean)
        feats["dip_last64_std"]    = _safe_stat(dip_win, np.std)
        feats["dip_last64_slope"]  = _slope(dip_win)
    else:
        feats["dip_last64_mean"]   = 0.0
        feats["dip_last64_std"]    = 0.0
        feats["dip_last64_slope"]  = 0.0

    # --- GR statistics from known prefix (last 256 rows) ---
    win256_start = max(0, anchor_row - 256)
    known_win256 = known_idx[known_idx >= win256_start]
    if len(known_win256) > 0:
        gr_win = gr[known_win256]
        feats["gr_mean"]   = _safe_stat(gr_win, np.mean)
        feats["gr_std"]    = _safe_stat(gr_win, np.std)
        feats["gr_p10"]    = _percentile(gr_win, 10)
        feats["gr_p50"]    = _percentile(gr_win, 50)
        feats["gr_p90"]    = _percentile(gr_win, 90)
        feats["gr_slope"]  = _slope(gr_win)
    else:
        feats["gr_mean"]  = 0.0
        feats["gr_std"]   = 0.0
        feats["gr_p10"]   = 0.0
        feats["gr_p50"]   = 0.0
        feats["gr_p90"]   = 0.0
        feats["gr_slope"] = 0.0

    # --- GR at anchor row ---
    feats["gr_anchor"] = float(gr[anchor_row]) if np.isfinite(gr[anchor_row]) else feats["gr_mean"]

    # --- Hidden segment geometry summary ---
    feats["n_hidden"]   = float(n_hidden)
    feats["n_known"]    = float(n_known)
    feats["frac_hidden"] = float(n_hidden) / max(1.0, float(n_hidden + n_known))

    if n_hidden > 0:
        h_z    = z[hidden_rows]
        h_md   = md[hidden_rows]
        h_gr   = gr[hidden_rows]

        feats["hidden_dz_total"]  = float(h_z[-1] - anchor_z)
        feats["hidden_dmd_total"] = float(h_md[-1] - md[anchor_row]) if np.isfinite(md[anchor_row]) else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            dip_hidden = float(feats["hidden_dz_total"] / max(abs(feats["hidden_dmd_total"]), 1e-6))
        feats["hidden_dip_approx"] = dip_hidden

        feats["hidden_z_mean"]  = _safe_stat(h_z, np.mean)
        feats["hidden_z_std"]   = _safe_stat(h_z, np.std)
        feats["hidden_gr_mean"] = _safe_stat(h_gr, np.mean)
        feats["hidden_gr_std"]  = _safe_stat(h_gr, np.std)
        feats["hidden_gr_p50"]  = _percentile(h_gr, 50)

        # Rate of Z change per row in hidden segment
        feats["hidden_dz_per_row"] = feats["hidden_dz_total"] / max(n_hidden, 1)
    else:
        for k in ["hidden_dz_total", "hidden_dmd_total", "hidden_dip_approx",
                  "hidden_z_mean", "hidden_z_std", "hidden_gr_mean",
                  "hidden_gr_std", "hidden_gr_p50", "hidden_dz_per_row"]:
            feats[k] = 0.0

    # --- Ratio: C-drift std relative to c0 ---
    feats["c0_snr"] = abs(feats["c0"]) / (feats["dC_last512_std"] + 1e-6)

    # --- Last known C-value relative to anchor_C ---
    if n_known >= 2:
        C_vals = (tvt_input[known_idx] + z[known_idx])
        dC_vals = np.diff(C_vals[np.isfinite(C_vals)])
        feats["dC_last1"]  = float(dC_vals[-1]) if len(dC_vals) >= 1 else 0.0
        feats["dC_last3"]  = _safe_stat(dC_vals[-3:], np.mean) if len(dC_vals) >= 1 else 0.0
        feats["dC_last10"] = _safe_stat(dC_vals[-10:], np.mean) if len(dC_vals) >= 1 else 0.0
    else:
        feats["dC_last1"]  = 0.0
        feats["dC_last3"]  = 0.0
        feats["dC_last10"] = 0.0

    # --- Short-window dC stats (added for residual learning) ---
    if n_known >= 2:
        C_vals = (tvt_input[known_idx] + z[known_idx])
        dC_vals = np.diff(C_vals[np.isfinite(C_vals)])

        # Short-window medians (16, 32 rows)
        for w in (16, 32):
            w_vals = dC_vals[-w:] if len(dC_vals) >= w else dC_vals
            feats[f"dC_last{w}_median"] = _safe_stat(w_vals, np.median)
            feats[f"dC_last{w}_std"]    = _safe_stat(w_vals, np.std)

        # Short-window vs long-window difference (drift acceleration)
        feats["dC_16_vs_512"]  = feats["dC_last16_median"] - feats["dC_last512_median"]
        feats["dC_32_vs_512"]  = feats["dC_last32_median"] - feats["dC_last512_median"]
        feats["dC_16_vs_128"]  = feats["dC_last16_median"] - feats["dC_last128_median"]

        # Slope of short-window dC (drift 2nd derivative)
        feats["dC_slope_last32"]  = _slope(dC_vals[-32:])  if len(dC_vals) >= 4 else 0.0
        feats["dC_slope_last128"] = _slope(dC_vals[-128:]) if len(dC_vals) >= 8 else 0.0

        # Lag-1 autocorrelation of recent dC (persistence / mean-reversion)
        w_ac = min(128, len(dC_vals))
        if w_ac >= 4:
            ac_vals = dC_vals[-w_ac:]
            ac_vals_d = ac_vals - ac_vals.mean()
            denom = np.dot(ac_vals_d, ac_vals_d)
            if denom > 0:
                feats["dC_autocorr1"] = float(np.dot(ac_vals_d[:-1], ac_vals_d[1:]) / denom)
            else:
                feats["dC_autocorr1"] = 0.0
        else:
            feats["dC_autocorr1"] = 0.0

        # IQR of dC in last 128 rows (drift variability / regime stability)
        w128 = dC_vals[-128:] if len(dC_vals) >= 8 else dC_vals
        feats["dC_iqr_last128"] = float(np.percentile(w128, 75) - np.percentile(w128, 25)) if len(w128) >= 4 else 0.0

        # Fraction of recent rows where dC > c0 (directional bias)
        c0_val = feats["c0"]
        feats["dC_frac_above_c0_last64"] = float(np.mean(dC_vals[-64:] > c0_val)) if len(dC_vals) >= 4 else 0.5
    else:
        for k in ["dC_last16_median", "dC_last16_std", "dC_last32_median", "dC_last32_std",
                  "dC_16_vs_512", "dC_32_vs_512", "dC_16_vs_128",
                  "dC_slope_last32", "dC_slope_last128",
                  "dC_autocorr1", "dC_iqr_last128", "dC_frac_above_c0_last64"]:
            feats[k] = 0.0

    # --- GR regime change near anchor (last 32 vs 256 rows) ---
    win32_start = max(0, anchor_row - 32)
    known_win32 = known_idx[known_idx >= win32_start]
    if len(known_win32) > 0:
        gr_win32 = gr[known_win32]
        feats["gr_last32_mean"]    = _safe_stat(gr_win32, np.mean)
        feats["gr_last32_std"]     = _safe_stat(gr_win32, np.std)
        feats["gr_last32_vs_256"]  = feats["gr_last32_mean"] - feats.get("gr_mean", 0.0)
    else:
        feats["gr_last32_mean"]   = feats.get("gr_mean", 0.0)
        feats["gr_last32_std"]    = 0.0
        feats["gr_last32_vs_256"] = 0.0

    # --- dip change over last 32 vs last 64 rows ---
    win32_dip_start = max(0, anchor_row - 32)
    known_win32_dip = known_idx[known_idx >= win32_dip_start]
    if len(known_win32_dip) >= 2:
        dz32  = np.diff(z[known_win32_dip])
        dmd32 = np.diff(md[known_win32_dip])
        with np.errstate(divide="ignore", invalid="ignore"):
            dip32 = np.where(np.abs(dmd32) > 0, dz32 / dmd32, 0.0)
        feats["dip_last32_mean"]  = _safe_stat(dip32, np.mean)
        feats["dip_32_vs_64"]     = feats["dip_last32_mean"] - feats.get("dip_last64_mean", 0.0)
    else:
        feats["dip_last32_mean"] = feats.get("dip_last64_mean", 0.0)
        feats["dip_32_vs_64"]    = 0.0

    return feats


def make_well_features(
    samples: list,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Build a feature matrix from a list of OffsetSample objects.

    Args:
        samples   : list of OffsetSample
        return_df : if True, return a DataFrame instead of ndarray

    Returns:
        (N, F) float32 array or DataFrame
    """
    dicts = [make_well_feature_dict(s) for s in samples]
    df = pd.DataFrame(dicts)
    # Coerce to float32, fill NaN with 0
    df = df.fillna(0.0).astype(np.float32)
    if return_df:
        return df
    return df.values


# ---------------------------------------------------------------------------
# Per-segment features (extends well features with segment-specific info)
# ---------------------------------------------------------------------------

def make_segment_feature_dict(
    sample,
    k: int,
    K: int,
) -> dict[str, float]:
    """Features for a specific segment k (0-indexed) of K total segments.

    Includes all well-level features plus segment-specific geometry.
    """
    feats = make_well_feature_dict(sample)

    n_hidden = sample.n_hidden
    z = sample.z.astype(np.float64)
    md = sample.md.astype(np.float64)
    gr = sample.gr.astype(np.float64)
    hidden_rows = sample.hidden_rows

    # Segment position
    feats["seg_k"]             = float(k)
    feats["seg_K"]             = float(K)
    feats["seg_pos_norm"]      = float(k) / max(K - 1, 1)     # 0 = first, 1 = last

    # Segment row range
    from .offsets import segment_assignment
    seg_assign = segment_assignment(n_hidden, K)
    seg_mask = seg_assign == k
    seg_rows = hidden_rows[seg_mask]

    if len(seg_rows) > 0:
        feats["seg_n_rows"]    = float(len(seg_rows))
        seg_z  = z[seg_rows]
        seg_md = md[seg_rows]
        seg_gr = gr[seg_rows]

        feats["seg_z_start"]   = float(z[seg_rows[0]])
        feats["seg_z_end"]     = float(z[seg_rows[-1]])
        feats["seg_dz"]        = float(z[seg_rows[-1]] - z[seg_rows[0]])
        feats["seg_z_mean"]    = _safe_stat(seg_z, np.mean)
        feats["seg_dz_per_row"] = feats["seg_dz"] / max(len(seg_rows), 1)

        # Z relative to anchor
        feats["seg_z_start_rel"] = feats["seg_z_start"] - float(sample.anchor_z)
        feats["seg_z_end_rel"]   = feats["seg_z_end"]   - float(sample.anchor_z)

        feats["seg_md_start"]  = float(md[seg_rows[0]])
        feats["seg_md_end"]    = float(md[seg_rows[-1]])
        feats["seg_md_delta"]  = float(md[seg_rows[-1]] - md[seg_rows[0]])

        feats["seg_gr_mean"]   = _safe_stat(seg_gr, np.mean)
        feats["seg_gr_std"]    = _safe_stat(seg_gr, np.std)
        feats["seg_gr_p50"]    = _percentile(seg_gr, 50)

        # GR comparison vs recent known section (formation change indicator)
        feats["seg_gr_vs_anchor"]  = feats["seg_gr_mean"] - feats.get("gr_anchor", 0.0)
        feats["seg_gr_vs_recent"]  = feats["seg_gr_mean"] - feats.get("gr_last32_mean", 0.0)
        feats["seg_gr_vs_known"]   = feats["seg_gr_mean"] - feats.get("gr_mean", 0.0)
    else:
        for key in ["seg_n_rows", "seg_z_start", "seg_z_end", "seg_dz",
                    "seg_z_mean", "seg_dz_per_row", "seg_z_start_rel",
                    "seg_z_end_rel", "seg_md_start", "seg_md_end",
                    "seg_md_delta", "seg_gr_mean", "seg_gr_std", "seg_gr_p50",
                    "seg_gr_vs_anchor", "seg_gr_vs_recent", "seg_gr_vs_known"]:
            feats[key] = 0.0

    return feats


def make_segment_features(
    samples: list,
    K: int,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Build per-segment feature matrix.

    For each well, produce K rows (one per segment), stacking all wells.

    Args:
        samples   : list of OffsetSample
        K         : number of segments
        return_df : if True, return DataFrame with well_id and seg_k columns

    Returns:
        If return_df: DataFrame with N_wells*K rows
        Else: (N_wells*K, F) float32 array
    """
    dicts = []
    for s in samples:
        for k in range(K):
            d = make_segment_feature_dict(s, k, K)
            if return_df:
                d["_well_id"] = s.well_id
                d["_seg_k"]   = k
            dicts.append(d)

    df = pd.DataFrame(dicts)
    if return_df:
        meta_cols = ["_well_id", "_seg_k"]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        df[feat_cols] = df[feat_cols].fillna(0.0).astype(np.float32)
        return df
    df = df.fillna(0.0).astype(np.float32)
    return df.values


# ---------------------------------------------------------------------------
# Label extraction helpers
# ---------------------------------------------------------------------------

def make_global_offset_labels(
    samples: list,
    grid: np.ndarray,
) -> np.ndarray:
    """Extract oracle global offset bin labels from samples.

    Args:
        samples : training samples with global_offset_star set
        grid    : offset grid

    Returns:
        (N,) int array of bin indices
    """
    from .offsets import offset_to_bin
    labels = []
    for s in samples:
        if s.global_offset_star is None:
            raise ValueError(f"Oracle not computed for well {s.well_id}")
        labels.append(offset_to_bin(s.global_offset_star, grid))
    return np.array(labels, dtype=np.int32)


def make_kseg_offset_labels(
    samples: list,
    K: int,
    grid: np.ndarray,
) -> np.ndarray:
    """Extract per-segment oracle bin labels.

    Returns:
        (N_wells * K,) int array (same row order as make_segment_features)
    """
    from .offsets import offset_to_bin
    labels = []
    for s in samples:
        if K not in s.kseg_offset_star:
            raise ValueError(f"K={K} oracle not computed for well {s.well_id}")
        offsets_k = s.kseg_offset_star[K]
        for k in range(K):
            c_k = offsets_k[k] if k < len(offsets_k) else offsets_k[-1]
            labels.append(offset_to_bin(float(c_k), grid))
    return np.array(labels, dtype=np.int32)
