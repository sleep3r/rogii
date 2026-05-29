"""
mtpnet/local_search.py — GR-guided local path scorer (hengck23 approach).

Core idea (from competition discussion):
  For each current position s0 in the hidden section:
    1. Enumerate candidate constant offsets over a lookahead window [s0, s1]
    2. Integrate TVT: dtvt = -dZ + offset   (MTPNet sign: Z is negative/subsea)
    3. At candidate TVT values, sample typewell GR via interpolation
    4. Score = RMSE(typewell_GR_sampled, horizontal_GR_smooth)
    5. Commit only commit_frac of the best path (receding-horizon / MPC)
    6. Update last_tvt from OWN prediction (no ground-truth leakage)
    7. Advance s0, repeat

Key distinction from static GR-match features:
  - GR is used DYNAMICALLY along each candidate TVT path
  - Not a pre-computed feature: the typewell GR lookup depends on WHERE the
    candidate path goes, so it carries information about whether the path
    is physically consistent with the formation structure
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Typewell loading
# ---------------------------------------------------------------------------

def load_typewell(well_id: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load typewell TVT and GR arrays, sorted by TVT ascending."""
    path = os.path.join(data_dir, f"{well_id}__typewell.csv")
    df = pd.read_csv(path, usecols=["TVT", "GR"])
    df = df.dropna(subset=["TVT", "GR"]).sort_values("TVT").reset_index(drop=True)
    return df["TVT"].values.astype(np.float64), df["GR"].values.astype(np.float64)


# ---------------------------------------------------------------------------
# GR smoothing helper
# ---------------------------------------------------------------------------

def smooth_gr(gr_raw: np.ndarray, window: int = 101, poly: int = 3) -> np.ndarray:
    """Fill NaN → Savitzky-Golay smooth (odd window, handles short arrays)."""
    gr = gr_raw.astype(np.float64).copy()
    nans = ~np.isfinite(gr)
    if nans.any():
        x = np.arange(len(gr))
        finite = ~nans
        if finite.sum() >= 2:
            gr[nans] = np.interp(x[nans], x[finite], gr[finite])
        else:
            gr[nans] = 0.0
    # Ensure odd window, not larger than array
    w = min(window, len(gr))
    if w % 2 == 0:
        w -= 1
    w = max(w, poly + 2)  # polyfit needs at least poly+2 points
    if w < 5:
        return gr
    return savgol_filter(gr, w, poly)


# ---------------------------------------------------------------------------
# Core vectorised scorer
# ---------------------------------------------------------------------------

def gr_path_score_brute(
    sample,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offset_grid: np.ndarray,
    *,
    lookahead: int = 100,
    commit_frac: float = 0.2,
    gr_smooth_window: int = 101,
) -> np.ndarray:
    """
    Rolling GR-guided path scorer.

    Args:
        sample         : OffsetSample with z, gr, anchor_row, anchor_tvt,
                         hidden_rows already set
        typewell_tvt   : sorted TVT array for this well's typewell
        typewell_gr    : GR at each typewell TVT (same order)
        offset_grid    : 1-D array of candidate per-row offsets (ft/row)
        lookahead      : rows to score ahead at each step
        commit_frac    : fraction of lookahead to commit before re-scoring
        gr_smooth_window : Savitzky-Golay window for horizontal GR smoothing

    Returns:
        predicted_tvt : (n_hidden,) float64 — one prediction per hidden row,
                        in the same order as sample.hidden_rows
    """
    hidden_rows = sample.hidden_rows
    n_hidden    = len(hidden_rows)
    if n_hidden == 0:
        return np.array([], dtype=np.float64)

    z          = sample.z.astype(np.float64)
    gr_smooth  = smooth_gr(sample.gr.astype(np.float64), window=gr_smooth_window)

    anchor_row = sample.anchor_row
    last_tvt   = float(sample.anchor_tvt)

    n_off = len(offset_grid)
    predict = np.empty(n_hidden, dtype=np.float64)

    pos = 0   # index into hidden_rows
    while pos < n_hidden:
        end_pos = min(pos + lookahead, n_hidden)
        seg_hi  = hidden_rows[pos:end_pos]   # absolute row indices
        n_seg   = len(seg_hi)

        # dZ for this segment (from the last committed row / anchor)
        prev_row = hidden_rows[pos - 1] if pos > 0 else anchor_row
        prev_z   = float(z[prev_row])
        seg_z    = z[seg_hi]  # (n_seg,)
        all_z    = np.concatenate([[prev_z], seg_z])
        dz_seg   = np.diff(all_z)  # (n_seg,)

        # Vectorised integration: shape (n_off, n_seg)
        # MTPNet sign: dtvt = -dZ + offset
        dtvt = -dz_seg[None, :] + offset_grid[:, None]       # (n_off, n_seg)
        tvt_cand = last_tvt + np.cumsum(dtvt, axis=1)        # (n_off, n_seg)

        # Sample typewell GR at every (offset, row) pair
        gr_cand = np.interp(
            tvt_cand.ravel(), typewell_tvt, typewell_gr
        ).reshape(n_off, n_seg)                               # (n_off, n_seg)

        # GR score per offset = RMSE vs smoothed horizontal GR on this segment
        gr_target = gr_smooth[seg_hi]                         # (n_seg,)
        gr_scores = np.sqrt(np.mean((gr_cand - gr_target[None, :]) ** 2, axis=1))
        best_j = int(np.argmin(gr_scores))

        # Commit commit_frac of the path
        n_commit = max(1, int(commit_frac * n_seg))
        predict[pos: pos + n_commit] = tvt_cand[best_j, :n_commit]
        last_tvt = float(tvt_cand[best_j, n_commit - 1])

        pos += n_commit

    return predict


# ---------------------------------------------------------------------------
# Build a default offset grid for local search
# ---------------------------------------------------------------------------

def make_local_offset_grid(
    lo: float = -0.30,
    hi: float =  0.30,
    n:  int   =  121,
) -> np.ndarray:
    """
    ±0.30 ft/row at 121 bins (step ≈ 0.005 ft/row).

    Rationale:
      - Global offset std ≈ 0.034 ft/row → ±0.30 covers ±8.8 sigma
      - Local offset can briefly exceed global range during formation changes
      - 121 bins gives 0.005 ft/row resolution (adequate for GR scoring)
    """
    return np.linspace(lo, hi, n, dtype=np.float64)
