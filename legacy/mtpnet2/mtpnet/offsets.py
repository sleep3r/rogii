"""Canonical physical integrators for the offset-state TVT decoder.

Convention (canonical — never change):
    dtvt[k] = -(z[h[k]] - z[prev[k]]) + offset_per_row[k]
    pred_tvt  = anchor_tvt + cumsum(dtvt)

    where h[k] = hidden_rows[k], prev[0] = anchor_row, prev[k] = h[k-1]

For constant global offset c:
    pred_tvt[i] = anchor_tvt - (z[h[i]] - z[anchor_row]) + c * row_delta[i]
    row_delta[i] = h[i] - anchor_row

This is the "TVT ≈ -Z + C" decomposition where C is the cumulative formation offset.

Do NOT mix these two expressions:
    (a) global offset c → row_delta scale
    (b) per-row offset c_k → cumsum(-dZ + c_k)
Both are implemented here; (b) reduces to (a) when c_k = c for all k.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Offset grid constants
# ---------------------------------------------------------------------------

OFFSET_LO: float = -0.16      # ft/row  (minimum grid value)
OFFSET_HI: float = 0.16       # ft/row  (maximum grid value)
OFFSET_STEP: float = 0.001    # ft/row
N_OFFSET_BINS: int = 321       # len(np.arange(-0.16, 0.1601, 0.001))


def make_offset_grid(
    lo: float = OFFSET_LO,
    hi: float = OFFSET_HI,
    step: float = OFFSET_STEP,
) -> np.ndarray:
    """Return the canonical offset grid (float64)."""
    grid = np.arange(lo, hi + step * 0.5, step)
    return grid.astype(np.float64)


def offset_to_bin(offset: float, grid: np.ndarray) -> int:
    """Map a float offset value to the nearest grid bin index (0-indexed)."""
    idx = int(np.argmin(np.abs(grid - offset)))
    return int(np.clip(idx, 0, len(grid) - 1))


def bin_to_offset(bin_idx: int | np.ndarray, grid: np.ndarray) -> float | np.ndarray:
    """Map bin index (or array of indices) back to offset values."""
    return grid[bin_idx]


# ---------------------------------------------------------------------------
# Integration function 1: global constant offset
# ---------------------------------------------------------------------------

def tvt_from_global_offset(
    z: np.ndarray,
    anchor_row: int,
    anchor_tvt: float,
    hidden_rows: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Predict TVT for hidden rows using a single global offset (ft/row).

    Formula:
        pred[i] = anchor_tvt
                  - (z[hidden_rows[i]] - z[anchor_row])
                  + offset * (hidden_rows[i] - anchor_row)

    Args:
        z           : full-well Z array (N,)
        anchor_row  : index of the last known row
        anchor_tvt  : TVT at the anchor row
        hidden_rows : indices of hidden rows to predict (H,)
        offset      : global offset in ft/row

    Returns:
        pred : (H,) float32 predicted TVT values
    """
    rows = np.asarray(hidden_rows, dtype=np.int64)
    row_delta = (rows - anchor_row).astype(np.float64)
    z_f = np.asarray(z, dtype=np.float64)
    pred = (
        float(anchor_tvt)
        - (z_f[rows] - float(z_f[anchor_row]))
        + float(offset) * row_delta
    )
    return pred.astype(np.float32)


# ---------------------------------------------------------------------------
# Integration function 2: per-hidden-row offsets (cumsum form)
# ---------------------------------------------------------------------------

def tvt_from_offset_per_row(
    z: np.ndarray,
    anchor_row: int,
    anchor_tvt: float,
    hidden_rows: np.ndarray,
    offset_per_row: np.ndarray,
) -> np.ndarray:
    """Predict TVT using per-hidden-row offsets via cumulative integration.

    Formula:
        dtvt[k] = -(z[h[k]] - z[prev[k]]) + offset_per_row[k]
        pred     = anchor_tvt + cumsum(dtvt)

    where prev[0] = anchor_row, prev[k] = h[k-1]

    This is the canonical form. For constant offset c the result equals
    tvt_from_global_offset when hidden_rows are consecutive integers
    starting at anchor_row + 1.

    Args:
        z              : full-well Z array (N,)
        anchor_row     : index of the last known row
        anchor_tvt     : TVT at the anchor row
        hidden_rows    : indices of hidden rows (H,) — need not be consecutive
        offset_per_row : per-row offset values (H,) in ft/row

    Returns:
        pred : (H,) float32 predicted TVT values
    """
    rows = np.asarray(hidden_rows, dtype=np.int64)
    H = len(rows)
    if H == 0:
        return np.array([], dtype=np.float32)

    z_f = np.asarray(z, dtype=np.float64)
    prev_rows = np.empty(H, dtype=np.int64)
    prev_rows[0] = anchor_row
    prev_rows[1:] = rows[:-1]

    dz = z_f[rows] - z_f[prev_rows]
    dtvt = -dz + np.asarray(offset_per_row, dtype=np.float64)
    return (float(anchor_tvt) + np.cumsum(dtvt)).astype(np.float32)


# ---------------------------------------------------------------------------
# Segment expansion
# ---------------------------------------------------------------------------

def expand_segment_offsets(n_hidden: int, offsets: np.ndarray) -> np.ndarray:
    """Map K segment offsets to n_hidden per-row offsets.

    Equal-spaced segments: hidden row i belongs to segment k where
        k = floor(i / (n_hidden - 1) * K)  clipped to [0, K-1]

    Args:
        n_hidden : number of hidden rows
        offsets  : (K,) segment offset values

    Returns:
        (n_hidden,) per-row offset array (float64)
    """
    K = len(offsets)
    if n_hidden == 0 or K == 0:
        return np.array([], dtype=np.float64)
    pos = np.arange(n_hidden, dtype=np.float64) / max(n_hidden - 1, 1)
    seg = np.minimum((pos * K).astype(np.int64), K - 1)
    return np.asarray(offsets, dtype=np.float64)[seg]


def segment_assignment(n_hidden: int, K: int) -> np.ndarray:
    """Return the segment index for each of the n_hidden hidden rows.

    Args:
        n_hidden : number of hidden rows
        K        : number of segments

    Returns:
        (n_hidden,) int array with values in [0, K-1]
    """
    if n_hidden == 0:
        return np.array([], dtype=np.int64)
    pos = np.arange(n_hidden, dtype=np.float64) / max(n_hidden - 1, 1)
    return np.minimum((pos * K).astype(np.int64), K - 1)


def tvt_from_ksegment_offsets(
    z: np.ndarray,
    anchor_row: int,
    anchor_tvt: float,
    hidden_rows: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Predict TVT using K equal-spaced segment constant offsets.

    Convenience wrapper: expand_segment_offsets → tvt_from_offset_per_row.

    Args:
        z           : full-well Z array (N,)
        anchor_row  : index of the last known row
        anchor_tvt  : TVT at the anchor row
        hidden_rows : indices of hidden rows (H,)
        offsets     : (K,) per-segment offsets in ft/row

    Returns:
        pred : (H,) float32 predicted TVT values
    """
    H = len(hidden_rows)
    offset_per_row = expand_segment_offsets(H, offsets)
    return tvt_from_offset_per_row(z, anchor_row, anchor_tvt, hidden_rows, offset_per_row)


# ---------------------------------------------------------------------------
# Base TVT (zero-offset physical baseline)
# ---------------------------------------------------------------------------

def tvt_base(
    z: np.ndarray,
    anchor_row: int,
    anchor_tvt: float,
    hidden_rows: np.ndarray,
    c0: float = 0.0,
) -> np.ndarray:
    """Physical baseline TVT (zero residual / known c0 drift).

    pred[i] = anchor_tvt - (z[h[i]] - z[anchor_row]) + c0 * (h[i] - anchor_row)

    This is tvt_from_global_offset with offset = c0.

    Args:
        c0 : known drift of C = TVT + Z per row (from prefix statistics)
    """
    return tvt_from_global_offset(z, anchor_row, anchor_tvt, hidden_rows, offset=c0)
