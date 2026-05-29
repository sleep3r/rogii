"""Segment-wise soft-DTW alignment prior.

A faster, lighter complement to HMM: aligns GR segments to typewell using
a Sakoe-Chiba banded DP, allowing forward and reverse alignment.

This produces a per-well TVT path that can be used as an additional feature
or distillation target.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback only when optional dep is absent
    njit = None
    _NUMBA_AVAILABLE = False

_EPS = 1e-8


def run_segment_dtw(
    gr_obs: np.ndarray,  # [L] float, NaN-filled already
    tw_tvt: np.ndarray,  # [M] typewell TVT
    tw_gr: np.ndarray,  # [M] typewell GR
    anchor_tvt: np.ndarray,  # [L] float, NaN where hidden
    tvt_step: float = 2.0,
    band_pct: float = 0.15,  # Sakoe-Chiba band as fraction of L
    smooth_sigma: float = 10.0,
) -> dict[str, np.ndarray]:
    """
    DTW-based TVT path estimation.

    Returns:
        dtw_tvt       — [L] float32 TVT path
        dtw_score     — scalar cost (per-row)
        dtw_gr_pred   — [L] predicted GR along the DTW path
    """
    L = len(gr_obs)
    valid = np.isfinite(gr_obs)
    if valid.sum() < 10:
        return _dtw_fallback(anchor_tvt, L)

    # Smooth GR to reduce noise
    gr_filled = _fill_gr(gr_obs)
    gr_s = gaussian_filter1d(gr_filled, sigma=smooth_sigma)

    # Use linear anchor as TVT reference to slice typewell window
    lin_tvt = _linear_interp(anchor_tvt)
    tvt_lo = max(lin_tvt.min() - 80.0, tw_tvt.min())
    tvt_hi = min(lin_tvt.max() + 80.0, tw_tvt.max())

    # Typewell indices within window
    tw_mask = (tw_tvt >= tvt_lo) & (tw_tvt <= tvt_hi)
    tw_tvt_w = tw_tvt[tw_mask]
    tw_gr_w = tw_gr[tw_mask]
    M = len(tw_tvt_w)
    if M < 5:
        return _dtw_fallback(anchor_tvt, L)

    # Smooth typewell GR
    tw_gr_s = gaussian_filter1d(tw_gr_w, sigma=max(2.0, smooth_sigma / 3.0))

    # Normalise both GR traces
    gr_mu, gr_std = gr_s.mean(), gr_s.std() + _EPS
    tw_mu, tw_std = tw_gr_s.mean(), tw_gr_s.std() + _EPS
    gr_norm = (gr_s - gr_mu) / gr_std
    tw_norm = (tw_gr_s - tw_mu) / tw_std

    # Banded DTW: try forward and reverse typewell orientation.
    band = max(3, int(band_pct * max(L, M)))
    cost_f, path_i_f, path_j_f = _banded_dtw(gr_norm, tw_norm, band)
    cost_r, path_i_r, path_j_r = _banded_dtw(gr_norm, tw_norm[::-1], band)
    if cost_r < cost_f:
        cost, path_i, path_j = cost_r, path_i_r, path_j_r
        tw_tvt_oriented = tw_tvt_w[::-1]
        orientation = -1.0
    else:
        cost, path_i, path_j = cost_f, path_i_f, path_j_f
        tw_tvt_oriented = tw_tvt_w
        orientation = 1.0
    if float(np.nanstd(gr_s)) < 1e-3:
        orientation = 0.0

    # Map DTW j-indices back to TVT
    dtw_tvt = np.full(L, np.nan, dtype=np.float64)
    for i, j in zip(path_i, path_j):
        if 0 <= i < L and 0 <= j < M:
            dtw_tvt[i] = tw_tvt_oriented[j]

    # Fill any remaining NaN using linear interp
    finite_i = np.where(np.isfinite(dtw_tvt))[0]
    if len(finite_i) > 1:
        dtw_tvt = np.interp(np.arange(L), finite_i, dtw_tvt[finite_i])
    else:
        dtw_tvt = lin_tvt.copy()

    # Predicted GR along path
    dtw_gr_pred = np.interp(dtw_tvt, tw_tvt, tw_gr).astype(np.float32)
    dtw_tvt = dtw_tvt.astype(np.float32)
    per_row_cost = cost / max(len(path_i), 1)

    return {
        "dtw_tvt": dtw_tvt,
        "dtw_score": np.full(L, float(per_row_cost), dtype=np.float32),
        "dtw_gr_pred": dtw_gr_pred,
        "dtw_orientation": np.full(L, orientation, dtype=np.float32),
        "dtw_segment_id": np.zeros(L, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Banded DTW
# ---------------------------------------------------------------------------


def _banded_dtw(
    x: np.ndarray,  # [N] query
    y: np.ndarray,  # [M] reference
    band: int,
) -> tuple[float, list, list]:
    if _NUMBA_AVAILABLE:
        cost, path_i, path_j = _banded_dtw_numba(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            int(band),
        )
        return float(cost), path_i.tolist(), path_j.tolist()
    return _banded_dtw_python(x, y, band)


def _banded_dtw_python(
    x: np.ndarray,  # [N] query
    y: np.ndarray,  # [M] reference
    band: int,
) -> tuple[float, list, list]:
    N, M = len(x), len(y)
    INF = 1e30
    D = np.full((N, M), INF, dtype=np.float64)

    # Fill only within band
    for i in range(N):
        j_lo = max(0, i - band)
        j_hi = min(M, i + band + 1)
        for j in range(j_lo, j_hi):
            cost_ij = (x[i] - y[j]) ** 2
            if i == 0 and j == 0:
                D[i, j] = cost_ij
            elif i == 0:
                D[i, j] = D[i, j - 1] + cost_ij if j > 0 and D[i, j - 1] < INF else INF
            elif j == 0:
                D[i, j] = D[i - 1, j] + cost_ij if i > 0 and D[i - 1, j] < INF else INF
            else:
                prev = INF
                if abs((i - 1) - j) <= band and D[i - 1, j] < INF:
                    prev = min(prev, D[i - 1, j])
                if abs(i - (j - 1)) <= band and D[i, j - 1] < INF:
                    prev = min(prev, D[i, j - 1])
                if abs((i - 1) - (j - 1)) <= band and D[i - 1, j - 1] < INF:
                    prev = min(prev, D[i - 1, j - 1])
                D[i, j] = prev + cost_ij if prev < INF else INF

    total_cost = D[N - 1, M - 1] if D[N - 1, M - 1] < INF else INF

    # Backtrack
    path_i, path_j = [N - 1], [M - 1]
    i, j = N - 1, M - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = []
            if abs((i - 1) - (j - 1)) <= band:
                candidates.append((D[i - 1, j - 1], i - 1, j - 1))
            if abs((i - 1) - j) <= band:
                candidates.append((D[i - 1, j], i - 1, j))
            if abs(i - (j - 1)) <= band:
                candidates.append((D[i, j - 1], i, j - 1))
            if not candidates:
                break
            _, i, j = min(candidates, key=lambda t: t[0])
        path_i.append(i)
        path_j.append(j)

    path_i.reverse()
    path_j.reverse()
    return total_cost, path_i, path_j


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _banded_dtw_numba(x: np.ndarray, y: np.ndarray, band: int):
        N = x.shape[0]
        M = y.shape[0]
        INF = 1e30
        D = np.empty((N, M), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                D[i, j] = INF

        for i in range(N):
            j_lo = 0
            if i - band > j_lo:
                j_lo = i - band
            j_hi = M
            if i + band + 1 < j_hi:
                j_hi = i + band + 1

            for j in range(j_lo, j_hi):
                diff = x[i] - y[j]
                cost_ij = diff * diff
                if i == 0 and j == 0:
                    D[i, j] = cost_ij
                elif i == 0:
                    if j > 0 and D[i, j - 1] < INF:
                        D[i, j] = D[i, j - 1] + cost_ij
                    else:
                        D[i, j] = INF
                elif j == 0:
                    if D[i - 1, j] < INF:
                        D[i, j] = D[i - 1, j] + cost_ij
                    else:
                        D[i, j] = INF
                else:
                    prev = INF
                    if abs((i - 1) - j) <= band and D[i - 1, j] < prev:
                        prev = D[i - 1, j]
                    if abs(i - (j - 1)) <= band and D[i, j - 1] < prev:
                        prev = D[i, j - 1]
                    if abs((i - 1) - (j - 1)) <= band and D[i - 1, j - 1] < prev:
                        prev = D[i - 1, j - 1]
                    if prev < INF:
                        D[i, j] = prev + cost_ij
                    else:
                        D[i, j] = INF

        total_cost = D[N - 1, M - 1]
        if total_cost >= INF:
            total_cost = INF

        path_i_rev = np.empty(N + M, dtype=np.int64)
        path_j_rev = np.empty(N + M, dtype=np.int64)
        path_len = 1
        i = N - 1
        j = M - 1
        path_i_rev[0] = i
        path_j_rev[0] = j

        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                best = INF
                best_i = i
                best_j = j

                if abs((i - 1) - (j - 1)) <= band and D[i - 1, j - 1] < best:
                    best = D[i - 1, j - 1]
                    best_i = i - 1
                    best_j = j - 1
                if abs((i - 1) - j) <= band and D[i - 1, j] < best:
                    best = D[i - 1, j]
                    best_i = i - 1
                    best_j = j
                if abs(i - (j - 1)) <= band and D[i, j - 1] < best:
                    best = D[i, j - 1]
                    best_i = i
                    best_j = j - 1
                if best >= INF:
                    break
                i = best_i
                j = best_j

            path_i_rev[path_len] = i
            path_j_rev[path_len] = j
            path_len += 1

        path_i = np.empty(path_len, dtype=np.int64)
        path_j = np.empty(path_len, dtype=np.int64)
        for k in range(path_len):
            src = path_len - 1 - k
            path_i[k] = path_i_rev[src]
            path_j[k] = path_j_rev[src]

        return total_cost, path_i, path_j

else:
    _banded_dtw_numba = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_gr(gr: np.ndarray) -> np.ndarray:
    gr = gr.copy()
    valid = np.isfinite(gr)
    if valid.sum() == 0:
        return np.zeros_like(gr)
    return np.interp(np.arange(len(gr)), np.where(valid)[0], gr[valid]).astype(np.float32)


def _linear_interp(tvt_input: np.ndarray) -> np.ndarray:
    n = len(tvt_input)
    known = np.where(np.isfinite(tvt_input))[0]
    if len(known) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.interp(np.arange(n), known, tvt_input[known]).astype(np.float32)


def _dtw_fallback(anchor_tvt: np.ndarray, L: int) -> dict[str, np.ndarray]:
    lin = _linear_interp(anchor_tvt)
    return {
        "dtw_tvt": lin,
        "dtw_score": np.full(L, 999.0, dtype=np.float32),
        "dtw_gr_pred": np.zeros(L, dtype=np.float32),
        "dtw_orientation": np.zeros(L, dtype=np.float32),
        "dtw_segment_id": np.zeros(L, dtype=np.float32),
    }
