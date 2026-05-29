"""Typewell-anchored scoring for plane-coordinate extrapolation.

Given a typewell with the curves ``GR(TVT)`` and (sometimes) ``Geology``, we can
score a *plane-coordinate hypothesis* ``c_hat(MD)`` by comparing the resampled
typewell GR signal against the horizontal-well GR signal:

* For each hidden row ``i``, the candidate TVT is ``tvt_hat_i = c_hat(MD_i) - Z_i``.
* The expected GR at that TVT is ``typewell_GR(tvt_hat_i)``.
* A good plane hypothesis aligns the horizontal GR with the typewell GR.

This module provides a small, fast scoring helper that evaluates a grid of
constant *offsets* applied to a baseline plane curve, returning the offset that
best matches GR. It is intentionally narrow: a full search would belong in a
separate path-search module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TypewellRef:
    """Sorted typewell reference curve.

    Attributes
    ----------
    tvt
        Strictly increasing TVT grid, shape ``(M,)``.
    gr
        GR values aligned with ``tvt``, shape ``(M,)``.
    tvt_min, tvt_max
        Cached range of the typewell grid.
    """

    tvt: np.ndarray
    gr: np.ndarray
    tvt_min: float
    tvt_max: float


def build_typewell_ref(typewell_df) -> TypewellRef:
    """Build a sorted, dedup'd typewell reference from a DataFrame-like object.

    Expects columns ``TVT`` and ``GR``.
    """

    import pandas as pd  # local import; keeps the dependency optional.

    df = typewell_df[["TVT", "GR"]].dropna()
    df = df.sort_values("TVT").drop_duplicates(subset="TVT")
    tvt = df["TVT"].to_numpy(dtype=np.float64)
    gr = df["GR"].to_numpy(dtype=np.float64)
    if tvt.size < 2:
        raise ValueError("Typewell reference requires at least 2 finite (TVT, GR) rows")
    return TypewellRef(
        tvt=tvt,
        gr=gr,
        tvt_min=float(tvt[0]),
        tvt_max=float(tvt[-1]),
    )


def _interp_gr(ref: TypewellRef, tvt_query: np.ndarray) -> np.ndarray:
    """Linear interpolation with NaN outside the typewell range."""

    out = np.interp(tvt_query, ref.tvt, ref.gr, left=np.nan, right=np.nan)
    return out


def score_plane_offsets(
    *,
    plane_baseline: np.ndarray,
    z: np.ndarray,
    gr_h: np.ndarray,
    typewell: TypewellRef,
    offsets: np.ndarray,
    eval_mask: np.ndarray,
) -> np.ndarray:
    """Score a grid of constant offsets applied to ``plane_baseline``.

    For each offset ``o`` in ``offsets``:

    1. ``tvt_hat = plane_baseline + o - z``
    2. ``gr_pred = typewell_GR(tvt_hat)``
    3. score = ``- mean( (gr_pred - gr_h)^2 )`` over rows in ``eval_mask`` where
       both sides are finite.

    Returns ``scores`` of shape ``(len(offsets),)``; higher is better.
    """

    plane_baseline = np.asarray(plane_baseline, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    gr_h = np.asarray(gr_h, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    if not eval_mask.any():
        return np.full_like(offsets, np.nan, dtype=np.float64)

    z_e = z[eval_mask]
    gr_e = gr_h[eval_mask]
    plane_e = plane_baseline[eval_mask]

    scores = np.empty(offsets.shape[0], dtype=np.float64)
    for k, o in enumerate(offsets):
        tvt_hat = plane_e + o - z_e
        gr_pred = _interp_gr(typewell, tvt_hat)
        valid = np.isfinite(gr_pred) & np.isfinite(gr_e)
        if valid.sum() < 32:
            scores[k] = -np.inf
            continue
        err = gr_pred[valid] - gr_e[valid]
        scores[k] = -float(np.mean(err * err))
    return scores


def best_offset(
    *,
    plane_baseline: np.ndarray,
    z: np.ndarray,
    gr_h: np.ndarray,
    typewell: TypewellRef,
    search_radius_ft: float = 50.0,
    n_grid: int = 401,
    eval_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the best constant offset on a symmetric grid plus the score curve."""

    offsets = np.linspace(-search_radius_ft, search_radius_ft, n_grid)
    if eval_mask is None:
        eval_mask = np.isfinite(plane_baseline) & np.isfinite(z) & np.isfinite(gr_h)
    scores = score_plane_offsets(
        plane_baseline=plane_baseline,
        z=z,
        gr_h=gr_h,
        typewell=typewell,
        offsets=offsets,
        eval_mask=eval_mask,
    )
    if not np.isfinite(scores).any():
        return 0.0, scores
    best = int(np.nanargmax(scores))
    return float(offsets[best]), scores


def locally_anchored_offset(
    *,
    plane_baseline: np.ndarray,
    z: np.ndarray,
    gr_h: np.ndarray,
    typewell: TypewellRef,
    hidden_mask: np.ndarray,
    search_radius_ft: float = 30.0,
    n_grid: int = 201,
    n_chunks: int = 6,
    smooth_lambda: float = 1.0,
) -> np.ndarray:
    """Per-chunk offset search with quadratic smoothness regularization.

    The hidden region is split into ``n_chunks`` contiguous chunks (by row
    index). For each chunk we evaluate ``score_plane_offsets`` and recover the
    raw best offset. Then we apply a small global smoothing pass:

        offset_k* = argmin_{o}  -score_k(o) + lambda * (o - offset_{k-1}*)^2

    solved as a 1-D forward sweep — keeps neighbouring chunks aligned (avoids
    flipping between PWL plateaus on noisy GR).

    Returns an array of the same shape as ``plane_baseline`` containing the
    per-row offset to add to ``plane_baseline``. Non-hidden rows get ``0``.
    """

    plane_baseline = np.asarray(plane_baseline, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    gr_h = np.asarray(gr_h, dtype=np.float64)
    hidden_mask = np.asarray(hidden_mask, dtype=bool)

    n = plane_baseline.shape[0]
    per_row_offset = np.zeros(n, dtype=np.float64)
    if not hidden_mask.any():
        return per_row_offset

    hidden_idx = np.nonzero(hidden_mask)[0]
    chunk_starts = np.linspace(hidden_idx[0], hidden_idx[-1] + 1, n_chunks + 1, dtype=int)
    offsets = np.linspace(-search_radius_ft, search_radius_ft, n_grid)

    chunk_scores: list[np.ndarray] = []
    chunk_ranges: list[tuple[int, int]] = []
    for k in range(n_chunks):
        a = int(chunk_starts[k])
        b = int(chunk_starts[k + 1])
        chunk_mask = np.zeros(n, dtype=bool)
        chunk_mask[a:b] = hidden_mask[a:b]
        if chunk_mask.sum() < 32:
            # Not enough rows in this chunk – skip with flat zero score.
            chunk_scores.append(np.zeros_like(offsets))
            chunk_ranges.append((a, b))
            continue
        scores = score_plane_offsets(
            plane_baseline=plane_baseline,
            z=z,
            gr_h=gr_h,
            typewell=typewell,
            offsets=offsets,
            eval_mask=chunk_mask,
        )
        chunk_scores.append(scores)
        chunk_ranges.append((a, b))

    # Smoothed DP across chunks.
    K = len(chunk_scores)
    G = offsets.size
    finite_scores = np.stack(
        [np.where(np.isfinite(s), s, -1e18) for s in chunk_scores], axis=0
    )  # (K, G)
    # We want to maximise sum_k score_k(o_k) - lambda * sum_k (o_k - o_{k-1})^2
    # Use scaled lambda relative to the score range.
    score_scale = float(np.ptp(finite_scores[np.isfinite(finite_scores)])) or 1.0
    lam = smooth_lambda * score_scale / max(1.0, (2 * search_radius_ft) ** 2)
    dp = np.full((K, G), -np.inf)
    backptr = np.zeros((K, G), dtype=np.int32)
    dp[0] = finite_scores[0]
    for k in range(1, K):
        prev = dp[k - 1]
        diff = offsets[:, None] - offsets[None, :]
        cost = -lam * (diff ** 2)  # (G, G) penalty for transitioning
        candidate = prev[None, :] + cost  # (G, G_prev)
        backptr[k] = np.argmax(candidate, axis=1)
        dp[k] = finite_scores[k] + candidate[np.arange(G), backptr[k]]

    # Trace back.
    best_o = np.zeros(K, dtype=np.int32)
    best_o[-1] = int(np.argmax(dp[-1]))
    for k in range(K - 2, -1, -1):
        best_o[k] = backptr[k + 1, best_o[k + 1]]

    for k, (a, b) in enumerate(chunk_ranges):
        per_row_offset[a:b] = offsets[best_o[k]]

    # Force offset to zero outside the hidden region; smoothly ramp in at the
    # boundary so we don't introduce a step.
    per_row_offset = per_row_offset * hidden_mask.astype(np.float64)
    return per_row_offset
