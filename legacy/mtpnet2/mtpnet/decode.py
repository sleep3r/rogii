"""Viterbi and beam-search decoders for the offset-state TVT prediction.

The decoder takes a soft probability distribution over offset bins at each
step (row or segment) and finds the most probable sequence of offset states
subject to smooth-transition constraints.

Two regimes:
  1. Well-level (global): single step, argmax or posterior mean (no Viterbi needed)
  2. K-segment: K steps, Viterbi over segment-level probabilities
  3. Per-row: N_hidden steps, Viterbi over row-level probabilities (expensive,
     use beam for large N_hidden)

Transition model:
  - log_transition[i, j] = -alpha * |i - j|  (exponential penalty for bin jumps)
  - alpha controls smoothness: larger = more penalisation of bin changes

Usage:
    from mtpnet.decode import viterbi_decode, beam_decode

    # K-segment: proba shape (K, N_BINS)
    bins = viterbi_decode(log_proba, alpha=1.0)          # (K,) bin indices
    bins = beam_decode(log_proba, alpha=1.0, beam=32)    # (K,) bin indices
"""
from __future__ import annotations

import numpy as np

from .offsets import (
    N_OFFSET_BINS,
    make_offset_grid,
    tvt_from_ksegment_offsets,
    tvt_from_offset_per_row,
    expand_segment_offsets,
)


# ---------------------------------------------------------------------------
# Transition cost
# ---------------------------------------------------------------------------

def make_log_transition(
    n_bins: int = N_OFFSET_BINS,
    alpha: float = 10.0,
    step: float = 0.001,
) -> np.ndarray:
    """Build the log-transition matrix (n_bins x n_bins).

    log_T[i, j] = -alpha * |grid[i] - grid[j]|
                = -alpha * |i - j| * step

    Larger alpha → more penalisation of large bin jumps.
    step = offset grid step size (0.001 ft/row by default).

    Args:
        n_bins : number of offset bins (321)
        alpha  : smoothness penalty (>0)
        step   : grid step size (ft/row)

    Returns:
        (n_bins, n_bins) float32 log-transition matrix
    """
    diff = np.abs(np.arange(n_bins)[:, None] - np.arange(n_bins)[None, :]).astype(np.float32)
    log_T = -alpha * diff * step
    return log_T


# ---------------------------------------------------------------------------
# Viterbi decoder (exact, O(T * N^2) — practical for K ≤ 15, N ≤ 321)
# ---------------------------------------------------------------------------

def viterbi_decode(
    log_proba: np.ndarray,
    alpha: float = 10.0,
    step: float = 0.001,
    log_transition: np.ndarray | None = None,
) -> np.ndarray:
    """Viterbi decoding over a sequence of log-probability arrays.

    Finds the most probable bin sequence under:
        score(b_1, ..., b_T) = sum_t log_proba[t, b_t]
                             + sum_{t>1} log_T[b_{t-1}, b_t]

    Args:
        log_proba      : (T, N_BINS) log-probability array (can also be proba,
                         will be log-transformed if values are in [0,1])
        alpha          : transition smoothness penalty
        step           : grid step (ft/row), used to scale transition
        log_transition : pre-computed (N_BINS, N_BINS) matrix; if None, computed
                         from alpha and step

    Returns:
        (T,) int32 array of most probable bin indices
    """
    log_p = np.asarray(log_proba, dtype=np.float64)
    T, N = log_p.shape

    # Convert proba → log-proba if needed
    if log_p.max() <= 1.0 + 1e-6 and log_p.min() >= -1e-6:
        log_p = np.log(np.clip(log_p, 1e-30, None))

    if log_transition is None:
        log_T = make_log_transition(N, alpha, step).astype(np.float64)
    else:
        log_T = np.asarray(log_transition, dtype=np.float64)

    # Forward pass
    delta = np.empty((T, N), dtype=np.float64)
    psi   = np.zeros((T, N), dtype=np.int32)

    delta[0] = log_p[0]

    for t in range(1, T):
        # (N,) + (N, N) → best previous for each current bin
        scores = delta[t - 1][:, None] + log_T      # (N_prev, N_curr)
        psi[t]   = np.argmax(scores, axis=0)         # (N_curr,)
        delta[t] = scores[psi[t], np.arange(N)] + log_p[t]

    # Backtrack
    path = np.empty(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


# ---------------------------------------------------------------------------
# Beam search decoder (O(T * beam * N) — practical for large T)
# ---------------------------------------------------------------------------

def beam_decode(
    log_proba: np.ndarray,
    alpha: float = 10.0,
    step: float = 0.001,
    beam_width: int = 32,
    log_transition: np.ndarray | None = None,
) -> np.ndarray:
    """Beam-search decoding (approximate Viterbi for large T).

    Maintains a beam of the top-`beam_width` partial sequences at each step.

    Args:
        log_proba  : (T, N_BINS)
        alpha      : transition smoothness penalty
        step       : grid step (ft/row)
        beam_width : number of hypotheses to keep
        log_transition : pre-computed log-transition matrix

    Returns:
        (T,) int32 best bin path
    """
    log_p = np.asarray(log_proba, dtype=np.float64)
    T, N = log_p.shape

    if log_p.max() <= 1.0 + 1e-6 and log_p.min() >= -1e-6:
        log_p = np.log(np.clip(log_p, 1e-30, None))

    if log_transition is None:
        log_T = make_log_transition(N, alpha, step).astype(np.float64)
    else:
        log_T = np.asarray(log_transition, dtype=np.float64)

    # Beam: list of (score, path_list)
    # Seed: top-beam_width bins at t=0
    init_scores = log_p[0]
    top_init = np.argsort(init_scores)[::-1][:beam_width]
    beam: list[tuple[float, list[int]]] = [
        (float(init_scores[b]), [int(b)]) for b in top_init
    ]

    for t in range(1, T):
        candidates: list[tuple[float, list[int]]] = []
        obs = log_p[t]  # (N,)
        for score, path in beam:
            last_bin = path[-1]
            trans = log_T[last_bin]       # (N,)
            extended = score + trans + obs  # (N,)
            # Top candidates from this beam hypothesis
            top_bins = np.argsort(extended)[::-1][:beam_width]
            for b in top_bins:
                candidates.append((float(extended[b]), path + [int(b)]))
        # Prune beam
        candidates.sort(key=lambda x: -x[0])
        beam = candidates[:beam_width]

    best_score, best_path = max(beam, key=lambda x: x[0])
    return np.array(best_path, dtype=np.int32)


# ---------------------------------------------------------------------------
# Greedy / argmax decoder (baseline, no smoothing)
# ---------------------------------------------------------------------------

def greedy_decode(
    proba: np.ndarray,
) -> np.ndarray:
    """Simple argmax decoder (no transition constraints).

    Args:
        proba : (T, N_BINS) or (N_BINS,) probability array

    Returns:
        (T,) or scalar int32 bin index
    """
    p = np.asarray(proba)
    if p.ndim == 1:
        return np.int32(np.argmax(p))
    return np.argmax(p, axis=1).astype(np.int32)


def smooth_decode(
    proba: np.ndarray,
    sigma_bins: float = 5.0,
) -> np.ndarray:
    """Smooth proba with a Gaussian before argmax (soft prior).

    Equivalent to a very weak Viterbi without sequence constraints.
    Useful for the global single-offset case.

    Args:
        proba      : (N_BINS,) probability vector
        sigma_bins : Gaussian smoothing width in bins

    Returns:
        scalar int32 best bin after smoothing
    """
    from scipy.ndimage import gaussian_filter1d  # type: ignore
    p = np.asarray(proba, dtype=np.float64)
    p_smooth = gaussian_filter1d(p, sigma=sigma_bins)
    return np.int32(np.argmax(p_smooth))


# ---------------------------------------------------------------------------
# Full well decode: K-segment → TVT predictions
# ---------------------------------------------------------------------------

def decode_kseg_predictions(
    samples: list,          # list[OffsetSample]
    proba_list: list[np.ndarray],   # per-well (K, N_BINS) proba arrays
    K: int,
    grid: np.ndarray | None = None,
    decoder: str = "viterbi",
    alpha: float = 10.0,
    beam_width: int = 32,
) -> list[np.ndarray]:
    """Decode K-segment proba arrays → TVT predictions for each well.

    Args:
        samples    : list of OffsetSample
        proba_list : list of (K, N_BINS) arrays (one per well)
        K          : number of segments
        grid       : offset grid; defaults to make_offset_grid()
        decoder    : "viterbi", "beam", or "greedy"
        alpha      : Viterbi/beam transition penalty
        beam_width : beam size for beam decoder

    Returns:
        List of (H_i,) float32 TVT prediction arrays (one per well)
    """
    if grid is None:
        grid = make_offset_grid()

    step = float(grid[1] - grid[0]) if len(grid) > 1 else 0.001
    log_T = make_log_transition(len(grid), alpha, step)

    predictions = []
    for s, proba_k in zip(samples, proba_list):
        proba_k = np.asarray(proba_k, dtype=np.float64)   # (K, N_BINS)

        if decoder == "viterbi":
            bins = viterbi_decode(proba_k, alpha=alpha, step=step, log_transition=log_T)
        elif decoder == "beam":
            bins = beam_decode(proba_k, alpha=alpha, step=step, beam_width=beam_width, log_transition=log_T)
        else:  # greedy
            bins = greedy_decode(proba_k)

        offsets = grid[bins]  # (K,)
        pred = tvt_from_ksegment_offsets(
            s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, offsets
        )
        predictions.append(pred)

    return predictions


def decode_global_predictions(
    samples: list,          # list[OffsetSample]
    proba_list: list[np.ndarray],   # per-well (N_BINS,) arrays
    grid: np.ndarray | None = None,
    mode: str = "argmax",
) -> list[np.ndarray]:
    """Decode global proba arrays → TVT predictions for each well.

    Args:
        samples    : list of OffsetSample
        proba_list : list of (N_BINS,) arrays
        grid       : offset grid
        mode       : "argmax" or "mean"

    Returns:
        List of (H_i,) float32 TVT prediction arrays
    """
    from .offsets import tvt_from_global_offset
    if grid is None:
        grid = make_offset_grid()

    predictions = []
    for s, proba in zip(samples, proba_list):
        p = np.asarray(proba, dtype=np.float64)
        if mode == "mean":
            c = float(np.dot(p / (p.sum() + 1e-30), grid))
        else:
            c = float(grid[np.argmax(p)])
        pred = tvt_from_global_offset(s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, c)
        predictions.append(pred)

    return predictions
