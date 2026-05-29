"""
dlmtp/dp_decode.py  —  DP (Viterbi) path decoder over (L, J) logits.

The decoder finds a path {j_0, j_1, ..., j_{L-1}} that maximises:
    sum_s log_softmax(logits[s, j_s])  -  lambda_smooth * sum_s |j_s - j_{s-1}|^2

Transitions are band-limited (|j_s - j_{s-1}| <= max_trans) for speed.

Physical interpretation
-----------------------
With bin_ft = 2 ft and typical dZ/row ≈ 0–2 ft, the true TVT bin barely
moves per row when the path stays close to the prior.  A max_trans of 5–8
bins corresponds to ±10–16 ft per row, which is very generous.
"""
from __future__ import annotations

import numpy as np


def dp_decode(
    logits: np.ndarray,         # (L, J) float — raw logits (not softmax)
    lambda_smooth: float = 0.05,
    max_trans: int = 8,
) -> np.ndarray:
    """Band-limited Viterbi decoder — fully vectorised over J.

    At each step s, for every target bin j we try all previous bins
    j_prev = j - delta  for delta in [-max_trans, max_trans].
    The transition cost is  -lambda_smooth * delta^2.

    Complexity: O(L * J * max_trans)  —  all J operations are batched.

    Returns
    -------
    path : (L,) int array — bin index for each row
    """
    L, J = logits.shape
    # Row-wise log-softmax (stable)
    mx    = logits.max(axis=1, keepdims=True)
    lse   = np.log(np.exp(logits - mx).sum(axis=1, keepdims=True))
    log_p = (logits - mx - lse).astype(np.float64)    # (L, J)

    # Gaussian prior centred at bin J//2  (prevents cold-start collapse to bin 0).
    # lambda=5e-3: penalty at bin 0 = 11.5 nats; at ±20 bins = 2.0 nats.
    # Safe for trained model (logit advantage at true bin >> 2 nats).
    # Fixes cold-start for near-uniform logits: with std=0.01 noise, mean_bin → 48.
    _j = np.arange(J, dtype=np.float64) - J // 2
    log_p = log_p - 5e-3 * (_j * _j)[None, :]

    dp = np.full((L, J), -np.inf, dtype=np.float64)
    bt = np.zeros((L, J), dtype=np.int32)
    dp[0] = log_p[0]

    j_idx = np.arange(J, dtype=np.int32)               # (J,)

    for s in range(1, L):
        best_val  = np.full(J, -np.inf, dtype=np.float64)
        best_from = np.zeros(J, dtype=np.int32)
        prev      = dp[s - 1]                           # (J,)

        for delta in range(-max_trans, max_trans + 1):
            pen = lambda_smooth * float(delta * delta)
            # Transition: j_prev = j - delta  (delta>0 → move right; delta<0 → move left)
            # src[j] = prev[j - delta]
            src = np.full(J, -np.inf, dtype=np.float64)
            if delta >= 0:                              # j_prev = j - delta; valid j >= delta
                src[delta:] = prev[:J - delta]
            else:                                       # j_prev = j + |delta|; valid j < J-|delta|
                d = -delta
                src[:J - d] = prev[d:]
            cand = src - pen
            better = cand > best_val
            best_val  = np.where(better, cand, best_val)
            # j_prev = j - delta (consistent with forward pass above)
            j_prev = np.clip(j_idx - delta, 0, J - 1)
            best_from = np.where(better, j_prev, best_from)

        dp[s] = best_val + log_p[s]
        bt[s] = best_from

    # Traceback
    path        = np.empty(L, dtype=np.int32)
    path[L - 1] = int(np.argmax(dp[L - 1]))
    for s in range(L - 2, -1, -1):
        path[s] = bt[s + 1, path[s + 1]]

    return path


def argmax_decode(logits: np.ndarray) -> np.ndarray:
    """Trivial row-wise argmax (fast baseline, no smoothness)."""
    return logits.argmax(axis=1).astype(np.int32)


def bins_to_tvt(
    path: np.ndarray,           # (nh,) int bin indices
    prior_tvt: np.ndarray,      # (nh,) prior TVT (float64)
    tvt_bins: int,
    bin_ft: float,
) -> np.ndarray:
    """Convert bin indices to absolute TVT values.

    tvt[s] = prior_tvt[s] + (path[s] - J//2) * bin_ft
    """
    offsets = (path.astype(np.float64) - tvt_bins // 2) * bin_ft
    return (prior_tvt + offsets).astype(np.float64)
