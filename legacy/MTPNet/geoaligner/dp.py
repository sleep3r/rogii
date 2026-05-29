from __future__ import annotations

import numpy as np


def viterbi_decode(
    log_probs: np.ndarray,
    *,
    max_jump_bins: int = 6,
    jump_penalty: float = 0.03,
    start_bin: int | None = None,
    start_penalty: float = 0.0,
) -> np.ndarray:
    """Decode a locally monotone/smooth alignment path through [T,H] log-probs."""
    scores = np.asarray(log_probs, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("log_probs must have shape [T,H]")
    T, H = scores.shape
    if T == 0 or H == 0:
        return np.empty(0, dtype=np.int32)

    dp = np.full((T, H), -np.inf, dtype=np.float32)
    back = np.full((T, H), -1, dtype=np.int32)
    dp[0] = scores[0]
    if start_bin is not None:
        bins = np.arange(H, dtype=np.float32)
        dp[0] -= float(start_penalty) * np.abs(bins - float(start_bin))

    for t in range(1, T):
        for j in range(H):
            lo = max(0, j - max_jump_bins)
            hi = min(H, j + max_jump_bins + 1)
            prev = np.arange(lo, hi, dtype=np.int32)
            jump = np.abs(j - prev).astype(np.float32)
            cand = dp[t - 1, lo:hi] - float(jump_penalty) * jump
            best_local = int(np.argmax(cand))
            dp[t, j] = scores[t, j] + cand[best_local]
            back[t, j] = prev[best_local]

    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path

