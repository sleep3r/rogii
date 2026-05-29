"""Blend multiple TVT prediction sources.

Static blend weights (before meta-blender takes over):
    0.55 * NN_optimized
    0.20 * NN_raw
    0.10 * HMM
    0.07 * DTW
    0.05 * neighbor
    0.03 * linear/last_known
"""

from __future__ import annotations

import numpy as np


def static_blend(
    tvt_nn_opt: np.ndarray,
    tvt_nn_raw: np.ndarray,
    tvt_hmm: np.ndarray | None = None,
    tvt_dtw: np.ndarray | None = None,
    tvt_neighbor: np.ndarray | None = None,
    tvt_linear: np.ndarray | None = None,
    w_nn_opt: float = 0.55,
    w_nn_raw: float = 0.20,
    w_hmm: float = 0.10,
    w_dtw: float = 0.07,
    w_neighbor: float = 0.05,
    w_linear: float = 0.03,
) -> np.ndarray:
    """
    Static weighted blend of prediction sources.
    Missing sources have their weight redistributed to nn_opt.
    """
    optional = [
        (tvt_hmm, w_hmm),
        (tvt_dtw, w_dtw),
        (tvt_neighbor, w_neighbor),
        (tvt_linear, w_linear),
    ]
    total_w = w_nn_opt + w_nn_raw
    result = w_nn_opt * tvt_nn_opt + w_nn_raw * tvt_nn_raw

    for arr, w in optional:
        if arr is not None:
            result += w * arr
            total_w += w

    # Renormalise if some sources were missing
    if total_w > 0:
        result /= total_w

    return result.astype(np.float32)


def dynamic_blend(
    preds: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Blend using a dict of {name: array} and {name: weight}."""
    total_w = 0.0
    result = np.zeros_like(next(iter(preds.values())), dtype=np.float32)
    for name, arr in preds.items():
        w = weights.get(name, 0.0)
        if w > 0 and arr is not None:
            result += w * arr
            total_w += w
    if total_w > 0:
        result /= total_w
    return result


def enforce_anchors(
    tvt_pred: np.ndarray,
    tvt_input: np.ndarray,  # NaN where hidden
    known_mask: np.ndarray,
) -> np.ndarray:
    """Hard-replace known rows with TVT_input values."""
    out = tvt_pred.copy()
    km = known_mask > 0.5
    valid_input = np.isfinite(tvt_input)
    replace = km & valid_input
    out[replace] = tvt_input[replace]
    return out
