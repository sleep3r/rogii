from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Envelope:
    low: np.ndarray
    high: np.ndarray
    mid: np.ndarray
    selected_indices: np.ndarray


def select_template_indices(
    scores: np.ndarray,
    *,
    top_n: int,
    min_score: float | None = None,
) -> np.ndarray:
    """Pick up to `top_n` template indices by score.

    Only finite scores are eligible; if `min_score` is set, scores must also
    exceed that threshold. Returns indices in descending-score order. Returns
    an empty array if nothing qualifies (callers should treat this as a
    well-level skip rather than a forced envelope).
    """
    arr = np.asarray(scores, dtype=np.float64)
    eligible = np.isfinite(arr)
    if min_score is not None:
        eligible &= arr > float(min_score)
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        return np.array([], dtype=np.int64)
    order = indices[np.argsort(arr[indices])[::-1]]
    return order[: max(int(top_n), 1)].astype(np.int64)


def build_envelope_from_paths(
    paths: np.ndarray,
    *,
    selected_indices: np.ndarray | None = None,
    low_quantile: float = 0.0,
    high_quantile: float = 1.0,
) -> Envelope:
    arr = np.asarray(paths, dtype=np.float64)
    if selected_indices is not None:
        arr = arr[np.asarray(selected_indices, dtype=np.int64)]
    else:
        selected_indices = np.arange(arr.shape[0], dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("Need at least one selected path to build envelope")
    low = np.nanquantile(arr, float(low_quantile), axis=0)
    high = np.nanquantile(arr, float(high_quantile), axis=0)
    mid = 0.5 * (low + high)
    return Envelope(low=low, high=high, mid=mid, selected_indices=np.asarray(selected_indices, dtype=np.int64))

