from __future__ import annotations

import numpy as np


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std < 1e-8:
        return arr * 0.0
    return (arr - mean) / std


def score_template_match(
    template_gr: np.ndarray,
    horizontal_gr: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    min_finite_fraction: float = 0.5,
    min_finite_count: int = 3,
) -> float:
    """Weighted z-correlation between template and horizontal GR.

    Returns `-inf` if too few overlap points (avoids spurious near-perfect
    correlations when the path falls almost entirely outside the typewell
    TVT support, e.g. for extreme scale values).
    """
    template = np.asarray(template_gr, dtype=np.float64)
    horizontal = np.asarray(horizontal_gr, dtype=np.float64)
    finite = np.isfinite(template) & np.isfinite(horizontal)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        finite &= np.isfinite(w) & (w > 0)
    else:
        w = np.ones_like(template, dtype=np.float64)
    n_finite = int(finite.sum())
    n_eligible = int((np.isfinite(w) & (w > 0)).sum()) if weights is not None else template.size
    if n_finite < max(int(min_finite_count), 3):
        return float("-inf")
    if n_eligible > 0 and (n_finite / n_eligible) < float(min_finite_fraction):
        return float("-inf")
    x = _zscore(template[finite])
    y = _zscore(horizontal[finite])
    wf = w[finite]
    wf = wf / max(float(wf.sum()), 1e-12)
    return float(np.sum(wf * x * y))


def gr_variant(
    gr: np.ndarray,
    *,
    hidden_mask: np.ndarray,
    variant: str,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(gr, dtype=np.float64).copy()
    hidden = np.asarray(hidden_mask, dtype=bool)
    if variant == "normal_GR":
        return out
    if variant == "shuffled_hidden_GR":
        values = out[hidden].copy()
        rng.shuffle(values)
        out[hidden] = values
        return out
    if variant == "zero_hidden_GR":
        out[hidden] = 0.0
        return out
    raise ValueError(f"Unknown GR variant: {variant}")

