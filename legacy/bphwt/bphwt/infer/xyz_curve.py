from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class XYZCurveConfig:
    lambda_smooth: float = 20.0
    lambda_base: float = 0.02
    lambda_anchor: float = 1.0e6
    prior_weight: float = 1.0


def smooth_residual_curve(
    md: np.ndarray,
    residual_prior: np.ndarray,
    known_mask: np.ndarray,
    cfg: XYZCurveConfig | None = None,
) -> np.ndarray:
    """Fit a smooth residual curve around an XYZ residual prior.

    The baseline from ``TVT_input`` already reproduces known anchors, therefore
    known rows correspond to residual zero.
    """
    c = cfg or XYZCurveConfig()
    prior = np.asarray(residual_prior, dtype=np.float64)
    n = int(prior.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    known = np.asarray(known_mask, dtype=bool)
    if known.shape != prior.shape:
        known = np.zeros_like(prior, dtype=bool)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)

    weights = np.full(n, max(float(c.prior_weight), 0.0), dtype=np.float64)
    diag = weights + max(float(c.lambda_base), 0.0)
    rhs = weights * prior

    anchor_lambda = max(float(c.lambda_anchor), 0.0)
    if anchor_lambda > 0.0 and known.any():
        diag[known] += anchor_lambda

    A = sparse.diags(diag, offsets=0, shape=(n, n), format="csr")
    smooth = max(float(c.lambda_smooth), 0.0)
    if smooth > 0.0 and n >= 3:
        D2 = _second_difference_matrix(n)
        A = A + smooth * (D2.T @ D2)

    try:
        out = spsolve(A, rhs).astype(np.float64)
    except Exception:
        out = prior / np.maximum(1.0 + max(float(c.lambda_base), 0.0), 1.0e-9)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if known.any() and anchor_lambda >= 1.0e5:
        out[known] = 0.0
    return out.astype(np.float64)


def _second_difference_matrix(n: int) -> sparse.csr_matrix:
    rows = max(int(n) - 2, 0)
    if rows == 0:
        return sparse.csr_matrix((0, n), dtype=np.float64)
    data = np.vstack(
        [
            np.ones(rows, dtype=np.float64),
            -2.0 * np.ones(rows, dtype=np.float64),
            np.ones(rows, dtype=np.float64),
        ]
    )
    offsets = np.array([0, 1, 2])
    return sparse.diags(data, offsets, shape=(rows, n), format="csr")


__all__ = ["XYZCurveConfig", "smooth_residual_curve"]
