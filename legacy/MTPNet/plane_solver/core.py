"""Plane-coordinate primitives.

The "plane coordinate" of a horizontal well row is

    c(MD) := TVT(MD) + Z(MD)

By the algebraic relation TVT = -Z + marker + const (verified to ~0.006 ft RMS
across train wells; see Kaggle discussion 699853), ``c`` equals one of the
formation marker depths plus a per-well constant. It is therefore

* piecewise-linear in ``MD`` with ~22 slope changes per well,
* known exactly on the input prefix where ``TVT_input`` is provided.

This module contains pure numpy helpers around that coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneSeries:
    """Plane-coordinate series for a single horizontal well.

    Attributes
    ----------
    md
        Measured depth, shape ``(N,)``.
    z
        Z coordinate, shape ``(N,)``.
    plane_known
        Plane coordinate ``c = TVT_input + Z`` where defined, else ``NaN``.
        Shape ``(N,)``.
    known_mask
        Boolean mask, ``True`` where ``plane_known`` is finite.
    last_known_idx
        Index of the last finite prefix row, or ``-1`` if the prefix is empty.
    """

    md: np.ndarray
    z: np.ndarray
    plane_known: np.ndarray
    known_mask: np.ndarray
    last_known_idx: int

    @property
    def n(self) -> int:
        return int(self.md.shape[0])


def build_plane_series(md: np.ndarray, z: np.ndarray, tvt_input: np.ndarray) -> PlaneSeries:
    """Build a :class:`PlaneSeries` from the three columns available at test time."""

    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    tvt_input = np.asarray(tvt_input, dtype=np.float64)

    if md.shape != z.shape or md.shape != tvt_input.shape:
        raise ValueError(
            f"md, z, tvt_input must share a shape; got {md.shape}, {z.shape}, {tvt_input.shape}"
        )

    plane_known = tvt_input + z  # NaN propagates where TVT_input is missing
    known_mask = np.isfinite(plane_known)

    if known_mask.any():
        last_known_idx = int(np.nonzero(known_mask)[0].max())
    else:
        last_known_idx = -1

    return PlaneSeries(
        md=md,
        z=z,
        plane_known=plane_known,
        known_mask=known_mask,
        last_known_idx=last_known_idx,
    )


def extrapolate_plane_constant(series: PlaneSeries) -> np.ndarray:
    """Constant-plane extrapolation: ``c_hat = c(last_known)`` everywhere ahead.

    Equivalent to assuming the formation top is locally flat in MD from the last
    observed prefix row onwards.
    """

    n = series.n
    out = np.copy(series.plane_known)
    if series.last_known_idx < 0:
        # No prefix – fall back to zero plane (TVT_pred = -Z).
        out[:] = 0.0
        return out

    last_val = series.plane_known[series.last_known_idx]
    # Fill all rows (including any NaN gaps before the prefix end) with last_val.
    fill_mask = ~np.isfinite(out)
    out[fill_mask] = last_val
    # Hidden tail is everything strictly after last_known_idx.
    out[series.last_known_idx + 1 :] = last_val
    return out


def _robust_linear_fit(md: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``c ≈ a*md + b`` using only finite points."""

    finite = np.isfinite(md) & np.isfinite(c)
    if finite.sum() < 2:
        return 0.0, float(np.nanmean(c)) if np.isfinite(c).any() else 0.0
    x = md[finite]
    y = c[finite]
    # Centre for stability.
    x0 = x.mean()
    a, b0 = np.polyfit(x - x0, y, deg=1)
    b = b0 - a * x0
    return float(a), float(b)


def extrapolate_plane_linear(
    series: PlaneSeries,
    *,
    window: int = 600,
    min_window: int = 100,
) -> np.ndarray:
    """Locally-linear extrapolation of the plane coordinate.

    Estimates the local dip ``d(plane)/d(MD)`` from the last ``window`` rows of
    the known prefix and extrapolates linearly into the hidden tail. Continuity
    at the prefix boundary is enforced by anchoring the line to
    ``plane_known[last_known_idx]``.
    """

    n = series.n
    out = np.copy(series.plane_known)
    if series.last_known_idx < 0:
        return extrapolate_plane_constant(series)

    last = series.last_known_idx
    start = max(0, last - window + 1)
    if last - start + 1 < min_window:
        # Too little prefix to fit; fall back to constant.
        return extrapolate_plane_constant(series)

    md_win = series.md[start : last + 1]
    c_win = series.plane_known[start : last + 1]
    a, _ = _robust_linear_fit(md_win, c_win)

    anchor_md = series.md[last]
    anchor_val = series.plane_known[last]

    # Fill known prefix gaps with linear projection too (rare).
    fill_mask = ~np.isfinite(out)
    if fill_mask.any():
        out[fill_mask] = anchor_val + a * (series.md[fill_mask] - anchor_md)

    tail_slice = slice(last + 1, n)
    out[tail_slice] = anchor_val + a * (series.md[tail_slice] - anchor_md)
    return out


def plane_to_tvt(plane: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Recover TVT from the plane coordinate: ``TVT = plane - Z``."""

    return np.asarray(plane, dtype=np.float64) - np.asarray(z, dtype=np.float64)
