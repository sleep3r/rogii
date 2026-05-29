"""End-to-end plane-coordinate solver.

Three solver modes, in order of sophistication:

``constant``
    ``plane_hat(MD) = plane_known[last]`` for every hidden row.

``linear``
    Fit a slope to the last ``slope_window`` rows of the known prefix and
    extrapolate the plane linearly into the hidden tail.

``linear_typewell``
    Start from ``linear`` and then apply a single global offset that maximises
    the GR-match score against the typewell reference. Useful when the linear
    extrapolation is biased (level shift).

All modes share the same interface and return the predicted ``TVT`` on every
row of the well; downstream code is responsible for masking to hidden rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    PlaneSeries,
    build_plane_series,
    extrapolate_plane_constant,
    extrapolate_plane_linear,
    plane_to_tvt,
)
from .typewell_align import TypewellRef, best_offset, locally_anchored_offset


@dataclass(frozen=True)
class PlaneSolverConfig:
    """Configuration for :func:`solve_well`."""

    mode: str = "linear_typewell"
    slope_window: int = 600
    slope_min_window: int = 100
    offset_search_radius_ft: float = 30.0
    offset_grid_size: int = 301
    # DP-mode parameters (used when ``mode == "linear_typewell_dp"``).
    dp_n_chunks: int = 8
    dp_smooth_lambda: float = 1.0
    dp_search_radius_ft: float = 30.0
    dp_grid_size: int = 201
    # Slope cap (ft of plane per ft of MD). Empirically the prefix-fit slope is
    # ~0.05–0.20 because the prefix is more vertical than the tail, so we set
    # the cap permissively. Set to ``None`` or a large value to disable.
    max_abs_slope: float | None = 0.5
    fall_back_to_linear_if_no_typewell: bool = True


def _maybe_clip_slope(plane_lin: np.ndarray, plane_const: np.ndarray, cfg: PlaneSolverConfig) -> np.ndarray:
    """If the implied slope is unreasonable, fall back to the constant plane."""

    if cfg.max_abs_slope is None:
        return plane_lin
    diff = np.diff(plane_lin)
    if diff.size == 0:
        return plane_lin
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size == 0:
        return plane_lin
    median_slope = float(np.median(finite_diff))
    if abs(median_slope) > cfg.max_abs_slope:
        return plane_const
    return plane_lin


def solve_well(
    *,
    md: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    gr: np.ndarray | None = None,
    typewell: TypewellRef | None = None,
    config: PlaneSolverConfig | None = None,
) -> dict:
    """Solve a single horizontal well and return the plane + TVT predictions.

    Returns a dict with:

    * ``"plane"`` – plane coordinate prediction, shape ``(N,)``;
    * ``"tvt"`` – TVT prediction (``plane - z``), shape ``(N,)``;
    * ``"hidden_mask"`` – ``True`` where ``tvt_input`` is missing, shape ``(N,)``;
    * ``"mode_used"`` – string label of the solver mode actually executed
      (may differ from ``config.mode`` after fallbacks);
    * ``"offset_applied_ft"`` – constant offset added by typewell anchoring
      (``0.0`` if not applied).
    """

    cfg = config or PlaneSolverConfig()
    series = build_plane_series(md, z, tvt_input)

    plane_const = extrapolate_plane_constant(series)

    if cfg.mode == "constant":
        plane = plane_const
        mode_used = "constant"
    else:
        plane_lin = extrapolate_plane_linear(
            series, window=cfg.slope_window, min_window=cfg.slope_min_window
        )
        plane_lin = _maybe_clip_slope(plane_lin, plane_const, cfg)
        if cfg.mode == "linear":
            plane = plane_lin
            mode_used = "linear"
        elif cfg.mode == "linear_typewell":
            if typewell is None or gr is None:
                if not cfg.fall_back_to_linear_if_no_typewell:
                    raise ValueError("linear_typewell mode requires typewell and gr")
                plane = plane_lin
                mode_used = "linear"
            else:
                hidden_mask = ~series.known_mask
                gr_arr = np.asarray(gr, dtype=np.float64)
                z_arr = series.z
                # We allow the offset to be fit on *hidden* rows: typewell tells
                # us the formation column, and we have GR for those rows even
                # though TVT is missing.
                offset, _scores = best_offset(
                    plane_baseline=plane_lin,
                    z=z_arr,
                    gr_h=gr_arr,
                    typewell=typewell,
                    search_radius_ft=cfg.offset_search_radius_ft,
                    n_grid=cfg.offset_grid_size,
                    eval_mask=hidden_mask,
                )
                plane = plane_lin.copy()
                # Apply the offset only to hidden rows; the known prefix is
                # ground truth and must not move.
                plane[hidden_mask] = plane[hidden_mask] + offset
                mode_used = "linear_typewell"
        elif cfg.mode == "linear_typewell_dp":
            if typewell is None or gr is None:
                if not cfg.fall_back_to_linear_if_no_typewell:
                    raise ValueError("linear_typewell_dp mode requires typewell and gr")
                plane = plane_lin
                mode_used = "linear"
                offset = 0.0
            else:
                hidden_mask = ~series.known_mask
                gr_arr = np.asarray(gr, dtype=np.float64)
                z_arr = series.z
                offsets_per_row = locally_anchored_offset(
                    plane_baseline=plane_lin,
                    z=z_arr,
                    gr_h=gr_arr,
                    typewell=typewell,
                    hidden_mask=hidden_mask,
                    search_radius_ft=cfg.dp_search_radius_ft,
                    n_grid=cfg.dp_grid_size,
                    n_chunks=cfg.dp_n_chunks,
                    smooth_lambda=cfg.dp_smooth_lambda,
                )
                plane = plane_lin + offsets_per_row
                mode_used = "linear_typewell_dp"
                offset = float(np.nanmean(offsets_per_row[hidden_mask])) if hidden_mask.any() else 0.0
        elif cfg.mode == "constant_typewell_dp":
            if typewell is None or gr is None:
                if not cfg.fall_back_to_linear_if_no_typewell:
                    raise ValueError("constant_typewell_dp mode requires typewell and gr")
                plane = plane_const
                mode_used = "constant"
                offset = 0.0
            else:
                hidden_mask = ~series.known_mask
                gr_arr = np.asarray(gr, dtype=np.float64)
                z_arr = series.z
                offsets_per_row = locally_anchored_offset(
                    plane_baseline=plane_const,
                    z=z_arr,
                    gr_h=gr_arr,
                    typewell=typewell,
                    hidden_mask=hidden_mask,
                    # Larger radius since we start from a constant plane.
                    search_radius_ft=max(cfg.dp_search_radius_ft, 100.0),
                    n_grid=cfg.dp_grid_size,
                    n_chunks=cfg.dp_n_chunks,
                    smooth_lambda=cfg.dp_smooth_lambda,
                )
                plane = plane_const + offsets_per_row
                mode_used = "constant_typewell_dp"
                offset = float(np.nanmean(offsets_per_row[hidden_mask])) if hidden_mask.any() else 0.0
        else:
            raise ValueError(f"Unknown mode '{cfg.mode}'")

    tvt_pred = plane_to_tvt(plane, series.z)
    hidden_mask = ~series.known_mask

    return {
        "plane": plane,
        "tvt": tvt_pred,
        "hidden_mask": hidden_mask,
        "mode_used": mode_used,
        "offset_applied_ft": float(0.0 if mode_used == "constant" or mode_used == "linear" else offset),
    }


def predict_tvt(
    *,
    md: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    gr: np.ndarray | None = None,
    typewell: TypewellRef | None = None,
    config: PlaneSolverConfig | None = None,
) -> np.ndarray:
    """Convenience wrapper returning just the TVT prediction (full length)."""

    return solve_well(
        md=md,
        z=z,
        tvt_input=tvt_input,
        gr=gr,
        typewell=typewell,
        config=config,
    )["tvt"]
