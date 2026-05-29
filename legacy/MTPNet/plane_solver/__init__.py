"""Plane-coordinate solver for the ROGII Wellbore Geology Prediction competition.

Built around the algebraic identity discovered in Kaggle discussion 699853 (msg
3462931): ``TVT(MD) + Z(MD)`` collapses to a formation-top scalar that is

* known exactly on the input prefix (``TVT_input + Z``),
* piecewise-linear in ``MD`` with ~22 control points per well (StarSteer dip
  annotations).

The solver predicts the hidden tail of ``TVT`` by extrapolating the plane
coordinate ``c(MD) = TVT(MD) + Z(MD)`` and re-applying ``TVT = c - Z``.
"""

from .core import (
    PlaneSeries,
    build_plane_series,
    extrapolate_plane_constant,
    extrapolate_plane_linear,
)
from .solver import PlaneSolverConfig, solve_well, predict_tvt
from .typewell_align import TypewellRef, build_typewell_ref, score_plane_offsets, locally_anchored_offset

__all__ = [
    "PlaneSeries",
    "build_plane_series",
    "extrapolate_plane_constant",
    "extrapolate_plane_linear",
    "PlaneSolverConfig",
    "solve_well",
    "predict_tvt",
    "TypewellRef",
    "build_typewell_ref",
    "score_plane_offsets",
    "locally_anchored_offset",
]
