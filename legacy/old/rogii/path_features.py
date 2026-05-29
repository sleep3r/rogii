"""Direct path solver outputs as features for the GBM stack.

Schema16 feature pack. This module runs ``fit_geo_candidate``,
``cem_path_search`` and ``stage12_path`` per well and exposes the result as
numeric features that LightGBM/CatBoost can learn to weigh per well.

We deliberately do **not** ship raw cross-well prior outputs as features: the
773-well train-eval at schema15 + cross-well prior showed cross-well's
weighted RMSE is `16-31 ft` (vs `~1 ft` for geo_consensus and `~4 ft` for
CEM), and the systematic offset against the anchor on matched triples was
`+9.6 ft`. That family is noise, not new signal.

Fold safety: every direct solver call uses ``TVT_input``, which is NaN in
hidden rows for both train and test wells. Current-well formation columns
(``ANCC``/``ASTNU``/...) are train-only annotations, so they are ignored by
default. They may be enabled only through an explicit research flag.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .constants import FORMATIONS
from .direct_solver import (
    _ENERGY_VARIANT,
    clip_path_steps,
    fit_geo_candidate,
    fit_gr_calibration,
    fit_linear_candidate,
    known_tail_mask,
    read_typewell,
    robust_line_slope,
    score_candidate_path,
)
from .path_solver_extras import (
    EnergyContext,
    cem_path_search,
    stage12_path,
)
from .runlog import RunLogger


DEFAULT_PATH_FEATURE_NAMES: tuple[str, ...] = (
    # Raw path predictions (per row, np.nan in pre-hidden region)
    "kg_path_geo_consensus_tvt",
    "kg_path_geo_best_tvt",
    "kg_path_cem_raw_tvt",
    "kg_path_cem_top_median_tvt",
    "kg_path_stage1_tvt",
    "kg_path_stage12_tvt",
    # Anchor-relative deltas (per row)
    "kg_path_geo_consensus_minus_last",
    "kg_path_geo_consensus_minus_flat",
    "kg_path_cem_raw_minus_last",
    "kg_path_cem_top_median_minus_last",
    "kg_path_stage12_minus_last",
    # Disagreement signals (per row)
    "kg_path_geo_consensus_minus_best",
    "kg_path_cem_minus_geo",
    "kg_path_stage12_minus_geo",
    "kg_path_stage12_minus_stage1",
    "kg_path_cem_minus_stage12",
    # Per-well scalars (replicated to all rows)
    "kg_path_cem_best_offset",
    "kg_path_cem_best_slope_offset",
    "kg_path_cem_best_curvature",
    "kg_path_cem_best_score",
    "kg_path_stage1_best_a",
    "kg_path_stage1_best_b",
    "kg_path_stage1_best_score",
    "kg_path_stage2_accepted",
    "kg_path_stage2_max_offset_used",
    "kg_path_stage2_score",
    "kg_path_geo_n_surfaces",
    "kg_path_geo_rmse_min",
    "kg_path_geo_rmse_spread",
    "kg_path_gr_cal_a",
    "kg_path_gr_cal_b",
    "kg_path_gr_cal_rmse",
    "kg_path_tail_slope",
)


def empty_path_features(n: int) -> dict[str, np.ndarray]:
    return {name: np.full(n, np.nan, dtype=float) for name in DEFAULT_PATH_FEATURE_NAMES}


def _full_scalar(n: int, hidden_idx: np.ndarray, value: float) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    if len(hidden_idx):
        out[hidden_idx] = float(value)
    return out


def _full_array(n: int, hidden_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    if len(hidden_idx) and len(values):
        out[hidden_idx] = values
    return out


def _safe_finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def build_direct_path_features(
    df: pd.DataFrame,
    horizontal_path: Path,
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    flat_pred: np.ndarray,
    config: dict[str, Any],
    logger: RunLogger | None = None,
) -> dict[str, np.ndarray]:
    """Compute direct solver path families and return them as features.

    The schema16 caller passes the same inputs ``build_kaggle_top_signal_features``
    receives, plus the horizontal CSV path so we can locate the typewell.

    Returns a dict of ``{feature_name: np.ndarray of length n}``. Pre-hidden
    rows are filled with NaN exactly like the other ``kg_*`` blocks, so
    tree models route them via the missing branch.
    """
    n = len(df)
    features = empty_path_features(n)

    path_cfg = config.get("features", {}).get("direct_path", {}) or {}
    tail_rows = int(path_cfg.get("tail_rows", 384))
    cem_n_iter = int(path_cfg.get("cem_n_iter", 4))
    cem_pop_size = int(path_cfg.get("cem_pop_size", 200))
    cem_top_k = int(path_cfg.get("cem_top_k", 5))
    cem_seed = int(path_cfg.get("cem_seed", 17))
    stage2_n_knots = int(path_cfg.get("stage2_n_knots", 8))
    stage2_max_offset = float(path_cfg.get("stage2_max_offset", 12.0))
    stage2_passes = int(path_cfg.get("stage2_passes", 1))
    allow_current_well_formations = bool(
        path_cfg.get("allow_current_well_formations", False)
    )
    solver_df = (
        df
        if allow_current_well_formations
        else df.drop(columns=list(FORMATIONS), errors="ignore")
    )

    tvt_input_arr = np.asarray(tvt_input, dtype=float)
    known = np.isfinite(tvt_input_arr)
    known_idx = np.flatnonzero(known)
    hidden_idx = np.flatnonzero(~known)
    if len(known_idx) == 0 or len(hidden_idx) == 0:
        return features

    last_idx = int(known_idx[-1])
    last_tvt = float(tvt_input_arr[last_idx])
    tail_mask = known_tail_mask(tvt_input_arr, last_idx, tail_rows)
    tail_slope = robust_line_slope(md[tail_mask], tvt_input_arr[tail_mask], default=0.0)

    started_at = perf_counter()
    try:
        linear_path = fit_linear_candidate(
            solver_df,
            md,
            z,
            x,
            y,
            tvt_input_arr,
            hidden_idx,
            last_idx,
            last_tvt,
            tail_rows,
        )
        geo_best, geo_consensus, geo_diag = fit_geo_candidate(
            solver_df,
            md,
            z,
            tvt_input_arr,
            hidden_idx,
            last_idx,
            last_tvt,
            tail_rows,
        )
    except Exception as exc:
        if logger is not None:
            logger.warn(
                "direct path features: geo fit failed",
                well=horizontal_path.name,
                error=exc,
            )
        return features

    typewell = None
    try:
        typewell = read_typewell(horizontal_path)
    except Exception as exc:
        if logger is not None:
            logger.warn(
                "direct path features: typewell read failed",
                well=horizontal_path.name,
                error=exc,
            )
    cal_a, cal_b, cal_rmse = fit_gr_calibration(typewell, tvt_input_arr, gr, tail_mask)

    # Energy uses geo_consensus as the structural base, identical to the
    # production direct_solver. No anchor pull.
    energy_ctx = EnergyContext(
        md=md,
        gr=gr,
        z=z,
        typewell=typewell,
        hidden_indices=hidden_idx,
        last_idx=last_idx,
        last_tvt=last_tvt,
        tail_slope=float(tail_slope),
        cal_a=float(cal_a),
        cal_b=float(cal_b),
        linear_path=linear_path,
        geo_path=geo_consensus,
        anchor_path=geo_consensus,
    )

    def _energy_fn(path: np.ndarray) -> float:
        clipped = clip_path_steps(path, md, hidden_idx, last_idx, last_tvt, abs(tail_slope))
        return score_candidate_path(
            "_features", clipped, _ENERGY_VARIANT, typewell, gr, hidden_idx,
            md, linear_path, geo_consensus, None, cal_a, cal_b, tail_slope,
        )

    cem_outputs: dict[str, np.ndarray] = {}
    cem_diag: dict[str, object] = {}
    try:
        cem_outputs, cem_diag = cem_path_search(
            energy_ctx, _energy_fn,
            n_iter=cem_n_iter, pop_size=cem_pop_size,
            top_k=cem_top_k, seed=cem_seed,
        )
    except Exception as exc:
        if logger is not None:
            logger.warn(
                "direct path features: CEM failed",
                well=horizontal_path.name,
                error=exc,
            )

    stage12_outputs: dict[str, np.ndarray] = {}
    stage12_diag: dict[str, float] = {}
    try:
        stage12_outputs, stage12_diag = stage12_path(
            energy_ctx, _energy_fn,
            n_knots=stage2_n_knots, max_offset=stage2_max_offset, passes=stage2_passes,
        )
    except Exception as exc:
        if logger is not None:
            logger.warn(
                "direct path features: stage12 failed",
                well=horizontal_path.name,
                error=exc,
            )

    def _clip(path: np.ndarray) -> np.ndarray:
        return clip_path_steps(path, md, hidden_idx, last_idx, last_tvt, abs(tail_slope))

    geo_consensus_clipped = _clip(geo_consensus)
    geo_best_clipped = _clip(geo_best)
    cem_raw = _clip(cem_outputs["cem_raw"]) if "cem_raw" in cem_outputs else geo_consensus_clipped
    cem_top_median = _clip(cem_outputs["cem_top_median"]) if "cem_top_median" in cem_outputs else cem_raw
    stage1 = _clip(stage12_outputs["stage1_path"]) if "stage1_path" in stage12_outputs else geo_consensus_clipped
    stage12 = _clip(stage12_outputs["stage12_path"]) if "stage12_path" in stage12_outputs else stage1

    # Raw predictions
    features["kg_path_geo_consensus_tvt"] = _full_array(
        n, hidden_idx, geo_consensus_clipped[hidden_idx]
    )
    features["kg_path_geo_best_tvt"] = _full_array(
        n, hidden_idx, geo_best_clipped[hidden_idx]
    )
    features["kg_path_cem_raw_tvt"] = _full_array(n, hidden_idx, cem_raw[hidden_idx])
    features["kg_path_cem_top_median_tvt"] = _full_array(
        n, hidden_idx, cem_top_median[hidden_idx]
    )
    features["kg_path_stage1_tvt"] = _full_array(n, hidden_idx, stage1[hidden_idx])
    features["kg_path_stage12_tvt"] = _full_array(n, hidden_idx, stage12[hidden_idx])

    # Anchor-relative deltas
    features["kg_path_geo_consensus_minus_last"] = _full_array(
        n, hidden_idx, geo_consensus_clipped[hidden_idx] - last_tvt
    )
    features["kg_path_geo_consensus_minus_flat"] = _full_array(
        n, hidden_idx, geo_consensus_clipped[hidden_idx] - flat_pred[hidden_idx]
    )
    features["kg_path_cem_raw_minus_last"] = _full_array(
        n, hidden_idx, cem_raw[hidden_idx] - last_tvt
    )
    features["kg_path_cem_top_median_minus_last"] = _full_array(
        n, hidden_idx, cem_top_median[hidden_idx] - last_tvt
    )
    features["kg_path_stage12_minus_last"] = _full_array(
        n, hidden_idx, stage12[hidden_idx] - last_tvt
    )

    # Pairwise disagreements
    features["kg_path_geo_consensus_minus_best"] = _full_array(
        n, hidden_idx, geo_consensus_clipped[hidden_idx] - geo_best_clipped[hidden_idx]
    )
    features["kg_path_cem_minus_geo"] = _full_array(
        n, hidden_idx, cem_raw[hidden_idx] - geo_consensus_clipped[hidden_idx]
    )
    features["kg_path_stage12_minus_geo"] = _full_array(
        n, hidden_idx, stage12[hidden_idx] - geo_consensus_clipped[hidden_idx]
    )
    features["kg_path_stage12_minus_stage1"] = _full_array(
        n, hidden_idx, stage12[hidden_idx] - stage1[hidden_idx]
    )
    features["kg_path_cem_minus_stage12"] = _full_array(
        n, hidden_idx, cem_raw[hidden_idx] - stage12[hidden_idx]
    )

    # Per-well scalars
    features["kg_path_cem_best_offset"] = _full_scalar(
        n, hidden_idx, _safe_finite(cem_diag.get("cem_best_offset", float("nan")))
    )
    features["kg_path_cem_best_slope_offset"] = _full_scalar(
        n, hidden_idx, _safe_finite(cem_diag.get("cem_best_slope_offset", float("nan")))
    )
    features["kg_path_cem_best_curvature"] = _full_scalar(
        n, hidden_idx, _safe_finite(cem_diag.get("cem_best_curvature", float("nan")))
    )
    features["kg_path_cem_best_score"] = _full_scalar(
        n, hidden_idx, _safe_finite(cem_diag.get("cem_best_score", float("nan")))
    )
    features["kg_path_stage1_best_a"] = _full_scalar(
        n, hidden_idx, _safe_finite(stage12_diag.get("stage1_best_a", float("nan")))
    )
    features["kg_path_stage1_best_b"] = _full_scalar(
        n, hidden_idx, _safe_finite(stage12_diag.get("stage1_best_b", float("nan")))
    )
    features["kg_path_stage1_best_score"] = _full_scalar(
        n, hidden_idx, _safe_finite(stage12_diag.get("stage1_best_score", float("nan")))
    )
    features["kg_path_stage2_accepted"] = _full_scalar(
        n, hidden_idx, _safe_finite(stage12_diag.get("stage2_accepted", float("nan")))
    )
    features["kg_path_stage2_max_offset_used"] = _full_scalar(
        n, hidden_idx, _safe_finite(stage12_diag.get("stage2_max_offset_used", float("nan")))
    )
    features["kg_path_stage2_score"] = _full_scalar(
        n, hidden_idx, _safe_finite(stage12_diag.get("stage2_score", float("nan")))
    )
    features["kg_path_geo_n_surfaces"] = _full_scalar(
        n, hidden_idx, _safe_finite(geo_diag.get("geo_consensus_surfaces", float("nan")))
    )
    features["kg_path_geo_rmse_min"] = _full_scalar(
        n, hidden_idx, _safe_finite(geo_diag.get("geo_consensus_rmse_min", float("nan")))
    )
    features["kg_path_geo_rmse_spread"] = _full_scalar(
        n, hidden_idx, _safe_finite(geo_diag.get("geo_consensus_rmse_spread", float("nan")))
    )
    features["kg_path_gr_cal_a"] = _full_scalar(n, hidden_idx, _safe_finite(cal_a))
    features["kg_path_gr_cal_b"] = _full_scalar(n, hidden_idx, _safe_finite(cal_b))
    features["kg_path_gr_cal_rmse"] = _full_scalar(n, hidden_idx, _safe_finite(cal_rmse))
    features["kg_path_tail_slope"] = _full_scalar(n, hidden_idx, _safe_finite(tail_slope))

    duration = perf_counter() - started_at
    if logger is not None and bool(config.get("features", {}).get("profile_stages", False)):
        logger.info(
            "Feature stage",
            stage="well.direct_path",
            duration_sec=duration,
            features=len(features),
        )

    return features


__all__ = ["build_direct_path_features", "empty_path_features", "DEFAULT_PATH_FEATURE_NAMES"]
