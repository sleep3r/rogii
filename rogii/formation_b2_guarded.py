from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formation_b_lite import _prediction_from_candidate_map
from .formation_b2_constrained import _read_frame, _safe_median, _safe_percentile
from .formation_plane_knn import json_safe, markdown_table, write_frame
from .formation_selector import attach_schema10


DEFAULT_SELECTORS: tuple[str, ...] = (
    "A_among_B_top10",
    "B_among_A_top10",
    "A_among_B_top20",
    "B_among_A_top20",
    "rank_0_7A_0_3B",
    "rank_A_plus_B",
)
FIXED_SPECS: tuple[tuple[float, float], ...] = (
    (0.30, 30.0),
    (0.30, 15.0),
    (0.20, 30.0),
    (0.20, 15.0),
)


@dataclass(frozen=True)
class EvalContext:
    y: np.ndarray
    base: np.ndarray
    well_codes: np.ndarray
    well_names: np.ndarray
    folds: np.ndarray


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _make_context(frame: pd.DataFrame) -> EvalContext:
    well_codes, well_names = pd.factorize(frame["well_id"], sort=False)
    folds = _numeric(frame, "fold", default=0.0).astype(int)
    return EvalContext(
        y=_numeric(frame, "TVT"),
        base=_numeric(frame, "schema10_oof_raw"),
        well_codes=well_codes.astype(int),
        well_names=well_names.astype(object),
        folds=folds,
    )


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    err = pred[mask] - y[mask]
    return float(np.sqrt(np.mean(err * err)))


def _score(pred: np.ndarray, ctx: EvalContext, *, mask: np.ndarray | None = None) -> dict[str, Any]:
    row_mask = np.ones(len(pred), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    finite = row_mask & np.isfinite(pred) & np.isfinite(ctx.y)
    err = pred[finite] - ctx.y[finite]
    if not finite.any():
        return {
            "rmse": float("nan"),
            "mean_well_rmse": float("nan"),
            "p50_well_rmse": float("nan"),
            "p90_well_rmse": float("nan"),
            "p95_well_rmse": float("nan"),
            "worst_well_rmse": float("nan"),
            "worst_well": "",
        }
    n_wells = int(ctx.well_codes.max()) + 1
    sse = np.bincount(ctx.well_codes[finite], weights=err * err, minlength=n_wells)
    counts = np.bincount(ctx.well_codes[finite], minlength=n_wells)
    well_rmse = np.sqrt(np.divide(sse, counts, out=np.full(n_wells, np.nan), where=counts > 0))
    visible = well_rmse[np.isfinite(well_rmse)]
    worst_idx = int(np.nanargmax(well_rmse))
    return {
        "rmse": _rmse(pred[row_mask], ctx.y[row_mask]),
        "mean_well_rmse": float(np.nanmean(visible)),
        "p50_well_rmse": float(np.nanpercentile(visible, 50)),
        "p90_well_rmse": float(np.nanpercentile(visible, 90)),
        "p95_well_rmse": float(np.nanpercentile(visible, 95)),
        "worst_well_rmse": float(well_rmse[worst_idx]),
        "worst_well": str(ctx.well_names[worst_idx]),
    }


def _well_rmse(values: np.ndarray, y: np.ndarray) -> float:
    return _rmse(values, y)


def _roughness(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(finite, n=2) ** 2)))


def _slope_p95(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return float("nan")
    return _safe_percentile(np.abs(np.diff(finite)), 95)


def _lookup_series(meta: pd.DataFrame, column: str) -> dict[tuple[Any, str], float]:
    if column not in meta.columns:
        return {}
    return {
        (row.well_id, row.candidate_name): float(getattr(row, column))
        for row in meta[["well_id", "candidate_name", column]].itertuples(index=False)
    }


def _choice_for_selector(choices: pd.DataFrame, selector: str) -> pd.DataFrame:
    choice = choices[choices["selector_name"] == selector].copy()
    if choice.empty:
        raise ValueError(f"Selector not found in choices: {selector}")
    return choice.sort_values("well_id").drop_duplicates("well_id", keep="first")


def _candidate_map(choice: pd.DataFrame) -> dict[Any, str]:
    return dict(zip(choice["well_id"], choice["candidate_name"], strict=False))


def _score_gap_maps(meta: pd.DataFrame) -> tuple[dict[Any, float], dict[Any, float]]:
    gap2: dict[Any, float] = {}
    gap5: dict[Any, float] = {}
    for well, group in meta.groupby("well_id", sort=False):
        scores = pd.to_numeric(group["b_combined_score"], errors="coerce").to_numpy(dtype=float)
        scores = np.sort(scores[np.isfinite(scores)])
        if len(scores) >= 2:
            gap2[well] = float(scores[1] - scores[0])
        if len(scores) >= 5:
            gap5[well] = float(scores[4] - scores[0])
    return gap2, gap5


def selector_diagnostics(
    frame: pd.DataFrame,
    meta: pd.DataFrame,
    choices: pd.DataFrame,
    selector: str,
    selected: np.ndarray,
    ctx: EvalContext,
) -> pd.DataFrame:
    choice = _choice_for_selector(choices, selector)
    hidden_rmse = _lookup_series(meta, "hidden_rmse")
    b_score = _lookup_series(meta, "b_combined_score")
    a_rank = _lookup_series(meta, "a_rank")
    b_rank = _lookup_series(meta, "b_rank")
    late_anchor = _lookup_series(meta, "late_anchor_rmse")
    surface_std = _lookup_series(meta, "surface_std")
    b_std = _lookup_series(meta, "b_std")
    roughness_meta = _lookup_series(meta, "roughness")
    if "hidden_rmse" in meta.columns:
        oracle = (
            meta.sort_values(["well_id", "hidden_rmse", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
        )
        oracle_rmse = dict(zip(oracle["well_id"], oracle["hidden_rmse"], strict=False))
    else:
        oracle_rmse = {}
    gap2, gap5 = _score_gap_maps(meta)
    candidate_by_well = _candidate_map(choice)

    rows: list[dict[str, Any]] = []
    for well_code, well in enumerate(ctx.well_names):
        idx = np.flatnonzero(ctx.well_codes == well_code)
        candidate = candidate_by_well.get(well)
        if candidate is None:
            continue
        key = (well, candidate)
        base = ctx.base[idx]
        pred = selected[idx]
        y = ctx.y[idx]
        delta = pred - base
        finite_delta = delta[np.isfinite(delta)]
        pos = float(np.mean(finite_delta > 0.0)) if len(finite_delta) else float("nan")
        neg = float(np.mean(finite_delta < 0.0)) if len(finite_delta) else float("nan")
        rough_base = _roughness(base)
        rough_selected = roughness_meta.get(key, _roughness(pred))
        slope_base = _slope_p95(base)
        slope_selected = _slope_p95(pred)
        rows.append(
            {
                "selector_name": selector,
                "well_id": well,
                "fold": int(ctx.folds[idx[0]]) if len(idx) else -1,
                "base_rmse": _well_rmse(base, y),
                "b2_rmse_a03_clip30": _well_rmse(base + 0.30 * np.clip(delta, -30.0, 30.0), y),
                "b2_rmse_a03_clip15": _well_rmse(base + 0.30 * np.clip(delta, -15.0, 15.0), y),
                "selected_candidate": candidate,
                "selected_candidate_rmse": hidden_rmse.get(key, float("nan")),
                "oracle_A_rmse": float(oracle_rmse.get(well, np.nan)),
                "delta_abs_median": _safe_median(np.abs(delta)),
                "delta_abs_p95": _safe_percentile(np.abs(delta), 95),
                "delta_abs_max": _safe_percentile(np.abs(delta), 100),
                "delta_endpoint": float(finite_delta[-1]) if len(finite_delta) else float("nan"),
                "delta_mean_signed": _safe_median(delta),
                "delta_one_sided_frac": max(pos, neg) if np.isfinite(pos) and np.isfinite(neg) else float("nan"),
                "B_score_selected": b_score.get(key, float("nan")),
                "B_score_gap_top1_top2": gap2.get(well, float("nan")),
                "B_score_gap_top1_top5": gap5.get(well, float("nan")),
                "B_rank_of_selected": b_rank.get(key, float("nan")),
                "A_rank_of_selected": a_rank.get(key, float("nan")),
                "late_anchor_rmse": late_anchor.get(key, float("nan")),
                "surface_std": surface_std.get(key, float("nan")),
                "b_std": b_std.get(key, float("nan")),
                "roughness_selected": rough_selected,
                "roughness_base": rough_base,
                "roughness_ratio_vs_base": rough_selected / rough_base if np.isfinite(rough_base) and rough_base > 0 else float("nan"),
                "slope_p95_selected": slope_selected,
                "slope_p95_base": slope_base,
                "slope_ratio_vs_base": slope_selected / slope_base if np.isfinite(slope_base) and slope_base > 0 else float("nan"),
                "GR_nan_ratio": float(np.mean(~np.isfinite(_numeric(frame.iloc[idx], "GR")))) if len(idx) else float("nan"),
                "ncc_finite_frac": 1.0,
            }
        )
    out = pd.DataFrame(rows)
    out["delta_rmse_a03_clip30"] = out["b2_rmse_a03_clip30"] - out["base_rmse"]
    out["delta_rmse_a03_clip15"] = out["b2_rmse_a03_clip15"] - out["base_rmse"]
    out["danger_score"] = (
        (out["delta_abs_p95"] > 15.0).astype(float)
        + (out["delta_endpoint"].abs() > 20.0).astype(float)
        + (out["roughness_ratio_vs_base"] > 2.0).astype(float)
        + ((out["delta_one_sided_frac"] > 0.95) & (out["delta_abs_p95"] > 10.0)).astype(float)
        + (out["B_score_gap_top1_top2"].fillna(0.0) <= 0.10).astype(float)
    )
    return out


def _series_to_rows(diag: pd.DataFrame, ctx: EvalContext, column: str) -> np.ndarray:
    values = diag.set_index("well_id")[column].reindex(ctx.well_names).to_numpy(dtype=float)
    return values[ctx.well_codes]


def _well_stat_to_rows(values_by_well: np.ndarray, ctx: EvalContext) -> np.ndarray:
    return values_by_well[ctx.well_codes]


def _p95_abs_by_well(values: np.ndarray, ctx: EvalContext) -> np.ndarray:
    out = np.full(len(ctx.well_names), np.nan, dtype=float)
    for code in range(len(ctx.well_names)):
        out[code] = _safe_percentile(np.abs(values[ctx.well_codes == code]), 95)
    return out


def _danger_by_well(pre_shift: np.ndarray, ctx: EvalContext, diag: pd.DataFrame) -> pd.DataFrame:
    work = diag.set_index("well_id").reindex(ctx.well_names).copy()
    blend_p95 = _p95_abs_by_well(pre_shift, ctx)
    gap = pd.to_numeric(work["B_score_gap_top1_top2"], errors="coerce")
    late = pd.to_numeric(work["late_anchor_rmse"], errors="coerce")
    surface = pd.to_numeric(work["surface_std"], errors="coerce")
    gap_q25 = float(gap.quantile(0.25)) if gap.notna().any() else float("nan")
    late_q75 = float(late.quantile(0.75)) if late.notna().any() else float("nan")
    surface_q75 = float(surface.quantile(0.75)) if surface.notna().any() else float("nan")
    endpoint_abs = pd.to_numeric(work["delta_endpoint"], errors="coerce").abs().to_numpy(dtype=float)
    one_sided = pd.to_numeric(work["delta_one_sided_frac"], errors="coerce").to_numpy(dtype=float)
    rough_ratio = pd.to_numeric(work["roughness_ratio_vs_base"], errors="coerce").to_numpy(dtype=float)
    ncc = pd.to_numeric(work["ncc_finite_frac"], errors="coerce").to_numpy(dtype=float)
    danger = np.zeros(len(work), dtype=float)
    danger += (blend_p95 > 6.0).astype(float)
    danger += (endpoint_abs > 12.0).astype(float)
    danger += ((one_sided > 0.95) & (blend_p95 > 4.0)).astype(float)
    danger += (rough_ratio > 2.0).astype(float)
    if np.isfinite(gap_q25):
        danger += (gap.to_numpy(dtype=float) <= gap_q25).astype(float)
    if np.isfinite(late_q75):
        danger += (late.to_numpy(dtype=float) >= late_q75).astype(float)
    if np.isfinite(surface_q75):
        danger += (surface.to_numpy(dtype=float) >= surface_q75).astype(float)
    danger += (np.nan_to_num(ncc, nan=1.0) < 0.7).astype(float)
    work["policy_blend_delta_abs_p95"] = blend_p95
    work["policy_danger_score"] = danger
    work["policy_one_sided_huge"] = (
        (one_sided > 0.97) & (blend_p95 > 5.0) & (endpoint_abs > 10.0)
    )
    return work[["policy_blend_delta_abs_p95", "policy_danger_score", "policy_one_sided_huge"]]


def _apply_policy(
    base: np.ndarray,
    raw_delta: np.ndarray,
    ctx: EvalContext,
    diag: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = policy["mode"]
    if mode == "fixed":
        alpha = np.full(len(raw_delta), float(policy["alpha"]), dtype=float)
        clip = np.full(len(raw_delta), float(policy["clip"]), dtype=float)
    elif mode in {"danger_current", "danger_clip15_boost"}:
        raw_p95 = _series_to_rows(diag, ctx, "delta_abs_p95")
        endpoint = np.abs(_series_to_rows(diag, ctx, "delta_endpoint"))
        alpha = np.full(len(raw_delta), float(policy.get("alpha", 0.30)), dtype=float)
        clip_high = float(policy.get("clip_high", policy.get("clip", 30.0)))
        clip_low = float(policy.get("clip_low", 15.0))
        delta_p95 = float(policy.get("delta_p95_downgrade", 20.0))
        endpoint_limit = float(policy.get("endpoint_downgrade", 25.0))
        if mode == "danger_current":
            clip = np.where((raw_p95 > delta_p95) | (endpoint > endpoint_limit), clip_low, clip_high)
        else:
            clip = np.full(len(raw_delta), clip_low, dtype=float)
    elif mode == "downgrade_clip":
        danger = (
            (_series_to_rows(diag, ctx, "delta_abs_p95") > float(policy["delta_p95"]))
            | (np.abs(_series_to_rows(diag, ctx, "delta_endpoint")) > float(policy["endpoint"]))
        )
        alpha = np.full(len(raw_delta), 0.30, dtype=float)
        clip = np.where(danger, 15.0, 30.0)
    elif mode == "alpha_downgrade":
        p95 = _series_to_rows(diag, ctx, "delta_abs_p95")
        endpoint = np.abs(_series_to_rows(diag, ctx, "delta_endpoint"))
        rough_ratio = _series_to_rows(diag, ctx, "roughness_ratio_vs_base")
        one_sided = _series_to_rows(diag, ctx, "delta_one_sided_frac")
        gap = _series_to_rows(diag, ctx, "B_score_gap_top1_top2")
        high = (
            (p95 > 20.0)
            | (endpoint > 25.0)
            | (rough_ratio > float(policy["roughness_ratio"]))
            | ((one_sided > 0.95) & (p95 > 10.0))
        )
        medium = (
            (p95 > float(policy["delta_p95"]))
            | (endpoint > float(policy["endpoint"]))
            | (np.nan_to_num(gap, nan=0.0) <= float(policy["gap"]))
        )
        alpha = np.where(high, 0.0, np.where(medium, 0.15, 0.30))
        clip = np.full(len(raw_delta), 30.0, dtype=float)
    elif mode == "conservative_boost":
        p95 = _series_to_rows(diag, ctx, "delta_abs_p95")
        endpoint = np.abs(_series_to_rows(diag, ctx, "delta_endpoint"))
        rough_ratio = _series_to_rows(diag, ctx, "roughness_ratio_vs_base")
        gap = _series_to_rows(diag, ctx, "B_score_gap_top1_top2")
        boost = (
            (p95 <= float(policy["delta_p95"]))
            & (endpoint <= float(policy["endpoint"]))
            & (rough_ratio <= float(policy["roughness_ratio"]))
            & (np.nan_to_num(gap, nan=0.0) >= float(policy["gap"]))
        )
        alpha = np.full(len(raw_delta), 0.30, dtype=float)
        clip = np.where(boost, 30.0, 15.0)
    elif mode == "well_shift_cap":
        alpha = np.full(len(raw_delta), float(policy["alpha"]), dtype=float)
        clip = np.full(len(raw_delta), float(policy["clip"]), dtype=float)
    else:
        raise ValueError(f"Unknown policy mode: {mode}")

    shift = alpha * np.clip(raw_delta, -clip, clip)
    if mode in {"danger_current", "danger_clip15_boost"}:
        danger_frame = _danger_by_well(shift, ctx, diag)
        danger = _well_stat_to_rows(
            danger_frame["policy_danger_score"].to_numpy(dtype=float),
            ctx,
        )
        one_sided_huge = _well_stat_to_rows(
            danger_frame["policy_one_sided_huge"].to_numpy(dtype=bool),
            ctx,
        )
        if mode == "danger_clip15_boost":
            boost = danger <= float(policy["boost_max_danger"])
            clip = np.where(boost, 30.0, 15.0)
            shift = alpha * np.clip(raw_delta, -clip, clip)
            danger_frame = _danger_by_well(shift, ctx, diag)
            danger = _well_stat_to_rows(
                danger_frame["policy_danger_score"].to_numpy(dtype=float),
                ctx,
            )
            one_sided_huge = _well_stat_to_rows(
                danger_frame["policy_one_sided_huge"].to_numpy(dtype=bool),
                ctx,
            )
        action = str(policy["action"])
        factor = np.ones(len(raw_delta), dtype=float)
        if action == "kill_ge3":
            factor = np.where(danger >= 3.0, 0.0, 1.0)
        elif action == "kill_ge4":
            factor = np.where(danger >= 4.0, 0.0, 1.0)
        elif action == "half_ge2_kill_ge3":
            factor = np.where(danger >= 3.0, 0.0, np.where(danger >= 2.0, 0.5, 1.0))
        elif action == "half_ge2_kill_ge4":
            factor = np.where(danger >= 4.0, 0.0, np.where(danger >= 2.0, 0.5, 1.0))
        elif action == "one_sided_kill":
            factor = np.where(one_sided_huge, 0.0, 1.0)
        elif action == "one_sided_quarter":
            factor = np.where(one_sided_huge, 0.25, 1.0)
        elif action == "none":
            factor = np.ones(len(raw_delta), dtype=float)
        else:
            raise ValueError(f"Unknown danger action: {action}")
        alpha = alpha * factor
        shift = shift * factor
    if mode == "well_shift_cap":
        cap = float(policy["cap"])
        abs_shift = np.abs(shift)
        p95_by_well = np.zeros(len(ctx.well_names), dtype=float)
        for code in range(len(ctx.well_names)):
            idx = ctx.well_codes == code
            p95_by_well[code] = _safe_percentile(abs_shift[idx], 95)
        scale_by_well = np.divide(cap, p95_by_well, out=np.ones_like(p95_by_well), where=p95_by_well > cap)
        scale_by_well = np.minimum(1.0, scale_by_well)
        shift = shift * scale_by_well[ctx.well_codes]
        alpha = alpha * scale_by_well[ctx.well_codes]
    pred = base + np.where(np.isfinite(shift), shift, 0.0)
    return pred, np.abs(pred - base), alpha, clip


def _alpha_summary(alpha: np.ndarray, ctx: EvalContext) -> dict[str, Any]:
    alpha_by_well = np.full(len(ctx.well_names), np.nan, dtype=float)
    for code in range(len(ctx.well_names)):
        alpha_by_well[code] = _safe_median(alpha[ctx.well_codes == code])
    disabled_wells = alpha_by_well <= 1e-9
    half_wells = (alpha_by_well > 1e-9) & (alpha_by_well < 0.299)
    well_map = dict(zip(ctx.well_names.astype(str), alpha_by_well, strict=False))
    return {
        "disabled_wells_count": int(np.nansum(disabled_wells)),
        "half_scaled_wells_count": int(np.nansum(half_wells)),
        "disabled_rows_frac": float(np.mean(alpha <= 1e-9)),
        "half_scaled_rows_frac": float(np.mean((alpha > 1e-9) & (alpha < 0.299))),
        "alpha_389ae58f": float(well_map.get("389ae58f", np.nan)),
    }


def make_policies(selector: str) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for alpha, clip in FIXED_SPECS:
        policies.append(
            {
                "policy_name": f"{selector}__fixed_a{alpha:g}_clip{clip:g}",
                "base_selector": selector,
                "mode": "fixed",
                "alpha": alpha,
                "clip": clip,
            }
        )
    for delta_p95 in (10.0, 12.0, 15.0, 20.0):
        for endpoint in (12.0, 15.0, 20.0, 25.0):
            policies.append(
                {
                    "policy_name": f"{selector}__clip_downgrade_p95{delta_p95:g}_end{endpoint:g}",
                    "base_selector": selector,
                    "mode": "downgrade_clip",
                    "delta_p95": delta_p95,
                    "endpoint": endpoint,
                }
            )
    for delta_p95 in (12.0, 15.0, 20.0):
        for endpoint in (15.0, 20.0, 25.0):
            for roughness_ratio in (1.5, 2.0):
                policies.append(
                    {
                        "policy_name": f"{selector}__alpha_down_p95{delta_p95:g}_end{endpoint:g}_rr{roughness_ratio:g}",
                        "base_selector": selector,
                        "mode": "alpha_downgrade",
                        "delta_p95": delta_p95,
                        "endpoint": endpoint,
                        "roughness_ratio": roughness_ratio,
                        "gap": 0.10,
                    }
                )
    for delta_p95 in (10.0, 12.0, 15.0):
        for endpoint in (12.0, 15.0, 20.0):
            for gap in (0.0, 0.10):
                policies.append(
                    {
                        "policy_name": f"{selector}__clip15_boost30_p95{delta_p95:g}_end{endpoint:g}_gap{gap:g}",
                        "base_selector": selector,
                        "mode": "conservative_boost",
                        "delta_p95": delta_p95,
                        "endpoint": endpoint,
                        "roughness_ratio": 2.0,
                        "gap": gap,
                    }
                )
    for cap in (4.0, 6.0, 8.0, 10.0):
        policies.append(
            {
                "policy_name": f"{selector}__well_cap{cap:g}_a0.3_clip30",
                "base_selector": selector,
                "mode": "well_shift_cap",
                "alpha": 0.30,
                "clip": 30.0,
                "cap": cap,
            }
        )
    for action in (
        "kill_ge3",
        "kill_ge4",
        "half_ge2_kill_ge3",
        "half_ge2_kill_ge4",
        "one_sided_kill",
        "one_sided_quarter",
    ):
        policies.append(
            {
                "policy_name": f"{selector}__danger_current_{action}",
                "base_selector": selector,
                "mode": "danger_current",
                "action": action,
            }
        )
    for boost_max in (0.0, 1.0):
        policies.append(
            {
                "policy_name": f"{selector}__clip15_boost30_danger_le{boost_max:g}",
                "base_selector": selector,
                "mode": "danger_clip15_boost",
                "action": "none",
                "boost_max_danger": boost_max,
            }
        )
    return policies


def evaluate_policies(
    frame: pd.DataFrame,
    choices: pd.DataFrame,
    meta: pd.DataFrame,
    selectors: list[str],
    ctx: EvalContext,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    base_score = _score(ctx.base, ctx)
    rows: list[dict[str, Any]] = []
    all_diag: list[pd.DataFrame] = []
    selected_predictions: dict[str, np.ndarray] = {}
    diagnostics_by_selector: dict[str, pd.DataFrame] = {}
    for selector in selectors:
        print(f"guard evaluate selector={selector}", flush=True)
        choice = _choice_for_selector(choices, selector)
        candidate_map = _candidate_map(choice)
        selected = _prediction_from_candidate_map(frame, candidate_map)
        raw_delta = selected - ctx.base
        diag = selector_diagnostics(frame, meta, choices, selector, selected, ctx)
        diagnostics_by_selector[selector] = diag
        all_diag.append(diag)
        selected_predictions[selector] = selected
        policies = make_policies(selector)
        for policy_idx, policy in enumerate(policies, start=1):
            if policy_idx == 1 or policy_idx % 10 == 0 or policy_idx == len(policies):
                print(
                    f"guard policy selector={selector} {policy_idx}/{len(policies)} "
                    f"name={policy['policy_name']}",
                    flush=True,
                )
            pred, shift, alpha_final, _clip_final = _apply_policy(ctx.base, raw_delta, ctx, diag, policy)
            score = _score(pred, ctx)
            score.update(
                {
                    **policy,
                    "gain": base_score["rmse"] - score["rmse"],
                    "p95_shift": _safe_percentile(shift, 95),
                    "median_shift": _safe_median(shift),
                    "worst_delta_vs_base": score["worst_well_rmse"] - base_score["worst_well_rmse"],
                    "p95_delta_vs_base": score["p95_well_rmse"] - base_score["p95_well_rmse"],
                    **_alpha_summary(alpha_final, ctx),
                }
            )
            rows.append(score)
    scores = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return scores, pd.concat(all_diag, ignore_index=True), selected_predictions, diagnostics_by_selector


def _choose_constrained(scores: pd.DataFrame, base_score: dict[str, Any]) -> pd.Series:
    strict = scores[
        (scores["p95_well_rmse"] <= 20.0)
        & (scores["worst_well_rmse"] <= base_score["worst_well_rmse"] + 1.0)
        & (scores["p95_shift"] <= 8.0)
    ]
    if not strict.empty:
        return strict.sort_values("rmse").iloc[0]
    relaxed = scores[
        (scores["p95_well_rmse"] <= base_score["p95_well_rmse"])
        & (scores["worst_well_rmse"] <= base_score["worst_well_rmse"] + 5.0)
        & (scores["p95_shift"] <= 8.0)
    ]
    if not relaxed.empty:
        return relaxed.sort_values("rmse").iloc[0]
    return scores.sort_values("rmse").iloc[0]


def _choose_submit_policy(scores: pd.DataFrame, base_score: dict[str, Any]) -> pd.Series:
    strong = scores[
        (scores["gain"] >= 0.40)
        & (scores["p95_well_rmse"] <= 20.20)
        & (scores["worst_well_rmse"] <= base_score["worst_well_rmse"] + 1.0)
        & (scores["p95_shift"] <= 5.0)
    ]
    if not strong.empty:
        return strong.sort_values("rmse").iloc[0]
    good = scores[
        (scores["gain"] >= 0.25)
        & (scores["p95_well_rmse"] <= base_score["p95_well_rmse"])
        & (scores["worst_well_rmse"] <= base_score["worst_well_rmse"] + 2.0)
        & (scores["p95_shift"] <= 6.0)
    ]
    if not good.empty:
        return good.sort_values("rmse").iloc[0]
    worst_safe = scores[scores["worst_well_rmse"] <= base_score["worst_well_rmse"] + 1.0]
    if not worst_safe.empty:
        return worst_safe.sort_values("rmse").iloc[0]
    return _choose_constrained(scores, base_score)


def _policy_dict(row: pd.Series) -> dict[str, Any]:
    data = row.dropna().to_dict()
    return {str(key): value for key, value in data.items() if key not in {"rmse", "mean_well_rmse", "p50_well_rmse", "p90_well_rmse", "p95_well_rmse", "worst_well_rmse", "worst_well", "gain", "p95_shift", "median_shift", "worst_delta_vs_base", "p95_delta_vs_base"}}


def cross_fold_check(
    scores: pd.DataFrame,
    frame: pd.DataFrame,
    selected_predictions: dict[str, np.ndarray],
    diagnostics_by_selector: dict[str, pd.DataFrame],
    ctx: EvalContext,
    base_score: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    folds = sorted(int(item) for item in np.unique(ctx.folds) if int(item) > 0)
    if not folds:
        return pd.DataFrame()
    policy_rows = [_policy_dict(row) for _idx, row in scores.iterrows()]
    for fold in folds:
        print(f"guard cross-fold fold={fold}/{folds[-1]} policies={len(policy_rows)}", flush=True)
        train_mask = ctx.folds != fold
        valid_mask = ctx.folds == fold
        train_rows = []
        pred_cache: dict[str, np.ndarray] = {}
        shift_cache: dict[str, np.ndarray] = {}
        for policy_idx, policy in enumerate(policy_rows, start=1):
            if policy_idx == 1 or policy_idx % 10 == 0 or policy_idx == len(policy_rows):
                print(
                    f"guard cross-fold fold={fold} policy={policy_idx}/{len(policy_rows)} "
                    f"name={policy['policy_name']}",
                    flush=True,
                )
            selector = str(policy["base_selector"])
            raw_delta = selected_predictions[selector] - ctx.base
            pred, shift, _alpha_final, _clip_final = _apply_policy(
                ctx.base,
                raw_delta,
                ctx,
                diagnostics_by_selector[selector],
                policy,
            )
            pred_cache[str(policy["policy_name"])] = pred
            shift_cache[str(policy["policy_name"])] = shift
            score = _score(pred, ctx, mask=train_mask)
            train_rows.append(
                {
                    "policy_name": policy["policy_name"],
                    "rmse": score["rmse"],
                    "p95_well_rmse": score["p95_well_rmse"],
                    "worst_well_rmse": score["worst_well_rmse"],
                    "p95_shift": _safe_percentile(shift[train_mask], 95),
                }
            )
        train_scores = pd.DataFrame(train_rows)
        chosen = _choose_constrained(train_scores, base_score)
        pred = pred_cache[str(chosen["policy_name"])]
        shift = shift_cache[str(chosen["policy_name"])]
        heldout = _score(pred, ctx, mask=valid_mask)
        base_heldout = _score(ctx.base, ctx, mask=valid_mask)
        rows.append(
            {
                "fold": fold,
                "selected_policy": str(chosen["policy_name"]),
                "heldout_rmse": heldout["rmse"],
                "heldout_gain": base_heldout["rmse"] - heldout["rmse"],
                "heldout_p95": heldout["p95_well_rmse"],
                "heldout_worst": heldout["worst_well_rmse"],
                "heldout_p95_shift": _safe_percentile(shift[valid_mask], 95),
                "base_heldout_rmse": base_heldout["rmse"],
            }
        )
    return pd.DataFrame(rows)


def fixed_cross_fold_check(
    policy_rows: pd.DataFrame,
    selected_predictions: dict[str, np.ndarray],
    diagnostics_by_selector: dict[str, pd.DataFrame],
    ctx: EvalContext,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    folds = sorted(int(item) for item in np.unique(ctx.folds) if int(item) > 0)
    if not folds or policy_rows.empty:
        return pd.DataFrame()
    for _idx, row in policy_rows.drop_duplicates("policy_name").iterrows():
        policy = _policy_dict(row)
        selector = str(policy["base_selector"])
        raw_delta = selected_predictions[selector] - ctx.base
        pred, shift, _alpha_final, _clip_final = _apply_policy(
            ctx.base,
            raw_delta,
            ctx,
            diagnostics_by_selector[selector],
            policy,
        )
        for fold in folds:
            mask = ctx.folds == fold
            score = _score(pred, ctx, mask=mask)
            base_score = _score(ctx.base, ctx, mask=mask)
            rows.append(
                {
                    "policy_name": policy["policy_name"],
                    "fold": fold,
                    "heldout_rmse": score["rmse"],
                    "heldout_gain": base_score["rmse"] - score["rmse"],
                    "heldout_p95": score["p95_well_rmse"],
                    "heldout_worst": score["worst_well_rmse"],
                    "heldout_p95_shift": _safe_percentile(shift[mask], 95),
                    "base_heldout_rmse": base_score["rmse"],
                }
            )
    return pd.DataFrame(rows)


def guarded_predictions(
    frame: pd.DataFrame,
    ctx: EvalContext,
    selected_predictions: dict[str, np.ndarray],
    diagnostics_by_selector: dict[str, pd.DataFrame],
    best_policy: pd.Series,
    submit_policy: pd.Series,
) -> pd.DataFrame:
    selector = str(best_policy["base_selector"])
    raw_delta = selected_predictions[selector] - ctx.base
    best_pred, best_shift, alpha_final, clip_final = _apply_policy(
        ctx.base,
        raw_delta,
        ctx,
        diagnostics_by_selector[selector],
        _policy_dict(best_policy),
    )
    submit_selector = str(submit_policy["base_selector"])
    submit_raw_delta = selected_predictions[submit_selector] - ctx.base
    submit_pred, submit_shift, submit_alpha, submit_clip = _apply_policy(
        ctx.base,
        submit_raw_delta,
        ctx,
        diagnostics_by_selector[submit_selector],
        _policy_dict(submit_policy),
    )
    conservative = ctx.base + 0.30 * np.clip(raw_delta, -15.0, 15.0)
    aggressive = ctx.base + 0.30 * np.clip(raw_delta, -30.0, 30.0)
    diag = diagnostics_by_selector[selector].set_index("well_id")
    submit_diag = diagnostics_by_selector[submit_selector].set_index("well_id")
    return pd.DataFrame(
        {
            "id": frame["id"].astype(str).to_numpy(),
            "well_id": frame["well_id"].astype(str).to_numpy(),
            "TVT": ctx.y,
            "b2_base_tvt": ctx.base,
            "b2_selected_A_path": selected_predictions[selector],
            "b2_selected_delta": raw_delta,
            "b2_safe_a03_clip30": aggressive,
            "b2_safe_a03_clip15": conservative,
            "b2_guarded_best": best_pred,
            "b2_guarded_shift": best_shift,
            "b2_blend_delta": best_pred - ctx.base,
            "b2_alpha_final": alpha_final,
            "b2_clip_final": clip_final,
            "b2_submit_selected_A_path": selected_predictions[submit_selector],
            "b2_submit_selected_delta": submit_raw_delta,
            "b2_guarded_submit": submit_pred,
            "b2_submit_guarded_shift": submit_shift,
            "b2_submit_blend_delta": submit_pred - ctx.base,
            "b2_submit_alpha_final": submit_alpha,
            "b2_submit_clip_final": submit_clip,
            "b2_submit_policy_name": np.full(len(frame), str(submit_policy["policy_name"]), dtype=object),
            "b2_danger_score": frame["well_id"].map(diag["danger_score"]).to_numpy(dtype=float),
            "b2_shift_p95": frame["well_id"].map(diag["delta_abs_p95"]).to_numpy(dtype=float),
            "b2_endpoint_shift": frame["well_id"].map(diag["delta_endpoint"]).to_numpy(dtype=float),
            "b2_submit_danger_score": frame["well_id"].map(submit_diag["danger_score"]).to_numpy(dtype=float),
            "b2_submit_shift_p95": frame["well_id"].map(submit_diag["delta_abs_p95"]).to_numpy(dtype=float),
            "b2_submit_endpoint_shift": frame["well_id"].map(submit_diag["delta_endpoint"]).to_numpy(dtype=float),
        }
    )


def write_report(
    output_dir: Path,
    *,
    base_score: dict[str, Any],
    policy_scores: pd.DataFrame,
    best_constrained: pd.Series,
    best_submit: pd.Series,
    per_well: pd.DataFrame,
    fold_check: pd.DataFrame,
    fixed_fold_check: pd.DataFrame,
) -> None:
    best_policy_name = str(best_constrained["policy_name"])
    submit_policy_name = str(best_submit["policy_name"])
    top_worsening = per_well[per_well["selector_name"] == str(best_constrained["base_selector"])].sort_values(
        "delta_rmse_a03_clip30",
        ascending=False,
    )
    focus = top_worsening[top_worsening["well_id"] == "389ae58f"]
    danger_scores = policy_scores[
        policy_scores["mode"].astype(str).str.contains("danger", na=False)
    ].sort_values("rmse")
    lines = [
        "# B2 Guarded Report",
        "## Base",
        markdown_table(pd.DataFrame([base_score])),
        "## Unguarded And Guard Policies",
        markdown_table(policy_scores.head(30)),
        "## Danger Policies",
        markdown_table(danger_scores.head(30)),
        "## Best Constrained Policy",
        markdown_table(pd.DataFrame([best_constrained.to_dict()])),
        "## Submit Policy",
        markdown_table(pd.DataFrame([best_submit.to_dict()])),
        "## Top Worsening Wells",
        markdown_table(top_worsening.head(20)),
        "## 389ae58f Diagnostics",
        markdown_table(focus),
        "## Cross-Fold Auto-Select Check",
        markdown_table(fold_check),
        "## Fixed Cross-Fold Check",
        markdown_table(fixed_fold_check),
        "## Submit Fixed Cross-Fold",
        markdown_table(fixed_fold_check[fixed_fold_check["policy_name"] == submit_policy_name]),
        "## Decision",
        f"`{submit_policy_name}` is the current submit candidate; `{best_policy_name}` is the best relaxed guarded candidate.",
    ]
    (output_dir / "B2_GUARDED_REPORT.md").write_text("\n\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"guard load input={args.input}", flush=True)
    frame = _read_frame(Path(args.input))
    print(f"guard attach schema10={args.schema10_oof}", flush=True)
    frame, baseline_available = attach_schema10(
        frame,
        Path(args.schema10_oof) if args.schema10_oof else None,
        schema10_column=args.schema10_column,
    )
    if not baseline_available:
        raise ValueError("B2 guarded safe blend requires schema10 OOF baseline.")
    print(f"guard load choices={args.choices}", flush=True)
    choices = _read_frame(Path(args.choices))
    print(f"guard load metadata={args.metadata}", flush=True)
    meta = _read_frame(Path(args.metadata))
    selectors = [item for item in str(args.selectors).split(",") if item]
    ctx = _make_context(frame)
    base_score = _score(ctx.base, ctx)
    base_score["selector_name"] = "base"

    print(f"guard evaluate policies selectors={len(selectors)}", flush=True)
    policy_scores, per_well, selected_predictions, diagnostics_by_selector = evaluate_policies(
        frame,
        choices,
        meta,
        selectors,
        ctx,
    )
    best_constrained = _choose_constrained(policy_scores, base_score)
    best_submit = _choose_submit_policy(policy_scores, base_score)
    print(
        "guard best constrained "
        f"name={best_constrained['policy_name']} rmse={best_constrained['rmse']:.6f} "
        f"p95={best_constrained['p95_well_rmse']:.6f} "
        f"worst={best_constrained['worst_well_rmse']:.6f}",
        flush=True,
    )
    print(
        "guard best submit "
        f"name={best_submit['policy_name']} rmse={best_submit['rmse']:.6f} "
        f"p95={best_submit['p95_well_rmse']:.6f} "
        f"worst={best_submit['worst_well_rmse']:.6f}",
        flush=True,
    )
    print(f"guard cross-fold limit={args.cross_fold_policy_limit}", flush=True)
    fold_check = cross_fold_check(
        policy_scores.head(int(args.cross_fold_policy_limit)),
        frame,
        selected_predictions,
        diagnostics_by_selector,
        ctx,
        base_score,
    )
    fixed_policy_rows = pd.concat(
        [
            policy_scores.head(5),
            pd.DataFrame([best_constrained]),
            pd.DataFrame([best_submit]),
            policy_scores[policy_scores["mode"].astype(str).str.contains("danger", na=False)].head(10),
            policy_scores[policy_scores["worst_well_rmse"] <= base_score["worst_well_rmse"] + 1.0].head(5),
        ],
        ignore_index=True,
    ).drop_duplicates("policy_name")
    print(f"guard fixed cross-fold policies={len(fixed_policy_rows)}", flush=True)
    fixed_fold_check = fixed_cross_fold_check(
        fixed_policy_rows,
        selected_predictions,
        diagnostics_by_selector,
        ctx,
    )
    print("guard build predictions", flush=True)
    predictions = guarded_predictions(
        frame,
        ctx,
        selected_predictions,
        diagnostics_by_selector,
        best_constrained,
        best_submit,
    )

    print(f"guard write outputs={output_dir}", flush=True)
    policy_scores.to_csv(output_dir / "guard_policy_scores.csv", index=False)
    per_well.to_csv(output_dir / "guard_per_well_diagnostics.csv", index=False)
    fold_check.to_csv(output_dir / "guard_cross_fold.csv", index=False)
    fixed_fold_check.to_csv(output_dir / "guard_fixed_cross_fold.csv", index=False)
    write_frame(predictions, output_dir / "guarded_predictions.parquet")
    write_report(
        output_dir,
        base_score=base_score,
        policy_scores=policy_scores,
        best_constrained=best_constrained,
        best_submit=best_submit,
        per_well=per_well,
        fold_check=fold_check,
        fixed_fold_check=fixed_fold_check,
    )
    submit_fixed = fixed_fold_check[fixed_fold_check["policy_name"] == str(best_submit["policy_name"])]
    metrics = {
        "input": str(args.input),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "base": base_score,
        "best_rmse_policy": policy_scores.head(1).to_dict("records"),
        "best_constrained_policy": [best_constrained.to_dict()],
        "best_submit_policy": [best_submit.to_dict()],
        "best_worst_safe_policy": policy_scores[
            policy_scores["worst_well_rmse"] <= base_score["worst_well_rmse"] + 1.0
        ].head(1).to_dict("records"),
        "best_danger_policy": policy_scores[
            policy_scores["mode"].astype(str).str.contains("danger", na=False)
        ].head(1).to_dict("records"),
        "cross_fold_mean_gain": float(fold_check["heldout_gain"].mean()) if not fold_check.empty else None,
        "fixed_cross_fold_min_gain": float(fixed_fold_check["heldout_gain"].min()) if not fixed_fold_check.empty else None,
        "submit_fixed_cross_fold_min_gain": float(submit_fixed["heldout_gain"].min()) if not submit_fixed.empty else None,
        "submit_fixed_cross_fold_mean_gain": float(submit_fixed["heldout_gain"].mean()) if not submit_fixed.empty else None,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(metrics), file, indent=2)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded B2 safe-blend policy search.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/formation_plane_knn/oof_candidates.parquet"))
    parser.add_argument("--choices", type=Path, default=Path("artifacts/formation_b2_constrained/b2_selector_choices.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/formation_b2_constrained/b2_candidate_metadata.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formation_b2_guarded"))
    parser.add_argument("--schema10-oof", type=Path, required=True)
    parser.add_argument("--schema10-column", type=str, default=None)
    parser.add_argument("--selectors", type=str, default=",".join(DEFAULT_SELECTORS))
    parser.add_argument("--cross-fold-policy-limit", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = run(parse_args(argv))
    print(json.dumps(json_safe(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
