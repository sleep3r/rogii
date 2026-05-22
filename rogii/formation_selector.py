from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FORMATIONS
from .formation_plane_knn import (
    CANDIDATE_COLUMNS,
    _rmse,
    json_safe,
    markdown_table,
    oracle_scores,
    write_frame,
)


ROBUST_WEIGHTS: dict[str, float] = {
    "pseudo_hidden_rmse": 1.00,
    "late_anchor_fit_rmse": 0.30,
    "abs_anchor_bias": 0.15,
    "anchor_slope_error": 0.15,
    "b_std": 0.10,
    "surface_std": 0.10,
    "roughness": 0.05,
    "abs_endpoint_shift_vs_schema10": 0.05,
}
HARD_SELECTORS: dict[str, str] = {
    "full_anchor_min": "anchor_fit_rmse",
    "late_anchor_min": "late_anchor_fit_rmse",
    "pseudo_hidden_min": "pseudo_hidden_rmse",
    "robust_score": "robust_score",
}
SOFT_TOP_K: tuple[int, ...] = (3, 5)
SOFT_TEMPERATURES: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
SAFE_CLIPS: tuple[float, ...] = (15.0, 20.0, 30.0, 40.0)
SAFE_ALPHA_HIGH: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0)
SAFE_ALPHA_MID: tuple[float, ...] = (0.2, 0.3, 0.4)
SAFE_ALPHA_LOW: tuple[float, ...] = (0.0, 0.1, 0.2)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _safe_first(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns or group.empty:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite[0]) if len(finite) else float("nan")


def _mean_existing(group: pd.DataFrame, columns: list[str]) -> float:
    values: list[float] = []
    for column in columns:
        if column not in group.columns:
            continue
        arr = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite):
            values.append(float(np.nanmean(finite)))
    return float(np.nanmean(values)) if values else float("nan")


def _roughness(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return float("nan")
    second = np.diff(finite, n=2)
    return float(np.sqrt(np.mean(second * second))) if len(second) else 0.0


def _parse_candidate(name: str) -> dict[str, Any]:
    if name.startswith("tvtF_"):
        parts = name.split("_")
        if len(parts) >= 3:
            return {
                "formation": parts[1],
                "knn_type": "plane",
                "bias_mode": parts[2],
                "bootstrap_id": np.nan,
            }
    if name == "row_ancc_tvt":
        return {"formation": "ANCC", "knn_type": "row", "bias_mode": "wls", "bootstrap_id": np.nan}
    if name == "dense_ancc_tvt":
        return {"formation": "ANCC", "knn_type": "dense", "bias_mode": "wls", "bootstrap_id": np.nan}
    if name == "formation_median_tvt":
        return {"formation": "ALL", "knn_type": "plane_median", "bias_mode": "full", "bootstrap_id": np.nan}
    if name == "formation_wls_median_tvt":
        return {"formation": "ALL", "knn_type": "plane_median", "bias_mode": "wls", "bootstrap_id": np.nan}
    if name.startswith("nearby_path_"):
        return {"formation": "nearby", "knn_type": "nearby_path", "bias_mode": "", "bootstrap_id": np.nan}
    if name.startswith("formation_sample_"):
        return {
            "formation": "sample",
            "knn_type": "bootstrap",
            "bias_mode": name.removeprefix("formation_sample_"),
            "bootstrap_id": np.nan,
        }
    return {"formation": "", "knn_type": "unknown", "bias_mode": "", "bootstrap_id": np.nan}


def _formation_for_candidate(name: str) -> str | None:
    parsed = _parse_candidate(name)
    formation = str(parsed["formation"])
    return formation if formation in FORMATIONS else None


def _schema_column_from_file(frame: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in frame.columns:
        return requested
    for candidate in ("schema10_oof_raw", "prediction", "pred", "TVT", "tvt"):
        if candidate in frame.columns:
            return candidate
    id_like = {"id", "ID", "well_id", "row_idx"}
    numeric = [
        column
        for column in frame.columns
        if column not in id_like and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric:
        raise ValueError("Could not infer schema10 prediction column.")
    return numeric[0]


def attach_schema10(
    frame: pd.DataFrame,
    schema10_oof: Path | None,
    *,
    schema10_column: str | None = None,
) -> tuple[pd.DataFrame, bool]:
    if "schema10_oof_raw" in frame.columns and np.isfinite(_numeric(frame, "schema10_oof_raw")).any():
        return frame, True
    if schema10_oof is None:
        if "schema10_oof_raw" not in frame.columns:
            frame = frame.copy()
            frame["schema10_oof_raw"] = np.nan
        return frame, False

    schema = _read_frame(schema10_oof)
    id_col = "id" if "id" in schema.columns else ("ID" if "ID" in schema.columns else schema.columns[0])
    pred_col = _schema_column_from_file(schema, schema10_column)
    if id_col not in schema.columns or pred_col not in schema.columns:
        raise ValueError(f"Invalid schema10 file columns: {schema10_oof}")
    schema = schema[[id_col, pred_col]].rename(
        columns={id_col: "id", pred_col: "schema10_oof_raw"}
    )
    out = frame.drop(columns=["schema10_oof_raw"], errors="ignore").merge(
        schema, on="id", how="left"
    )
    return out, bool(np.isfinite(_numeric(out, "schema10_oof_raw")).any())


def _candidate_static_stats(group: pd.DataFrame, candidate: str) -> dict[str, float]:
    formation = _formation_for_candidate(candidate)
    all_surface_std = [f"S_hat_{formation}_std" for formation in FORMATIONS]
    all_residual = [f"S_hat_{formation}_plane_residual" for formation in FORMATIONS]
    if formation:
        surface_std = _mean_existing(group, [f"S_hat_{formation}_std"])
        plane_residual = _mean_existing(group, [f"S_hat_{formation}_plane_residual"])
        b_full = _safe_first(group, f"b_{formation}_full")
        b_late = _safe_first(group, f"b_{formation}_late")
        b_wls = _safe_first(group, f"b_{formation}_wls")
        b_std = _safe_first(group, f"b_{formation}_std")
        b_late_minus_full = _safe_first(group, f"b_{formation}_late_minus_full")
    elif candidate.startswith("nearby_path_"):
        surface_std = _mean_existing(group, ["nearby_path_std", "surface_candidates_std"])
        plane_residual = _mean_existing(group, all_residual)
        b_full = b_late = b_wls = float("nan")
        b_std = _mean_existing(group, [f"b_{formation}_std" for formation in FORMATIONS])
        b_late_minus_full = _mean_existing(
            group, [f"b_{formation}_late_minus_full" for formation in FORMATIONS]
        )
    else:
        surface_std = _mean_existing(group, all_surface_std + ["surface_candidates_std"])
        plane_residual = _mean_existing(group, all_residual)
        b_full = _mean_existing(group, [f"b_{formation}_full" for formation in FORMATIONS])
        b_late = _mean_existing(group, [f"b_{formation}_late" for formation in FORMATIONS])
        b_wls = _mean_existing(group, [f"b_{formation}_wls" for formation in FORMATIONS])
        b_std = _mean_existing(group, [f"b_{formation}_std" for formation in FORMATIONS])
        b_late_minus_full = _mean_existing(
            group, [f"b_{formation}_late_minus_full" for formation in FORMATIONS]
        )

    pred = _numeric(group, candidate)
    schema = _numeric(group, "schema10_oof_raw")
    endpoint_shift = float("nan")
    mask = np.isfinite(pred) & np.isfinite(schema)
    if mask.any():
        endpoint_shift = float(pred[mask][-1] - schema[mask][-1])

    return {
        "surface_std": surface_std,
        "plane_fit_residual": plane_residual,
        "neighbor_dist_mean": _mean_existing(group, ["neighbor_dist_mean"]),
        "neighbor_dist_min": _mean_existing(group, ["neighbor_dist_min"]),
        "b_full": b_full,
        "b_late": b_late,
        "b_wls": b_wls,
        "b_std": b_std,
        "b_late_minus_full": b_late_minus_full,
        "roughness": _safe_first(group, f"roughness__{candidate}"),
        "finite_frac": _safe_first(group, f"finite_frac__{candidate}"),
        "endpoint_shift_vs_schema10": endpoint_shift,
    }


def build_candidate_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = [column for column in CANDIDATE_COLUMNS if column in frame.columns]
    for well, group in frame.groupby("well_id", sort=False):
        for candidate in candidates:
            parsed = _parse_candidate(candidate)
            stats = _candidate_static_stats(group, candidate)
            pred = _numeric(group, candidate)
            if not np.isfinite(stats["roughness"]):
                stats["roughness"] = _roughness(pred)
            if not np.isfinite(stats["finite_frac"]):
                stats["finite_frac"] = float(np.isfinite(pred).mean())
            pseudo_col = f"pseudo_hidden_rmse__{candidate}"
            pseudo_source = "pseudo_hidden"
            pseudo_hidden_rmse = _safe_first(group, pseudo_col)
            if not np.isfinite(pseudo_hidden_rmse):
                pseudo_hidden_rmse = _safe_first(group, f"late_anchor_fit_rmse__{candidate}")
                pseudo_source = "late_anchor_fallback"
            rows.append(
                {
                    "well_id": well,
                    "candidate_name": candidate,
                    **parsed,
                    **stats,
                    "anchor_fit_rmse": _safe_first(group, f"anchor_fit_rmse__{candidate}"),
                    "late_anchor_fit_rmse": _safe_first(
                        group, f"late_anchor_fit_rmse__{candidate}"
                    ),
                    "anchor_bias": _safe_first(group, f"anchor_bias__{candidate}"),
                    "anchor_slope_error": _safe_first(
                        group, f"anchor_slope_error__{candidate}"
                    ),
                    "pseudo_hidden_rmse": pseudo_hidden_rmse,
                    "pseudo_hidden_bias": _safe_first(
                        group, f"pseudo_hidden_bias__{candidate}"
                    ),
                    "pseudo_hidden_slope_error": _safe_first(
                        group, f"pseudo_hidden_slope_error__{candidate}"
                    ),
                    "pseudo_hidden_source": pseudo_source,
                }
            )
    metadata = pd.DataFrame(rows)
    return add_robust_scores(metadata)


def _normalize_component(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.full(len(arr), 10.0, dtype=float)
    med = float(np.nanmedian(finite))
    q25, q75 = np.nanpercentile(finite, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-9:
        mad = float(np.nanmedian(np.abs(finite - med)))
        scale = mad if mad > 1e-9 else 1.0
    norm = (arr - med) / scale
    finite_norm = norm[np.isfinite(norm)]
    penalty = float(np.nanmax(finite_norm) + 5.0) if len(finite_norm) else 10.0
    return np.where(np.isfinite(norm), norm, penalty)


def add_robust_scores(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return metadata
    out = metadata.copy()
    out["abs_anchor_bias"] = np.abs(_numeric(out, "anchor_bias"))
    out["abs_endpoint_shift_vs_schema10"] = np.abs(_numeric(out, "endpoint_shift_vs_schema10"))
    out["robust_score"] = 0.0
    for component, weight in ROBUST_WEIGHTS.items():
        norm_col = f"norm_{component}"
        out[norm_col] = (
            out.groupby("well_id", group_keys=False)[component]
            .transform(lambda values: pd.Series(_normalize_component(values), index=values.index))
            .astype(float)
        )
        out["robust_score"] += float(weight) * out[norm_col]
    out = out.sort_values(["well_id", "robust_score", "candidate_name"]).reset_index(drop=True)
    out["robust_rank"] = out.groupby("well_id").cumcount() + 1
    gap_by_well: dict[Any, float] = {}
    for well, group in out.groupby("well_id", sort=False):
        scores = group["robust_score"].to_numpy(dtype=float)
        gap_by_well[well] = float(scores[1] - scores[0]) if len(scores) > 1 else float("nan")
    out["score_gap"] = out["well_id"].map(gap_by_well).astype(float)
    return out


def _selector_choice(metadata: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _well, group in metadata.groupby("well_id", sort=False):
        sort_metric = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
        penalty = np.where(np.isfinite(sort_metric), sort_metric, np.inf)
        order = np.lexsort((group["candidate_name"].astype(str).to_numpy(), penalty))
        rows.append(group.iloc[int(order[0])])
    return pd.DataFrame(rows).reset_index(drop=True)


def _weighted_nanmean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(matrix)
    weighted = np.where(finite, matrix * weights[None, :], 0.0)
    denom = np.where(finite, weights[None, :], 0.0).sum(axis=1)
    return np.divide(weighted.sum(axis=1), denom, out=np.full(matrix.shape[0], np.nan), where=denom > 0)


def build_selector_outputs(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_columns = [
        column
        for column in (
            "id",
            "well_id",
            "fold",
            "row_idx",
            "TVT",
            "GR",
            "hidden_frac",
            "hidden_rows",
            "schema10_oof_raw",
        )
        if column in frame.columns
    ]
    predictions = frame[base_columns].copy()
    choices: list[dict[str, Any]] = []
    selector_names: list[str] = []

    for selector_name, metric in HARD_SELECTORS.items():
        selected = _selector_choice(metadata, metric)
        selector_names.append(selector_name)
        predictions[selector_name] = np.nan
        for row in selected.itertuples(index=False):
            well = row.well_id
            candidate = row.candidate_name
            mask = frame["well_id"].to_numpy() == well
            predictions.loc[mask, selector_name] = frame.loc[mask, candidate].to_numpy(dtype=float)
            choices.append(
                {
                    "selector_name": selector_name,
                    "well_id": well,
                    "selected_candidate": candidate,
                    "blend_candidates": candidate,
                    "blend_weights": "1.0",
                    "selector_metric": metric,
                    "selector_metric_value": getattr(row, metric),
                    "robust_score": row.robust_score,
                    "score_gap": row.score_gap,
                    "pseudo_hidden_rmse": row.pseudo_hidden_rmse,
                    "confidence_band": confidence_band(row.pseudo_hidden_rmse, row.score_gap),
                }
            )

    for top_k in SOFT_TOP_K:
        for temperature in SOFT_TEMPERATURES:
            selector_name = f"top{top_k}_soft_t{temperature:g}"
            selector_names.append(selector_name)
            predictions[selector_name] = np.nan
            for well, group in metadata.groupby("well_id", sort=False):
                ranked = group.sort_values(["robust_score", "candidate_name"]).head(top_k)
                scores = ranked["robust_score"].to_numpy(dtype=float)
                finite_scores = np.where(np.isfinite(scores), scores, np.nanmax(scores[np.isfinite(scores)]) if np.isfinite(scores).any() else 0.0)
                centered = finite_scores - float(np.nanmin(finite_scores))
                weights = np.exp(-centered / max(float(temperature), 1e-6))
                weights = weights / max(float(weights.sum()), 1e-12)
                candidates = ranked["candidate_name"].astype(str).tolist()
                mask = frame["well_id"].to_numpy() == well
                matrix = frame.loc[mask, candidates].to_numpy(dtype=float)
                predictions.loc[mask, selector_name] = _weighted_nanmean(matrix, weights)
                top = ranked.iloc[0]
                choices.append(
                    {
                        "selector_name": selector_name,
                        "well_id": well,
                        "selected_candidate": str(top["candidate_name"]),
                        "blend_candidates": ",".join(candidates),
                        "blend_weights": json.dumps([float(w) for w in weights]),
                        "selector_metric": "robust_score_softblend",
                        "selector_metric_value": float(top["robust_score"]),
                        "robust_score": float(top["robust_score"]),
                        "score_gap": float(top["score_gap"]),
                        "pseudo_hidden_rmse": float(top["pseudo_hidden_rmse"]),
                        "confidence_band": confidence_band(
                            float(top["pseudo_hidden_rmse"]), float(top["score_gap"])
                        ),
                    }
                )

    return predictions, pd.DataFrame(choices), selector_names


def confidence_band(pseudo_hidden_rmse: float, score_gap: float) -> str:
    if np.isfinite(pseudo_hidden_rmse) and pseudo_hidden_rmse < 6.0 and np.isfinite(score_gap) and score_gap > 2.0:
        return "high"
    if np.isfinite(pseudo_hidden_rmse) and pseudo_hidden_rmse < 10.0:
        return "mid"
    if np.isfinite(pseudo_hidden_rmse) and pseudo_hidden_rmse < 15.0:
        return "low"
    return "off"


def _alpha_for_band(band: np.ndarray, high: float, mid: float, low: float) -> np.ndarray:
    alpha = np.zeros(len(band), dtype=float)
    alpha[band == "high"] = float(high)
    alpha[band == "mid"] = float(mid)
    alpha[band == "low"] = float(low)
    return alpha


@dataclass(frozen=True)
class ScoreContext:
    y: np.ndarray
    well_codes: np.ndarray
    well_names: np.ndarray
    n_wells: int
    long_mask: np.ndarray
    short_mask: np.ndarray
    gr_nan: np.ndarray


def make_score_context(frame: pd.DataFrame) -> ScoreContext:
    well_codes, well_names = pd.factorize(frame["well_id"], sort=False)
    counts = np.bincount(well_codes, minlength=len(well_names)).astype(float)
    hidden_counts = counts[well_codes]
    long_mask = hidden_counts > np.nanmedian(hidden_counts)
    gr_nan = (
        ~np.isfinite(_numeric(frame, "GR"))
        if "GR" in frame.columns
        else np.zeros(len(frame), dtype=bool)
    )
    return ScoreContext(
        y=_numeric(frame, "TVT"),
        well_codes=well_codes.astype(int),
        well_names=well_names.astype(object),
        n_wells=int(len(well_names)),
        long_mask=long_mask,
        short_mask=~long_mask,
        gr_nan=gr_nan,
    )


def _well_rmse(pred: np.ndarray, ctx: ScoreContext) -> np.ndarray:
    mask = np.isfinite(pred) & np.isfinite(ctx.y)
    if not mask.any():
        return np.full(ctx.n_wells, np.nan, dtype=float)
    err = pred[mask] - ctx.y[mask]
    sse = np.bincount(
        ctx.well_codes[mask], weights=err * err, minlength=ctx.n_wells
    ).astype(float)
    counts = np.bincount(ctx.well_codes[mask], minlength=ctx.n_wells).astype(float)
    return np.sqrt(
        np.divide(sse, counts, out=np.full(ctx.n_wells, np.nan), where=counts > 0)
    )


def score_prediction(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    name: str,
    context: ScoreContext | None = None,
) -> dict[str, Any]:
    ctx = context or make_score_context(frame)
    pred = np.asarray(prediction, dtype=float)
    well_scores = _well_rmse(pred, ctx)
    finite_wells = well_scores[np.isfinite(well_scores)]
    worst_well = ""
    worst_value = float("nan")
    if len(finite_wells):
        worst_idx = int(np.nanargmax(well_scores))
        worst_well = str(ctx.well_names[worst_idx])
        worst_value = float(well_scores[worst_idx])
    return {
        "selector_name": name,
        "rmse": _rmse(pred, ctx.y),
        "mean_well_rmse": float(np.nanmean(well_scores)),
        "p50_well_rmse": float(np.nanpercentile(finite_wells, 50)) if len(finite_wells) else float("nan"),
        "p90_well_rmse": float(np.nanpercentile(finite_wells, 90)) if len(finite_wells) else float("nan"),
        "p95_well_rmse": float(np.nanpercentile(finite_wells, 95)) if len(finite_wells) else float("nan"),
        "worst_well_rmse": worst_value,
        "worst_well": worst_well,
        "long_hidden_rmse": _rmse(pred[ctx.long_mask], ctx.y[ctx.long_mask]) if ctx.long_mask.any() else float("nan"),
        "short_hidden_rmse": _rmse(pred[ctx.short_mask], ctx.y[ctx.short_mask]) if ctx.short_mask.any() else float("nan"),
        "gr_nan_rmse": _rmse(pred[ctx.gr_nan], ctx.y[ctx.gr_nan]) if ctx.gr_nan.any() else float("nan"),
        "finite_frac": float(np.isfinite(pred).mean()),
    }


def score_selectors(predictions: pd.DataFrame, selector_names: list[str]) -> pd.DataFrame:
    context = make_score_context(predictions)
    rows = [
        score_prediction(
            predictions,
            predictions[selector].to_numpy(dtype=float),
            selector,
            context,
        )
        for selector in selector_names
    ]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def score_safe_blends(
    predictions: pd.DataFrame,
    choices: pd.DataFrame,
    selector_names: list[str],
) -> pd.DataFrame:
    if "schema10_oof_raw" not in predictions.columns:
        return pd.DataFrame()
    schema = _numeric(predictions, "schema10_oof_raw")
    if not np.isfinite(schema).any():
        return pd.DataFrame()
    band_lookup = {
        (row.selector_name, row.well_id): row.confidence_band
        for row in choices.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    wells = predictions["well_id"].to_numpy()
    context = make_score_context(predictions)
    for selector_name in selector_names:
        selector_pred = _numeric(predictions, selector_name)
        band = np.array(
            [band_lookup.get((selector_name, well), "off") for well in wells],
            dtype=object,
        )
        for clip, high, mid, low in itertools.product(
            SAFE_CLIPS, SAFE_ALPHA_HIGH, SAFE_ALPHA_MID, SAFE_ALPHA_LOW
        ):
            alpha = _alpha_for_band(band, high, mid, low)
            raw_delta = selector_pred - schema
            delta = np.where(
                np.isfinite(raw_delta),
                np.clip(raw_delta, -float(clip), float(clip)),
                0.0,
            )
            safe_pred = schema + alpha * delta
            row = score_prediction(
                predictions,
                safe_pred,
                f"safe__{selector_name}__clip{clip:g}__hi{high:g}_mid{mid:g}_low{low:g}",
                context,
            )
            shift = np.abs(safe_pred - schema)
            finite_shift = shift[np.isfinite(shift)]
            row.update(
                {
                    "base_selector": selector_name,
                    "clip": float(clip),
                    "alpha_high": float(high),
                    "alpha_mid": float(mid),
                    "alpha_low": float(low),
                    "median_shift_vs_schema10": float(np.nanmedian(finite_shift)) if len(finite_shift) else float("nan"),
                    "p95_shift_vs_schema10": float(np.nanpercentile(finite_shift, 95)) if len(finite_shift) else float("nan"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def oracle_rankings(frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[tuple[str, int], list[str]]]:
    candidates = [column for column in CANDIDATE_COLUMNS if column in frame.columns]
    whole: dict[str, list[str]] = {}
    thirds: dict[tuple[str, int], list[str]] = {}
    for well, group in frame.groupby("well_id", sort=False):
        y = _numeric(group, "TVT")
        scores = {
            candidate: _rmse(_numeric(group, candidate), y)
            for candidate in candidates
        }
        whole[well] = [
            candidate
            for candidate, _score in sorted(
                scores.items(), key=lambda item: (np.inf if not np.isfinite(item[1]) else item[1], item[0])
            )
        ]
        local = np.arange(len(group))
        for segment_id, segment in enumerate(np.array_split(local, 3), start=1):
            segment_group = group.iloc[segment]
            seg_y = _numeric(segment_group, "TVT")
            seg_scores = {
                candidate: _rmse(_numeric(segment_group, candidate), seg_y)
                for candidate in candidates
            }
            thirds[(well, segment_id)] = [
                candidate
                for candidate, _score in sorted(
                    seg_scores.items(),
                    key=lambda item: (np.inf if not np.isfinite(item[1]) else item[1], item[0]),
                )
            ]
    return whole, thirds


def hit_rates(frame: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    whole, thirds = oracle_rankings(frame)
    rows: list[dict[str, Any]] = []
    for selector, group in choices.groupby("selector_name", sort=False):
        top1 = 0
        top3 = 0
        seg_top1 = 0
        seg_top3 = 0
        seg_count = 0
        for row in group.itertuples(index=False):
            ranking = whole.get(row.well_id, [])
            candidate = str(row.selected_candidate)
            top1 += int(bool(ranking) and candidate == ranking[0])
            top3 += int(candidate in ranking[:3])
            for segment_id in (1, 2, 3):
                seg_ranking = thirds.get((row.well_id, segment_id), [])
                if not seg_ranking:
                    continue
                seg_count += 1
                seg_top1 += int(candidate == seg_ranking[0])
                seg_top3 += int(candidate in seg_ranking[:3])
        n = max(len(group), 1)
        rows.append(
            {
                "selector_name": selector,
                "whole_top1_hit_rate": top1 / n,
                "whole_top3_hit_rate": top3 / n,
                "segment_top1_hit_rate": seg_top1 / max(seg_count, 1),
                "segment_top3_hit_rate": seg_top3 / max(seg_count, 1),
                "wells": int(len(group)),
                "segments": int(seg_count),
            }
        )
    return pd.DataFrame(rows)


def oracle_regret(
    selector_scores: pd.DataFrame,
    oracle_frame: pd.DataFrame,
    *,
    whole_anchor: float,
    thirds_anchor: float,
) -> pd.DataFrame:
    whole = whole_anchor
    thirds = thirds_anchor
    if not oracle_frame.empty and "oracle" in oracle_frame.columns:
        actual_whole = oracle_frame.loc[
            oracle_frame["oracle"] == "whole_well_oracle", "rmse"
        ]
        actual_thirds = oracle_frame.loc[
            oracle_frame["oracle"] == "thirds_segment_oracle", "rmse"
        ]
        if len(actual_whole):
            whole = float(actual_whole.iloc[0])
        if len(actual_thirds):
            thirds = float(actual_thirds.iloc[0])
    out = selector_scores[["selector_name", "rmse"]].copy()
    out["whole_well_oracle_rmse"] = whole
    out["thirds_segment_oracle_rmse"] = thirds
    out["regret_whole"] = out["rmse"] - whole
    out["regret_thirds"] = out["rmse"] - thirds
    return out.sort_values("regret_whole").reset_index(drop=True)


def bad_well_report(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    selector_names: list[str],
) -> pd.DataFrame:
    candidates = [column for column in CANDIDATE_COLUMNS if column in frame.columns]
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=False):
        pred_group = predictions[predictions["well_id"] == well]
        y = _numeric(group, "TVT")
        schema_rmse = _rmse(_numeric(group, "schema10_oof_raw"), y)
        candidate_scores = {
            candidate: _rmse(_numeric(group, candidate), y)
            for candidate in candidates
        }
        oracle_candidate = min(
            candidate_scores,
            key=lambda c: np.inf if not np.isfinite(candidate_scores[c]) else candidate_scores[c],
        )
        selector_scores = {
            selector: _rmse(_numeric(pred_group, selector), _numeric(pred_group, "TVT"))
            for selector in selector_names
        }
        best_selector = min(
            selector_scores,
            key=lambda s: np.inf if not np.isfinite(selector_scores[s]) else selector_scores[s],
        )
        robust_rmse = selector_scores.get("robust_score", float("nan"))
        rows.append(
            {
                "well_id": well,
                "schema10_rmse": schema_rmse,
                "a_oracle_best_candidate": oracle_candidate,
                "a_oracle_rmse": candidate_scores[oracle_candidate],
                "best_selector": best_selector,
                "best_selector_rmse": selector_scores[best_selector],
                "robust_selector_rmse": robust_rmse,
                "schema_bad_a_oracle_good": bool(
                    np.isfinite(schema_rmse)
                    and np.isfinite(candidate_scores[oracle_candidate])
                    and schema_rmse > 20.0
                    and candidate_scores[oracle_candidate] + 5.0 < schema_rmse
                ),
                "robust_selector_catastrophic": bool(
                    np.isfinite(robust_rmse)
                    and ((np.isfinite(schema_rmse) and robust_rmse > schema_rmse + 10.0) or robust_rmse > 30.0)
                ),
                "all_a_candidates_fail": bool(
                    np.isfinite(candidate_scores[oracle_candidate])
                    and candidate_scores[oracle_candidate] > 20.0
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["robust_selector_catastrophic", "schema_bad_a_oracle_good", "schema10_rmse"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_report(
    *,
    output_dir: Path,
    rows: int,
    wells: int,
    schema_available: bool,
    selector_scores: pd.DataFrame,
    safe_scores: pd.DataFrame,
    regret: pd.DataFrame,
    hits: pd.DataFrame,
    bad_wells: pd.DataFrame,
    oracle_frame: pd.DataFrame,
) -> None:
    best_selector = selector_scores.iloc[0].to_dict() if not selector_scores.empty else {}
    best_safe = safe_scores.iloc[0].to_dict() if not safe_scores.empty else {}
    lines = [
        "# A2 Formation Selector Report",
        f"Rows: `{rows}`",
        f"Wells: `{wells}`",
        f"Schema10 safe blend: `{'available' if schema_available else 'unavailable'}`",
        "## Selector Scores",
        markdown_table(selector_scores),
        "## Safe Blend Scores",
        markdown_table(safe_scores) if not safe_scores.empty else "_schema10 unavailable_",
        "## Oracles",
        markdown_table(oracle_frame),
        "## Oracle Regret",
        markdown_table(regret),
        "## Hit Rate",
        markdown_table(hits),
        "## Bad Wells",
        markdown_table(bad_wells),
        "## Decision Snapshot",
        f"Best selector: `{best_selector.get('selector_name', '')}` RMSE `{best_selector.get('rmse', float('nan'))}`",
        f"Best safe blend: `{best_safe.get('selector_name', '')}` RMSE `{best_safe.get('rmse', float('nan'))}`",
    ]
    (output_dir / "A2_SELECTOR_REPORT.md").write_text("\n\n".join(lines), encoding="utf-8")


def run_selector(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _read_frame(Path(args.input))
    frame, schema_available = attach_schema10(
        frame, Path(args.schema10_oof) if args.schema10_oof else None, schema10_column=args.schema10_column
    )
    metadata = build_candidate_metadata(frame)
    predictions, choices, selector_names = build_selector_outputs(frame, metadata)
    selector_scores = score_selectors(predictions, selector_names)
    safe_scores = score_safe_blends(predictions, choices, selector_names)
    oracle_frame, _oracle_winners = oracle_scores(frame)
    regret = oracle_regret(
        selector_scores,
        oracle_frame,
        whole_anchor=float(args.whole_oracle_anchor),
        thirds_anchor=float(args.thirds_oracle_anchor),
    )
    hits = hit_rates(frame, choices)
    bad_wells = bad_well_report(frame, predictions, selector_names)

    write_frame(predictions, output_dir / "selector_predictions.parquet")
    write_frame(choices, output_dir / "selector_choices.parquet")
    write_frame(metadata, output_dir / "candidate_metadata.parquet")
    selector_scores.to_csv(output_dir / "selector_scores.csv", index=False)
    safe_scores.to_csv(output_dir / "safe_blend_scores.csv", index=False)
    regret.to_csv(output_dir / "oracle_regret.csv", index=False)
    hits.to_csv(output_dir / "hit_rate.csv", index=False)
    bad_wells.to_csv(output_dir / "bad_wells.csv", index=False)
    write_report(
        output_dir=output_dir,
        rows=len(frame),
        wells=int(frame["well_id"].nunique()),
        schema_available=schema_available,
        selector_scores=selector_scores,
        safe_scores=safe_scores,
        regret=regret,
        hits=hits,
        bad_wells=bad_wells,
        oracle_frame=oracle_frame,
    )
    metrics = {
        "input": str(args.input),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "schema10_available": bool(schema_available),
        "best_selector": selector_scores.iloc[0].to_dict() if not selector_scores.empty else {},
        "best_safe_blend": safe_scores.iloc[0].to_dict() if not safe_scores.empty else {},
        "oracles": oracle_frame.to_dict("records"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(metrics), file, indent=2)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-free A2 selector for FormationPlaneKNN candidates.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/formation_plane_knn/oof_candidates.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formation_selector"))
    parser.add_argument("--schema10-oof", type=Path, default=None)
    parser.add_argument("--schema10-column", type=str, default=None)
    parser.add_argument("--whole-oracle-anchor", type=float, default=8.27952)
    parser.add_argument("--thirds-oracle-anchor", type=float, default=6.89153)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = run_selector(parse_args(argv))
    print(json.dumps(json_safe(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
