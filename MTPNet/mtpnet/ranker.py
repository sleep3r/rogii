from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MTPConfig
from .priors import load_prior_tables
from .stitch import (
    _apply_step_predictions_to_rows,
    _baseline_metrics,
    _blend_with_anchor,
    _load_gr_context,
    _load_hidden_rows,
    _load_run_config,
    _softmax_np,
    _window_metrics_from_mode_windows,
    aggregate_mode_windows,
    attach_gr_rerank_scores,
    evaluate_with_b2_fallback,
)

CAT_FEATURES: tuple[str, ...] = ()

CONSERVATIVE_FEATURE_COLUMNS: list[str] = [
    "nn_logit",
    "nn_prob",
    "nn_rank",
    "logit_gap_to_top1",
    "mode_index",
    "mode_mean_tvt",
    "mode_std_tvt",
    "gr_corr",
    "dgr_corr",
    "ncc8",
    "ncc15",
    "ncc25",
    "gr_mad",
    "gr_finite_frac",
    "mean_abs_to_b2",
    "p95_abs_to_b2",
    "endpoint_abs_to_b2",
    "mean_abs_to_base",
    "mean_abs_to_A_p50",
    "inside_A_band_frac",
    "A_density_mean",
    "slope_mean",
    "slope_p95",
    "curvature_mean",
    "roughness",
    "window_hidden_pos_frac",
    "GR_nan_ratio",
    "B2_danger",
    "A_uncertainty",
    "base_b2_gap",
]

FEATURE_COLUMNS: list[str] = list(CONSERVATIVE_FEATURE_COLUMNS)
DEFAULT_RANKER_BETA_GRID: tuple[float, ...] = (0.25, 0.5, 1.0)

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "iterations": 4000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 6.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "od_type": "Iter",
    "od_wait": 200,
    "allow_writing_files": False,
    "verbose": False,
}


def split_ranker_wells(
    well_ids: list[str] | set[str] | tuple[str, ...],
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    wells = sorted({str(well_id) for well_id in well_ids})
    if len(wells) < 2:
        raise ValueError("MTPRanker requires at least two wells for train/valid split")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    order = np.asarray(wells, dtype=object)
    rng.shuffle(order)
    n_valid = int(round(len(order) * valid_fraction))
    n_valid = min(max(1, n_valid), len(order) - 1)
    valid = set(str(item) for item in order[:n_valid])
    train = set(str(item) for item in order[n_valid:])
    return train, valid


def make_group_folds(
    well_ids: list[str] | set[str] | tuple[str, ...],
    *,
    n_folds: int,
    seed: int,
) -> list[dict[str, list[str]]]:
    wells = sorted({str(well_id) for well_id in well_ids})
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > len(wells):
        raise ValueError("n_folds cannot exceed number of wells")
    rng = np.random.default_rng(seed)
    order = np.asarray(wells, dtype=object)
    rng.shuffle(order)
    chunks = np.array_split(order, n_folds)
    folds: list[dict[str, list[str]]] = []
    all_wells = set(wells)
    for index, chunk in enumerate(chunks):
        valid = sorted(str(item) for item in chunk.tolist())
        train = sorted(all_wells.difference(valid))
        folds.append({"fold": index, "train_wells": train, "valid_wells": valid})
    return folds


def _as_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        if arr.ndim == 1 and any(isinstance(item, (list, tuple, np.ndarray)) for item in arr):
            return np.stack([_as_float_array(item) for item in arr]).astype(np.float32)
        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _window_ids(windows: pd.DataFrame) -> list[str]:
    if "window_id" in windows.columns:
        return windows["window_id"].astype(str).tolist()
    ids: list[str] = []
    for index, row in enumerate(windows.itertuples(index=False)):
        ids.append(f"{row.well_id}:{int(row.start_step)}:{index}")
    return ids


def _ensure_window_ids(windows: pd.DataFrame) -> pd.DataFrame:
    if "window_id" in windows.columns:
        return windows.copy()
    out = windows.copy()
    out["window_id"] = _window_ids(out)
    return out


def normalize_mode_windows(mode_windows: pd.DataFrame) -> pd.DataFrame:
    out = mode_windows.copy()
    for column in ("logits", "probs", "path_tvt", "target_tvt"):
        if column in out.columns:
            out[column] = out[column].map(_as_float_array)
    return out


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)


def _mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0
    return float(np.nanmean(arr[finite]))


def _p95_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0
    return float(np.nanpercentile(np.abs(arr[finite]), 95.0))


def _mean_abs_diff(path: np.ndarray, ref: np.ndarray) -> float:
    mask = np.isfinite(path) & np.isfinite(ref)
    if not mask.any():
        return 0.0
    return float(np.nanmean(np.abs(path[mask] - ref[mask])))


def _p95_abs_diff(path: np.ndarray, ref: np.ndarray) -> float:
    mask = np.isfinite(path) & np.isfinite(ref)
    if not mask.any():
        return 0.0
    return float(np.nanpercentile(np.abs(path[mask] - ref[mask]), 95.0))


def _endpoint_abs_diff(path: np.ndarray, ref: np.ndarray) -> float:
    if len(path) == 0 or len(ref) == 0:
        return 0.0
    if not np.isfinite(path[-1]) or not np.isfinite(ref[-1]):
        return 0.0
    return float(abs(path[-1] - ref[-1]))


def _density_mean_from_quantiles(
    path: np.ndarray,
    p50: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
) -> float:
    mask = np.isfinite(path) & np.isfinite(p50)
    if not mask.any():
        return 0.0
    width = np.abs(p90 - p10)
    sigma = np.where(np.isfinite(width), np.maximum(width / 2.563, 1.0), 2.0)
    z = (path - p50) / sigma
    density = np.exp(-0.5 * np.square(z))
    return _mean(density[mask])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if int(mask.sum()) < 3:
        return 0.0
    x = a_arr[mask]
    y = b_arr[mask]
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _robust_z(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    med = float(np.nanmedian(arr[finite]))
    q25, q75 = np.nanpercentile(arr[finite], [25.0, 75.0])
    scale = max(float(q75 - q25), eps)
    return ((arr - med) / scale).astype(np.float32)


def _window_ncc(a: np.ndarray, b: np.ndarray, width: int) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    length = min(len(a_arr), len(b_arr))
    if length < 3:
        return 0.0
    local_width = min(int(width), length)
    scores = []
    for start in range(0, length - local_width + 1):
        scores.append(_corr(a_arr[start : start + local_width], b_arr[start : start + local_width]))
    if not scores:
        return _corr(a_arr[:length], b_arr[:length])
    return float(np.nanmean(scores))


def _gr_components(horizontal_gr: np.ndarray, typewell_gr: np.ndarray) -> dict[str, float]:
    h_z = _robust_z(horizontal_gr)
    tw_z = _robust_z(typewell_gr)
    finite = np.isfinite(h_z) & np.isfinite(tw_z)
    mad = float(np.nanmedian(np.abs(h_z[finite] - tw_z[finite]))) if finite.any() else 0.0
    gr_corr = _corr(h_z, tw_z)
    dgr_corr = _corr(np.gradient(h_z), np.gradient(tw_z))
    return {
        "gr_score": float(gr_corr + 0.8 * dgr_corr - 0.1 * mad),
        "gr_corr": gr_corr,
        "dgr_corr": dgr_corr,
        "gr_mad": mad,
        "gr_finite_frac": float(finite.mean()) if len(finite) else 0.0,
        "ncc8": _window_ncc(h_z, tw_z, 8),
        "ncc15": _window_ncc(h_z, tw_z, 15),
        "ncc25": _window_ncc(h_z, tw_z, 25),
    }


def _step_lookup(hidden_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    numeric = [
        column
        for column in (
            "base_tvt",
            "b2_tvt",
            "a_p50_tvt",
            "a_p10_tvt",
            "a_p90_tvt",
            "b2_danger",
        )
        if column in hidden_rows.columns
    ]
    lookup: dict[str, pd.DataFrame] = {}
    for well_id, group in hidden_rows.groupby("well_id"):
        if numeric:
            lookup[str(well_id)] = group.groupby("step", as_index=True)[numeric].mean()
        else:
            lookup[str(well_id)] = pd.DataFrame(index=group["step"].unique())
    return lookup


def _values_for_steps(
    lookup: dict[str, pd.DataFrame], well_id: str, steps: np.ndarray, column: str
) -> np.ndarray:
    frame = lookup.get(str(well_id))
    if frame is None or column not in frame.columns:
        return np.full(len(steps), np.nan, dtype=np.float32)
    return pd.to_numeric(frame.reindex(steps)[column], errors="coerce").to_numpy(
        dtype=np.float32
    )


def _hidden_position_lookup(hidden_rows: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for well_id, group in hidden_rows.groupby("well_id"):
        steps = pd.to_numeric(group["step"], errors="coerce")
        out[str(well_id)] = (float(steps.min()), float(steps.max()))
    return out


def build_mode_feature_frame(
    mode_windows: pd.DataFrame,
    hidden_rows: pd.DataFrame,
    gr_context: dict[str, dict[str, np.ndarray]] | None,
    *,
    history_steps: int,
    future_steps: int,
) -> pd.DataFrame:
    context = gr_context or {}
    lookup = _step_lookup(hidden_rows)
    hidden_pos = _hidden_position_lookup(hidden_rows)
    window_ids = _window_ids(mode_windows)
    rows: list[dict[str, Any]] = []

    for window_id, row in zip(window_ids, mode_windows.itertuples(index=False), strict=True):
        well_id = str(row.well_id)
        logits = _as_float_array(row.logits)
        probs = _as_float_array(getattr(row, "probs", _softmax_np(logits)))
        if len(probs) != len(logits):
            probs = _softmax_np(logits)
        paths = _as_float_array(row.path_tvt)
        target = _as_float_array(row.target_tvt)
        order = np.argsort(-logits)
        logit_ranks = np.empty_like(order)
        logit_ranks[order] = np.arange(len(order))
        errors = np.sqrt(np.mean(np.square(paths - target[None, :]), axis=1))
        error_order = np.argsort(errors)
        best_mode = int(error_order[0])
        top3_modes = set(int(item) for item in error_order[: min(3, len(error_order))])
        entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
        steps = np.arange(
            int(row.start_step) + history_steps,
            int(row.start_step) + history_steps + future_steps,
            dtype=np.int32,
        )
        base = _values_for_steps(lookup, well_id, steps, "base_tvt")
        b2 = _values_for_steps(lookup, well_id, steps, "b2_tvt")
        a_p50 = _values_for_steps(lookup, well_id, steps, "a_p50_tvt")
        a_p10 = _values_for_steps(lookup, well_id, steps, "a_p10_tvt")
        a_p90 = _values_for_steps(lookup, well_id, steps, "a_p90_tvt")
        b2_danger = _values_for_steps(lookup, well_id, steps, "b2_danger")
        pos_min, pos_max = hidden_pos.get(well_id, (float(steps[0]), float(steps[-1])))
        denom = max(pos_max - pos_min, 1.0)
        hidden_pos_frac = float((steps[0] - pos_min) / denom)
        ctx = context.get(well_id, {})
        horizontal_gr = _as_float_array(ctx.get("horizontal_gr", np.empty(0)))
        start = int(row.start_step) + history_steps
        h_gr = horizontal_gr[start : start + future_steps]
        if len(h_gr) < future_steps:
            h_gr = np.pad(h_gr, (0, future_steps - len(h_gr)), constant_values=np.nan)
        typewell_tvt = _as_float_array(ctx.get("typewell_tvt", np.empty(0)))
        typewell_gr = _as_float_array(ctx.get("typewell_gr", np.empty(0)))

        for mode_index in range(paths.shape[0]):
            path = paths[mode_index].astype(np.float32)
            slope = np.diff(path)
            curvature = np.diff(path, n=2)
            if len(typewell_tvt) and len(typewell_gr):
                tw_gr = np.interp(
                    path,
                    typewell_tvt,
                    typewell_gr,
                    left=float(typewell_gr[0]),
                    right=float(typewell_gr[-1]),
                ).astype(np.float32)
                gr = _gr_components(h_gr, tw_gr)
            else:
                gr = {
                    "gr_score": 0.0,
                    "gr_corr": 0.0,
                    "dgr_corr": 0.0,
                    "gr_mad": 0.0,
                    "gr_finite_frac": 0.0,
                    "ncc8": 0.0,
                    "ncc15": 0.0,
                    "ncc25": 0.0,
                }
            inside_band = (
                np.isfinite(a_p10)
                & np.isfinite(a_p90)
                & np.isfinite(path)
                & (path >= np.minimum(a_p10, a_p90))
                & (path <= np.maximum(a_p10, a_p90))
            )
            a_band_inside_frac = float(inside_band.mean()) if len(inside_band) else 0.0
            a_uncertainty = _mean(np.abs(a_p90 - a_p10))
            base_b2_gap = _mean(np.abs(base - b2))
            b2_danger_mean = _mean(b2_danger)
            gr_nan_ratio = float(1.0 - gr["gr_finite_frac"])
            slope_p95 = _p95_abs(slope)
            curvature_mean = _mean(curvature)
            roughness = _mean(np.abs(curvature))
            a_density_mean = _density_mean_from_quantiles(path, a_p50, a_p10, a_p90)
            mean_abs_to_base = _mean_abs_diff(path, base)
            mean_abs_to_b2 = _mean_abs_diff(path, b2)
            mean_abs_to_a_p50 = _mean_abs_diff(path, a_p50)
            p95_abs_to_b2 = _p95_abs_diff(path, b2)
            endpoint_abs_to_b2 = _endpoint_abs_diff(path, b2)
            item = {
                "well_id": well_id,
                "window_id": window_id,
                "mode_id": int(mode_index),
                "sample_type": str(getattr(row, "sample_type", "unknown")),
                "mode_rmse_ft": float(errors[mode_index]),
                "is_best_mode": int(mode_index == best_mode),
                "is_top3_mode": int(mode_index in top3_modes),
                "logit": float(logits[mode_index]),
                "prob": float(probs[mode_index]),
                "logit_rank": int(logit_ranks[mode_index]),
                "logit_gap_to_top1": float(logits[order[0]] - logits[mode_index]),
                "entropy": entropy,
                "mode_index": int(mode_index),
                "path_mean": _mean(path),
                "path_std": float(np.nanstd(path)) if np.isfinite(path).any() else 0.0,
                "path_start": float(path[0]) if len(path) else 0.0,
                "path_end": float(path[-1]) if len(path) else 0.0,
                "path_delta": float(path[-1] - path[0]) if len(path) else 0.0,
                "slope_mean": _mean(slope),
                "slope_abs_mean": _mean(np.abs(slope)),
                "slope_std": float(np.nanstd(slope)) if np.isfinite(slope).any() else 0.0,
                "curvature_abs_mean": _mean(np.abs(curvature)),
                "base_mae": mean_abs_to_base,
                "base_p95_abs": _p95_abs_diff(path, base),
                "base_endpoint_abs": _endpoint_abs_diff(path, base),
                "b2_mae": mean_abs_to_b2,
                "b2_p95_abs": p95_abs_to_b2,
                "b2_endpoint_abs": endpoint_abs_to_b2,
                "a_p50_mae": mean_abs_to_a_p50,
                "a_p50_p95_abs": _p95_abs_diff(path, a_p50),
                "a_p50_endpoint_abs": _endpoint_abs_diff(path, a_p50),
                "a_band_inside_frac": a_band_inside_frac,
                "a_band_width_mean": a_uncertainty,
                "base_b2_gap_mean": base_b2_gap,
                "b2_danger_mean": b2_danger_mean,
                "start_step": int(row.start_step),
                "hidden_pos_frac": hidden_pos_frac,
                "nn_logit": float(logits[mode_index]),
                "nn_prob": float(probs[mode_index]),
                "nn_rank": int(logit_ranks[mode_index]),
                "mode_mean_tvt": _mean(path),
                "mode_std_tvt": float(np.nanstd(path)) if np.isfinite(path).any() else 0.0,
                "mean_abs_to_b2": mean_abs_to_b2,
                "p95_abs_to_b2": p95_abs_to_b2,
                "endpoint_abs_to_b2": endpoint_abs_to_b2,
                "mean_abs_to_base": mean_abs_to_base,
                "mean_abs_to_A_p50": mean_abs_to_a_p50,
                "inside_A_band_frac": a_band_inside_frac,
                "A_density_mean": a_density_mean,
                "slope_p95": slope_p95,
                "curvature_mean": curvature_mean,
                "roughness": roughness,
                "window_hidden_pos_frac": hidden_pos_frac,
                "GR_nan_ratio": gr_nan_ratio,
                "B2_danger": b2_danger_mean,
                "A_uncertainty": a_uncertainty,
                "base_b2_gap": base_b2_gap,
                **gr,
            }
            for column in FEATURE_COLUMNS:
                item[column] = float(_finite_or_zero(np.asarray([item[column]]))[0])
            rows.append(item)
    return pd.DataFrame(rows)


def train_catboost_ranker(
    features: pd.DataFrame,
    *,
    train_wells: set[str],
    valid_wells: set[str],
    seed: int,
    params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, float]]:
    from catboost import CatBoostRegressor, Pool

    train = features[features["well_id"].isin(train_wells)].copy()
    valid = features[features["well_id"].isin(valid_wells)].copy()
    if train.empty or valid.empty:
        raise ValueError("ranker train and valid splits must both be non-empty")
    model_params = dict(DEFAULT_CATBOOST_PARAMS)
    model_params.update(params or {})
    if len(train) < 10:
        model_params["bootstrap_type"] = "No"
    if str(model_params.get("bootstrap_type", "")).lower() == "no":
        model_params.pop("subsample", None)
    model_params["random_seed"] = seed
    train_pool = Pool(
        train[list(FEATURE_COLUMNS)],
        label=np.log1p(train["mode_rmse_ft"].to_numpy(dtype=np.float32)),
        cat_features=list(CAT_FEATURES),
    )
    valid_pool = Pool(
        valid[list(FEATURE_COLUMNS)],
        label=np.log1p(valid["mode_rmse_ft"].to_numpy(dtype=np.float32)),
        cat_features=list(CAT_FEATURES),
    )
    model = CatBoostRegressor(**model_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    train_pred = np.expm1(model.predict(train[list(FEATURE_COLUMNS)]))
    valid_pred = np.expm1(model.predict(valid[list(FEATURE_COLUMNS)]))
    return model, {
        "train_mode_rmse_mae": float(
            np.mean(np.abs(train_pred - train["mode_rmse_ft"].to_numpy(dtype=np.float32)))
        ),
        "valid_mode_rmse_mae": float(
            np.mean(np.abs(valid_pred - valid["mode_rmse_ft"].to_numpy(dtype=np.float32)))
        ),
        "best_iteration": int(getattr(model, "get_best_iteration", lambda: -1)() or -1),
    }


def _predict_ranker_errors(model: Any, features: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(features[list(FEATURE_COLUMNS)])).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(np.mean(arr[finite]))
    std = float(np.std(arr[finite]))
    if std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def _ranker_oof_metrics(oof: pd.DataFrame) -> dict[str, float]:
    best_rows = (
        oof.sort_values(["window_id", "predicted_error_ft"])
        .groupby("window_id", as_index=False)
        .head(1)
    )
    top3 = (
        oof.sort_values(["window_id", "predicted_error_ft"])
        .groupby("window_id", as_index=False)
        .head(3)
    )
    return {
        "oof_mode_rmse_mae": float(
            np.mean(
                np.abs(
                    oof["predicted_error_ft"].to_numpy(dtype=np.float32)
                    - oof["mode_rmse_ft"].to_numpy(dtype=np.float32)
                )
            )
        ),
        "oof_top1_ft": float(best_rows["mode_rmse_ft"].mean()),
        "oof_best_mode_top1_rate": float(best_rows["is_best_mode"].mean()),
        "oof_best_mode_top3_rate": float(
            top3.groupby("window_id")["is_best_mode"].max().mean()
        ),
    }


def _write_crossfit_report(
    run_dir: Path,
    *,
    summary: dict[str, Any],
) -> Path:
    lines = [
        "MTP_RANKER_CROSSFIT_V0_REPORT",
        "",
        "dataset:",
        f"  feature_rows: {summary['feature_rows']}",
        f"  oof_rows: {summary['oof_rows']}",
        f"  wells: {summary['wells']}",
        f"  folds: {summary['folds']}",
        f"  variant: {summary['variant']}",
        f"  beta_grid: {summary['beta_grid']}",
        "",
        "oof metrics:",
        json.dumps(summary["oof_metrics"], indent=2),
        "",
        "folds:",
        json.dumps(summary["fold_summaries"], indent=2),
    ]
    path = run_dir / "crossfit_ranker_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ranker_crossfit_from_frames(
    *,
    run_dir: str | Path,
    cfg: MTPConfig,
    mode_windows: pd.DataFrame,
    hidden_rows_all: pd.DataFrame,
    gr_context: dict[str, dict[str, np.ndarray]] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    ranker_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "checkpoints").mkdir(exist_ok=True)
    mode_windows = _ensure_window_ids(normalize_mode_windows(mode_windows))
    hidden_rows_all = _attach_prior_columns(hidden_rows_all, cfg)
    features = build_mode_feature_frame(
        mode_windows,
        hidden_rows_all,
        gr_context or {},
        history_steps=cfg.window.history_steps,
        future_steps=cfg.window.future_steps,
    )
    folds = make_group_folds(
        set(mode_windows["well_id"].astype(str)), n_folds=n_folds, seed=seed
    )
    oof_parts: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in folds:
        train_wells = set(fold["train_wells"])
        valid_wells = set(fold["valid_wells"])
        model, train_metrics = train_catboost_ranker(
            features,
            train_wells=train_wells,
            valid_wells=valid_wells,
            seed=seed + int(fold["fold"]),
            params=ranker_params,
        )
        model_path = run_path / "checkpoints" / f"mtp_ranker_crossfit_fold{fold['fold']}.cbm"
        model.save_model(str(model_path))
        valid = features[features["well_id"].isin(valid_wells)].copy()
        valid["fold"] = int(fold["fold"])
        valid["predicted_error_ft"] = _predict_ranker_errors(model, valid)
        valid["ranker_score"] = -valid["predicted_error_ft"]
        valid["ranker_logit_t5"] = -valid["predicted_error_ft"] / 5.0
        oof_parts.append(valid)
        fold_summaries.append(
            {
                "fold": int(fold["fold"]),
                "train_wells": sorted(train_wells),
                "valid_wells": sorted(valid_wells),
                **train_metrics,
            }
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["ranker_prob_t5"] = 0.0
    for beta in DEFAULT_RANKER_BETA_GRID:
        oof[f"combined_logit_b{beta:g}"] = 0.0
        oof[f"combined_prob_b{beta:g}"] = 0.0
    for _, index in oof.groupby("window_id").groups.items():
        local = oof.loc[index, "ranker_logit_t5"].to_numpy(dtype=np.float32)
        oof.loc[index, "ranker_prob_t5"] = _softmax_np(local)
        ranker_z = _zscore(oof.loc[index, "ranker_score"].to_numpy(dtype=np.float32))
        nn_logits = oof.loc[index, "nn_logit"].to_numpy(dtype=np.float32)
        for beta in DEFAULT_RANKER_BETA_GRID:
            combined = (nn_logits + float(beta) * ranker_z).astype(np.float32)
            oof.loc[index, f"combined_logit_b{beta:g}"] = combined
            oof.loc[index, f"combined_prob_b{beta:g}"] = _softmax_np(combined)
    oof.to_parquet(run_path / "oof_ranker_logits.parquet", index=False)
    features.to_parquet(run_path / "crossfit_ranker_mode_features.parquet", index=False)
    summary: dict[str, Any] = {
        "feature_rows": int(len(features)),
        "oof_rows": int(len(oof)),
        "wells": int(oof["well_id"].nunique()),
        "folds": int(n_folds),
        "seed": int(seed),
        "variant": "conservative_regression",
        "feature_columns": list(FEATURE_COLUMNS),
        "beta_grid": list(DEFAULT_RANKER_BETA_GRID),
        "oof_metrics": _ranker_oof_metrics(oof),
        "fold_summaries": fold_summaries,
    }
    (run_path / "crossfit_ranker_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_crossfit_report(run_path, summary=summary)
    return summary


def apply_ranker_logits(
    mode_windows: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    tau_ft: float,
    beta: float | None = None,
) -> pd.DataFrame:
    out = mode_windows.copy()
    pred_lookup = predictions.set_index(["window_id", "mode_id"])["predicted_error_ft"]
    new_logits: list[np.ndarray] = []
    for window_id, row in zip(_window_ids(out), out.itertuples(index=False), strict=True):
        logits = _as_float_array(row.logits)
        errors = []
        for mode_index in range(len(logits)):
            key = (window_id, int(mode_index))
            value = pred_lookup.get(key, np.nan)
            errors.append(float(value) if np.isfinite(value) else 1e6)
        ranker_score = -np.asarray(errors, dtype=np.float32)
        if beta is None:
            values = (ranker_score / float(tau_ft)).astype(np.float32)
        else:
            values = (logits + float(beta) * _zscore(ranker_score)).astype(np.float32)
        new_logits.append(values)
    out["logits"] = new_logits
    out["probs"] = [_softmax_np(values) for values in new_logits]
    return out


def _attach_prior_columns(hidden_rows: pd.DataFrame, cfg: MTPConfig) -> pd.DataFrame:
    out = hidden_rows.copy()
    prior_tables = load_prior_tables(cfg.priors)
    if prior_tables is None:
        return out
    aligned = prior_tables.frame.reindex(out["id"].astype(str))
    for column in ("a_p50_tvt", "a_p10_tvt", "a_p90_tvt", "b2_danger"):
        if column in aligned.columns and column not in out.columns:
            out[column] = pd.to_numeric(aligned[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
    return out


def _candidate_table(metrics: list[dict[str, Any]]) -> str:
    columns = [
        "candidate",
        "rmse",
        "covered_rmse",
        "mean_well_rmse",
        "p95_well_rmse",
        "worst_well_rmse",
        "p95_abs_shift_vs_b2",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in sorted(metrics, key=lambda row: row.get("rmse", float("inf"))):
        values = []
        for column in columns:
            value = item.get(column, "n/a")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_ranker_report(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Path:
    best = min(candidates, key=lambda item: item.get("rmse", float("inf")))
    b2 = summary["baselines"]["b2_guarded_submit"]["rmse"]
    lines = [
        "MTP_RANKER_V0_REPORT",
        "",
        "dataset:",
        f"  feature_rows: {summary['dataset']['feature_rows']}",
        f"  windows: {summary['dataset']['windows']}",
        f"  modes_per_window: {summary['dataset']['modes_per_window']}",
        f"  ranker_train_wells: {summary['ranker_split']['train_wells']}",
        f"  ranker_valid_wells: {summary['ranker_split']['valid_wells']}",
        "",
        "baselines:",
        f"  base_schema10_pp: {summary['baselines']['base_schema10_pp']['rmse']}",
        f"  b2_guarded_submit: {b2}",
        "",
        "window metrics:",
        json.dumps(summary["window_metrics"], indent=2),
        "",
        "best row-level candidate:",
        f"  candidate: {best['candidate']}",
        f"  rmse: {best['rmse']}",
        f"  gain_vs_b2: {b2 - best['rmse']}",
        f"  covered_rmse: {best.get('covered_rmse', 'n/a')}",
        f"  p95_abs_shift_vs_b2: {best.get('p95_abs_shift_vs_b2', 'n/a')}",
        f"  worst_well_rmse: {best.get('worst_well_rmse', 'n/a')}",
        "",
        "row-level:",
        _candidate_table(candidates),
        "",
        "decision:",
        f"  beats_b2_by_0_10: {(b2 - best['rmse']) >= 0.10}",
        f"  strong_go_le_9_80: {best['rmse'] <= 9.80}",
    ]
    path = run_dir / "ranker_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ranker_window_metrics(
    base_windows: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    beta_grid: tuple[float, ...],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "nn_logits": _window_metrics_from_mode_windows(base_windows)
    }
    for beta in beta_grid:
        reranked = apply_ranker_logits(base_windows, predictions, tau_ft=5.0, beta=beta)
        metrics[f"ranker_beta_{beta:g}"] = _window_metrics_from_mode_windows(reranked)
    return metrics


def run_ranker_from_frames(
    *,
    run_dir: str | Path,
    cfg: MTPConfig,
    mode_windows: pd.DataFrame,
    hidden_rows_all: pd.DataFrame,
    gr_context: dict[str, dict[str, np.ndarray]] | None = None,
    seed: int = 42,
    valid_fraction: float = 0.35,
    ranker_params: dict[str, Any] | None = None,
    beta_grid: tuple[float, ...] = DEFAULT_RANKER_BETA_GRID,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "checkpoints").mkdir(exist_ok=True)
    mode_windows = _ensure_window_ids(normalize_mode_windows(mode_windows))
    hidden_rows_all = _attach_prior_columns(hidden_rows_all, cfg)
    all_wells = set(mode_windows["well_id"].astype(str))
    train_wells, valid_wells = split_ranker_wells(
        all_wells, valid_fraction=valid_fraction, seed=seed
    )
    features = build_mode_feature_frame(
        mode_windows,
        hidden_rows_all,
        gr_context or {},
        history_steps=cfg.window.history_steps,
        future_steps=cfg.window.future_steps,
    )
    features["split"] = np.where(features["well_id"].isin(valid_wells), "valid", "train")
    features.to_parquet(run_path / "ranker_mode_features.parquet", index=False)

    model, train_metrics = train_catboost_ranker(
        features,
        train_wells=train_wells,
        valid_wells=valid_wells,
        seed=seed,
        params=ranker_params,
    )
    model.save_model(str(run_path / "checkpoints" / "mtp_ranker_catboost.cbm"))
    features["predicted_error_ft"] = np.expm1(model.predict(features[list(FEATURE_COLUMNS)]))
    features["ranker_score"] = -features["predicted_error_ft"]
    features.to_parquet(run_path / "ranker_predictions.parquet", index=False)

    valid_windows = mode_windows[mode_windows["well_id"].astype(str).isin(valid_wells)].copy()
    valid_features = features[features["well_id"].isin(valid_wells)].copy()
    valid_hidden = hidden_rows_all[hidden_rows_all["well_id"].astype(str).isin(valid_wells)].copy()
    window_metrics = _ranker_window_metrics(
        valid_windows, valid_features, beta_grid=beta_grid
    )
    if gr_context:
        for beta in (0.25, 0.5, 1.0, 2.0):
            gr_windows = attach_gr_rerank_scores(
                valid_windows,
                gr_context,
                history_steps=cfg.window.history_steps,
                future_steps=cfg.window.future_steps,
                beta=beta,
            )
            window_metrics[f"simple_gr_beta_{beta:g}"] = _window_metrics_from_mode_windows(
                gr_windows
            )

    candidate_steps: dict[str, pd.DataFrame] = {
        "mtp_nn_top1_overlap": aggregate_mode_windows(
            valid_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="top1",
        ),
        "mtp_nn_weighted_overlap": aggregate_mode_windows(
            valid_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="weighted",
        ),
    }
    if gr_context:
        for beta in (0.25, 0.5, 1.0, 2.0):
            gr_windows = attach_gr_rerank_scores(
                valid_windows,
                gr_context,
                history_steps=cfg.window.history_steps,
                future_steps=cfg.window.future_steps,
                beta=beta,
            )
            candidate_steps[f"mtp_simple_gr_weighted_b{beta:g}"] = aggregate_mode_windows(
                gr_windows,
                history_steps=cfg.window.history_steps,
                future_steps=cfg.window.future_steps,
                strategy="weighted",
            )
    wrote_top1 = False
    for beta in beta_grid:
        ranker_windows = apply_ranker_logits(
            valid_windows, valid_features, tau_ft=5.0, beta=beta
        )
        if not wrote_top1:
            candidate_steps["mtp_ranker_top1_overlap"] = aggregate_mode_windows(
                ranker_windows,
                history_steps=cfg.window.history_steps,
                future_steps=cfg.window.future_steps,
                strategy="top1",
            )
            wrote_top1 = True
        candidate_steps[f"mtp_ranker_weighted_b{beta:g}"] = aggregate_mode_windows(
            ranker_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="weighted",
        )
        candidate_steps[f"mtp_ranker_top3_b{beta:g}"] = aggregate_mode_windows(
            ranker_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="top3",
        )

    covered_keys = pd.concat(candidate_steps.values(), ignore_index=True)[
        ["well_id", "step"]
    ].drop_duplicates()
    hidden_covered = valid_hidden.merge(covered_keys, on=["well_id", "step"], how="inner")
    candidate_metrics: list[dict[str, Any]] = []
    row_predictions: list[pd.DataFrame] = []

    def add_candidate(name: str, rows: pd.DataFrame) -> None:
        candidate_rows = rows.copy()
        candidate_rows["candidate"] = name
        row_predictions.append(candidate_rows)
        candidate_metrics.append(evaluate_with_b2_fallback(valid_hidden, candidate_rows, name))

    for name, steps in candidate_steps.items():
        rows = _apply_step_predictions_to_rows(
            hidden_covered, steps, anchor_column="base_tvt"
        )
        add_candidate(name, rows)
        if name.startswith("mtp_ranker_weighted"):
            for alpha in (0.1, 0.2, 0.3):
                for clip in (20.0, 30.0):
                    blend_name = f"b2_plus_{name}_a{alpha:g}_clip{int(clip)}"
                    add_candidate(
                        blend_name,
                        _blend_with_anchor(
                            rows,
                            anchor_column="b2_tvt",
                            alpha=alpha,
                            clip=clip,
                        ),
                    )

    candidate_frame = pd.DataFrame(candidate_metrics).sort_values("rmse")
    candidate_frame.to_csv(run_path / "ranker_candidates.csv", index=False)
    if row_predictions:
        pd.concat(row_predictions, ignore_index=True).to_parquet(
            run_path / "ranker_row_predictions.parquet", index=False
        )
    base_metrics = _baseline_metrics(valid_hidden, "base_tvt", "base_schema10_pp")
    b2_metrics = _baseline_metrics(valid_hidden, "b2_tvt", "b2_guarded_submit")
    summary: dict[str, Any] = {
        "dataset": {
            "feature_rows": int(len(features)),
            "windows": int(len(mode_windows)),
            "modes_per_window": float(len(features) / max(1, len(mode_windows))),
        },
        "ranker_split": {
            "seed": seed,
            "valid_fraction": valid_fraction,
            "train_wells": sorted(train_wells),
            "valid_wells": sorted(valid_wells),
        },
        "train_metrics": train_metrics,
        "window_metrics": window_metrics,
        "baselines": {
            "base_schema10_pp": base_metrics,
            "b2_guarded_submit": b2_metrics,
        },
        "candidates": candidate_metrics,
    }
    (run_path / "ranker_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_ranker_report(run_path, summary=summary, candidates=candidate_metrics)
    return summary


def run_ranker(
    run_dir: str | Path,
    *,
    seed: int = 42,
    valid_fraction: float = 0.35,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = _load_run_config(run_path)
    windows_path = run_path / "stitch_window_modes.parquet"
    if not windows_path.exists():
        raise FileNotFoundError(
            f"Missing {windows_path}; run `make stitch RUN_DIR={run_path}` first"
        )
    mode_windows = pd.read_parquet(windows_path)
    well_ids = set(mode_windows["well_id"].astype(str))
    hidden_rows = _load_hidden_rows(cfg, well_ids)
    gr_context = _load_gr_context(cfg, well_ids)
    summary = run_ranker_from_frames(
        run_dir=run_path,
        cfg=cfg,
        mode_windows=mode_windows,
        hidden_rows_all=hidden_rows,
        gr_context=gr_context,
        seed=seed,
        valid_fraction=valid_fraction,
    )
    best = min(summary["candidates"], key=lambda item: item.get("rmse", float("inf")))
    print(json.dumps(_json_safe(best), indent=2), flush=True)
    return summary


def run_ranker_crossfit(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    out_path = Path(output_dir) if output_dir is not None else run_path
    cfg = _load_run_config(run_path)
    windows_path = run_path / "stitch_window_modes.parquet"
    if not windows_path.exists():
        raise FileNotFoundError(
            f"Missing {windows_path}; run `make stitch RUN_DIR={run_path}` first"
        )
    mode_windows = pd.read_parquet(windows_path)
    well_ids = set(mode_windows["well_id"].astype(str))
    hidden_rows = _load_hidden_rows(cfg, well_ids)
    gr_context = _load_gr_context(cfg, well_ids)
    summary = run_ranker_crossfit_from_frames(
        run_dir=out_path,
        cfg=cfg,
        mode_windows=mode_windows,
        hidden_rows_all=hidden_rows,
        gr_context=gr_context,
        n_folds=n_folds,
        seed=seed,
    )
    print(json.dumps(_json_safe(summary["oof_metrics"]), indent=2), flush=True)
    return summary
