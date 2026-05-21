from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .constants import FORMATIONS
from .features import build_training_table
from .io import horizontal_files, resolve_data_dir, resolve_train_dir
from .pipeline import build_top_context, shuffled_path_folds
from .runlog import RunLogger, format_duration
from .surface_student_integration import reconstruct_schema10_oof


TARGET_TRUE_RESIDUAL = "target_true_residual_schema10"
TARGET_TEACHER_DELTA = "target_teacher_delta_last"
SURFACE_TARGET_PREFIX = "target_z_minus_surface_"

BASE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "id",
    "well",
    "row_index",
    "fold",
    "tvt_true",
    "schema10_oof_raw",
    "flat_tvt",
    "last_known_tvt",
)

SAFE_PREFIXES: tuple[str, ...] = (
    "kg_beam",
    "kg_ncc",
    "kg_dtw",
    "kg_dwt",
    "kg_pf",
    "kg_form",
    "kg_dense",
    "kg_signal",
    "tvtF",
    "frm_",
    "bw",
    "tda",
    "tdbc",
    "tdsc",
    "tdpf",
    "tddtw",
    "typewell",
    "tail_",
    "gr_",
    "spatial_",
    "dense_",
)

SAFE_EXACT: set[str] = {
    "idx",
    "idx_frac",
    "well_n_rows",
    "known_count",
    "known_frac",
    "md",
    "x",
    "y",
    "z",
    "gr",
    "flat_tvt",
    "baseline_tvt",
    "baseline_minus_flat",
    "first_known_idx",
    "last_known_idx",
    "first_known_tvt",
    "last_known_tvt",
    "anchor_tvt",
    "idx_from_last_known",
    "md_from_start",
    "md_from_last_known",
    "z_from_last_known",
    "x_from_last_known",
    "y_from_last_known",
    "md_since",
    "dx",
    "dy",
    "dz",
    "frac",
    "frac2",
    "sqrt_frac",
    "xy_dist_from_last_known",
    "dxy",
    "dzdmd",
    "dxdmd",
    "dydmd",
    "prev_known_tvt",
    "idx_from_prev_known",
    "md_from_prev_known",
    "z_from_prev_known",
    "gr",
    "gr_from_well_mean",
    "gr_grad_md",
    "z_grad_md",
    "x_grad_md",
    "y_grad_md",
}

LEAKY_PREFIXES: tuple[str, ...] = (
    "kg_path",
    "stage1",
    "stage12",
    "cem_",
    "geo_consensus",
    "geo_tailfit",
    "crosswell",
)


def _rmse(pred: np.ndarray | pd.Series, true: np.ndarray | pd.Series) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not np.any(mask):
        return float("nan")
    err = pred_arr[mask] - true_arr[mask]
    return float(np.sqrt(np.mean(err * err)))


def _safe_quantile(values: np.ndarray | pd.Series, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float("nan")


def _rank_corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    frame = pd.DataFrame({"a": np.asarray(a, dtype=float), "b": np.asarray(b, dtype=float)})
    frame = frame[np.isfinite(frame["a"]) & np.isfinite(frame["b"])]
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return float("nan")
    return float(frame["a"].rank().corr(frame["b"].rank()))


def _well_rmse(frame: pd.DataFrame, pred_col: str, true_col: str = "tvt_true") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        rows.append(
            {
                "well": str(well),
                "rows": int(len(group)),
                "rmse": _rmse(group[pred_col], group[true_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse", ascending=False)


def _well_summary(frame: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    by_well = _well_rmse(frame, pred_col)
    values = by_well["rmse"].to_numpy(dtype=float) if len(by_well) else np.array([])
    return {
        "mean_well_rmse": float(np.nanmean(values)) if len(values) else float("nan"),
        "median_well_rmse": float(np.nanmedian(values)) if len(values) else float("nan"),
        "p90_well_rmse": _safe_quantile(values, 0.90),
        "p95_well_rmse": _safe_quantile(values, 0.95),
        "worst_well_rmse": float(np.nanmax(values)) if len(values) else float("nan"),
        "worst_well": str(by_well.iloc[0]["well"]) if len(by_well) else None,
    }


def _prepare_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(config)
    prepared.setdefault("features", {})
    prepared["features"]["include_direct_path_features"] = False
    prepared["features"].setdefault("cache", {})
    top_cfg = prepared["features"].setdefault("kaggle_top", {})
    top_cfg["hmm_enabled"] = False
    prepared.setdefault("data", {})
    prepared["data"]["target_rows"] = "hidden_only"
    return prepared


def _hidden_ids(X: pd.DataFrame, groups: np.ndarray) -> tuple[list[str], np.ndarray]:
    if "idx" not in X.columns:
        raise ValueError("Feature table is missing idx column")
    row_index = np.rint(X["idx"].to_numpy(dtype=float)).astype(int)
    wells = np.asarray(groups, dtype=object)
    ids = [
        f"{str(well)}_{int(idx)}"
        for well, idx in zip(wells, row_index, strict=False)
    ]
    return ids, row_index


def build_fold_safe_feature_frame(
    *,
    train_paths: list[Path],
    config: dict[str, Any],
    seed: int,
    logger: RunLogger,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    n_splits = int(
        config.get("surface_student_v1", {}).get("n_splits")
        or config["validation"].get("n_splits", 5)
    )
    folds = shuffled_path_folds(train_paths, n_splits, seed)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    logger.info("Surface student v1 folds prepared", folds=len(folds), wells=len(train_paths))
    for fold_id, fold_train_paths, fold_valid_paths in folds:
        fold_started = perf_counter()
        context = build_top_context(
            fold_train_paths,
            config,
            logger,
            f"Build surface student v1 fold {fold_id} context",
        )
        with logger.step(
            "Build surface student v1 valid table",
            fold=fold_id,
            train_wells=len(fold_train_paths),
            valid_wells=len(fold_valid_paths),
        ):
            X, _residual, groups, _flat, y_true = build_training_table(
                fold_valid_paths,
                config,
                context,
                logger,
            )
        ids, row_index = _hidden_ids(X, groups)
        frame = X.reset_index(drop=True).copy()
        frame.insert(0, "id", ids)
        frame.insert(1, "well", np.asarray(groups, dtype=object))
        frame.insert(2, "row_index", row_index)
        frame.insert(3, "fold", int(fold_id))
        frame.insert(4, "tvt_true", np.asarray(y_true, dtype=float))
        parts.append(frame)
        fold_rows.append(
            {
                "fold": int(fold_id),
                "valid_rows": int(len(frame)),
                "valid_wells": int(len(fold_valid_paths)),
                "duration_sec": float(perf_counter() - fold_started),
            }
        )
        logger.info(
            "Surface student v1 fold table complete",
            fold=fold_id,
            rows=len(frame),
            columns=len(frame.columns),
            duration=format_duration(perf_counter() - fold_started),
        )
    combined = pd.concat(parts, ignore_index=True)
    logger.info(
        "Surface student v1 feature table complete",
        rows=len(combined),
        wells=combined["well"].nunique(),
        columns=len(combined.columns),
    )
    return combined, fold_rows


def load_teacher_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher rows not found: {path}. Run `uv run python -m rogii.surface_teacher` first."
        )
    keep = [
        "id",
        "well",
        "row_index",
        "geo_teacher_tvt",
        "geo_teacher_delta_last",
        "geo_teacher_conf",
        "teacher_best_surface_id",
        "teacher_surface_agreement",
        "teacher_surface_spread",
        "teacher_surface_fit_rmse_min",
        "teacher_surface_fit_rmse_gap",
        "teacher_abs_error",
        *(f"z_minus_{formation}_true" for formation in FORMATIONS),
    ]
    return pd.read_csv(path, usecols=lambda column: column in set(keep))


def attach_schema10_oof(
    frame: pd.DataFrame,
    *,
    model_path: Path,
    config_path: Path,
    data_dir: Path,
    logger: RunLogger,
) -> pd.DataFrame:
    with logger.step("Reconstruct schema10 OOF", model_path=model_path):
        schema_oof = reconstruct_schema10_oof(
            model_path=model_path,
            config_path=config_path,
            data_dir=data_dir,
        )
    keep = ["id", "schema10_oof_raw", "tvt_true_schema10"]
    merged = frame.merge(schema_oof[keep], on="id", how="left")
    missing = int(merged["schema10_oof_raw"].isna().sum())
    if missing:
        raise ValueError(f"Schema10 OOF missing for {missing} surface-student rows")
    if not np.allclose(
        merged["tvt_true"].to_numpy(dtype=float),
        merged["tvt_true_schema10"].to_numpy(dtype=float),
        equal_nan=False,
    ):
        raise ValueError("Surface student v1 rows are not aligned with schema10 OOF rows")
    return merged.drop(columns=["tvt_true_schema10"])


def attach_teacher_rows(frame: pd.DataFrame, teacher_rows: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(teacher_rows, on=["id", "well", "row_index"], how="left")
    missing = int(merged["geo_teacher_tvt"].isna().sum())
    if missing:
        raise ValueError(f"Teacher rows missing for {missing} surface-student rows")
    merged[TARGET_TRUE_RESIDUAL] = merged["tvt_true"] - merged["schema10_oof_raw"]
    merged[TARGET_TEACHER_DELTA] = merged["geo_teacher_tvt"] - merged["last_known_tvt"]
    for formation in FORMATIONS:
        merged[f"{SURFACE_TARGET_PREFIX}{formation}"] = merged[f"z_minus_{formation}_true"]
    return merged


def add_v1_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dxdmd = out["dxdmd"].to_numpy(dtype=float) if "dxdmd" in out else np.full(len(out), np.nan)
    dydmd = out["dydmd"].to_numpy(dtype=float) if "dydmd" in out else np.full(len(out), np.nan)
    dzdmd = out["dzdmd"].to_numpy(dtype=float) if "dzdmd" in out else np.full(len(out), np.nan)
    azimuth = np.arctan2(dydmd, dxdmd)
    out["azimuth_sin"] = np.sin(azimuth)
    out["azimuth_cos"] = np.cos(azimuth)
    out["trajectory_slope_norm"] = np.sqrt(
        dxdmd * dxdmd + dydmd * dydmd + dzdmd * dzdmd
    )
    out["gr_isna"] = out["gr"].isna().astype(float) if "gr" in out else np.nan
    if "well" in out and "gr" in out:
        out["gr_nan_frac_well"] = out.groupby("well")["gr_isna"].transform("mean")
        gr_filled = out.groupby("well")["gr"].transform(
            lambda s: s.interpolate(limit_direction="both").bfill().ffill()
        )
        out["gr_volatility_well"] = gr_filled.groupby(out["well"]).transform("std")
        out["gr_interp"] = gr_filled
    candidate_cols = [
        column
        for column in [
            "kg_pf_ancc_tvt",
            "pf_ancc",
            "kg_pf_z_tvt",
            "pf_z",
            "kg_dtw_tvt",
            "kg_dwt_tvt",
            "kg_beam_mean_tvt",
            "kg_ncc_mean_tvt",
            "kg_signal_mean_tvt",
            "kg_dense_ancc_tvt",
        ]
        if column in out.columns
    ]
    if candidate_cols:
        values = out[candidate_cols].to_numpy(dtype=float)
        out["candidate_tvt_count"] = np.isfinite(values).sum(axis=1).astype(float)
        out["candidate_tvt_mean"] = np.nanmean(values, axis=1)
        out["candidate_tvt_std"] = np.nanstd(values, axis=1)
        out["candidate_tvt_range"] = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    for left, right, name in [
        ("kg_pf_ancc_tvt", "kg_dtw_tvt", "student_feat_pf_vs_dtw"),
        ("kg_pf_ancc_tvt", "kg_dwt_tvt", "student_feat_pf_vs_dwt"),
        ("kg_dtw_tvt", "kg_dwt_tvt", "student_feat_dtw_vs_dwt"),
        ("kg_beam_mean_tvt", "kg_ncc_mean_tvt", "student_feat_beam_vs_ncc"),
    ]:
        if left in out.columns and right in out.columns:
            out[name] = out[left] - out[right]
    for formation in FORMATIONS:
        surface_col = f"kg_form_{formation}_tvt"
        if surface_col in out.columns:
            out[f"student_feat_{formation}_minus_schema10"] = out[surface_col] - out["schema10_oof_raw"]
    return out.replace([np.inf, -np.inf], np.nan)


def is_safe_v1_feature(column: str, frame: pd.DataFrame) -> bool:
    if column in set(BASE_OUTPUT_COLUMNS):
        return False
    if column.startswith("target_"):
        return False
    if column.startswith("geo_teacher") or column.startswith("teacher_"):
        return False
    if column.endswith("_true") or "_true_" in column:
        return False
    if any(column.startswith(prefix) for prefix in LEAKY_PREFIXES):
        return False
    if not pd.api.types.is_numeric_dtype(frame[column]):
        return False
    if column in SAFE_EXACT:
        return True
    if column.startswith(SAFE_PREFIXES):
        return True
    if column.startswith("student_feat_"):
        return True
    if column in {
        "azimuth_sin",
        "azimuth_cos",
        "trajectory_slope_norm",
        "gr_isna",
        "gr_nan_frac_well",
        "gr_volatility_well",
        "gr_interp",
        "candidate_tvt_count",
        "candidate_tvt_mean",
        "candidate_tvt_std",
        "candidate_tvt_range",
    }:
        return True
    return False


def select_v1_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if is_safe_v1_feature(column, frame)]
    return sorted(dict.fromkeys(columns))


def teacher_quality_weights(frame: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    threshold = float(cfg.get("teacher_error_threshold", 5.0))
    min_weight = float(cfg.get("teacher_min_weight", 0.05))
    abs_error = frame.get(
        "teacher_abs_error", pd.Series(np.nan, index=frame.index)
    ).to_numpy(dtype=float)
    weights = 1.0 / (1.0 + np.maximum(abs_error, 0.0) / max(threshold, 1e-6))
    weights[~np.isfinite(weights)] = min_weight
    return np.clip(weights, min_weight, 1.0)


def _fit_catboost(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    params: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> Any:
    from catboost import CatBoostRegressor

    model_params = dict(params)
    early_stopping_rounds = model_params.pop("early_stopping_rounds", None)
    model_params.setdefault("random_seed", seed)
    model_params.setdefault("loss_function", "RMSE")
    model_params.setdefault("eval_metric", "RMSE")
    model_params.setdefault("allow_writing_files", False)
    model_params.setdefault("verbose", False)
    model = CatBoostRegressor(**model_params)
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if early_stopping_rounds not in (None, "", 0) and len(X_valid) > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        fit_kwargs["use_best_model"] = True
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)] if len(X_valid) > 0 else None,
        **fit_kwargs,
    )
    return model


def _save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def _write_prediction_artifact(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    parquet_path = output_dir / "oof_predictions.parquet"
    csv_path = output_dir / "oof_predictions.csv"
    try:
        frame.to_parquet(parquet_path, index=False)
        return {"path": str(parquet_path), "format": "parquet", "parquet_written": True}
    except Exception as exc:
        frame.to_csv(csv_path, index=False, float_format="%.6f")
        return {
            "path": str(csv_path),
            "format": "csv",
            "parquet_written": False,
            "parquet_error": str(exc),
        }


def train_heads_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    cfg: dict[str, Any],
    output_dir: Path,
    logger: RunLogger,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    seed = int(cfg.get("seed", 42))
    params = dict(cfg.get("model", {}).get("params", {}))
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    surface_targets = [f"{SURFACE_TARGET_PREFIX}{formation}" for formation in FORMATIONS]
    targets = [TARGET_TRUE_RESIDUAL, TARGET_TEACHER_DELTA, *surface_targets]
    predictions = frame[list(BASE_OUTPUT_COLUMNS)].copy()
    predictions["geo_teacher_tvt"] = frame["geo_teacher_tvt"].to_numpy(dtype=float)
    predictions["geo_teacher_conf"] = frame["geo_teacher_conf"].to_numpy(dtype=float)
    predictions["teacher_abs_error"] = frame["teacher_abs_error"].to_numpy(dtype=float)
    predictions["pf_ancc_tvt"] = frame.get("kg_pf_ancc_tvt", frame.get("pf_ancc", np.nan))
    predictions["pf_z_tvt"] = frame.get("kg_pf_z_tvt", frame.get("pf_z", np.nan))
    predictions["dtw_tvt"] = frame.get("kg_dtw_tvt", np.nan)
    predictions["dwt_tvt"] = frame.get("kg_dwt_tvt", np.nan)
    fold_rows: list[dict[str, Any]] = []

    folds = sorted(int(item) for item in frame["fold"].dropna().unique())
    for target in targets:
        target_started = perf_counter()
        pred_col = target.replace("target_", "pred_")
        pred = np.full(len(frame), np.nan, dtype=float)
        target_values = frame[target].to_numpy(dtype=float)
        logger.info(
            "Surface student v1 target start",
            target=target,
            features=len(feature_columns),
            rows=int(np.isfinite(target_values).sum()),
        )
        for fold in folds:
            train_mask = (frame["fold"].to_numpy(dtype=int) != fold) & np.isfinite(
                target_values
            )
            valid_mask = (frame["fold"].to_numpy(dtype=int) == fold) & np.isfinite(
                target_values
            )
            train_idx = np.flatnonzero(train_mask)
            valid_idx = np.flatnonzero(valid_mask)
            if len(train_idx) < 64 or len(valid_idx) == 0:
                logger.warn(
                    "Surface student v1 target fold skipped",
                    target=target,
                    fold=fold,
                    train_rows=len(train_idx),
                    valid_rows=len(valid_idx),
                )
                continue
            weights = None
            if target == TARGET_TEACHER_DELTA:
                weights = teacher_quality_weights(frame.iloc[train_idx], cfg)
            fold_started = perf_counter()
            model = _fit_catboost(
                X_train=frame.iloc[train_idx][feature_columns],
                y_train=target_values[train_idx],
                X_valid=frame.iloc[valid_idx][feature_columns],
                y_valid=target_values[valid_idx],
                params=params,
                seed=seed + fold * 1000 + len(target),
                sample_weight=weights,
            )
            pred[valid_idx] = model.predict(frame.iloc[valid_idx][feature_columns])
            _save_pickle(model, model_dir / target / f"fold_{fold}.pkl")
            rmse = _rmse(pred[valid_idx], target_values[valid_idx])
            best_iteration = getattr(model, "get_best_iteration", lambda: None)()
            fold_rows.append(
                {
                    "target": target,
                    "fold": int(fold),
                    "valid_rows": int(len(valid_idx)),
                    "rmse": float(rmse),
                    "best_iteration": int(best_iteration or 0),
                    "duration_sec": float(perf_counter() - fold_started),
                }
            )
            logger.metric(
                "Surface student v1 target fold RMSE",
                target=target,
                fold=fold,
                rmse=rmse,
                best_iteration=best_iteration,
                duration=format_duration(perf_counter() - fold_started),
            )
        predictions[pred_col] = pred
        logger.info(
            "Surface student v1 target complete",
            target=target,
            rmse=_rmse(pred, target_values),
            duration=format_duration(perf_counter() - target_started),
        )

    true_pred_col = f"pred_{TARGET_TRUE_RESIDUAL.removeprefix('target_')}"
    teacher_pred_col = f"pred_{TARGET_TEACHER_DELTA.removeprefix('target_')}"
    predictions["geo_student_v1_true_tvt"] = (
        predictions["schema10_oof_raw"] + predictions[true_pred_col]
    )
    predictions["geo_student_v1_teacher_tvt"] = (
        predictions["last_known_tvt"] + predictions[teacher_pred_col]
    )
    true_weight = float(cfg.get("blend_true_weight", 0.7))
    teacher_weight = float(cfg.get("blend_teacher_weight", 0.3))
    denom = max(true_weight + teacher_weight, 1e-9)
    predictions["geo_student_v1_blend_tvt"] = (
        true_weight * predictions["geo_student_v1_true_tvt"]
        + teacher_weight * predictions["geo_student_v1_teacher_tvt"]
    ) / denom
    head_values = predictions[
        ["geo_student_v1_true_tvt", "geo_student_v1_teacher_tvt", "schema10_oof_raw"]
    ].to_numpy(dtype=float)
    predictions["student_ensemble_std"] = np.nanstd(head_values, axis=1)
    predictions["student_error_pred"] = predictions["student_ensemble_std"]
    predictions["geo_student_v1_true_minus_schema10"] = (
        predictions["geo_student_v1_true_tvt"] - predictions["schema10_oof_raw"]
    )
    predictions["geo_student_v1_teacher_minus_schema10"] = (
        predictions["geo_student_v1_teacher_tvt"] - predictions["schema10_oof_raw"]
    )
    predictions["geo_student_v1_blend_minus_schema10"] = (
        predictions["geo_student_v1_blend_tvt"] - predictions["schema10_oof_raw"]
    )
    predictions["geo_student_v1_minus_flat"] = (
        predictions["geo_student_v1_blend_tvt"] - predictions["flat_tvt"]
    )
    predictions["student_vs_pf"] = (
        predictions["geo_student_v1_blend_tvt"] - predictions["pf_ancc_tvt"]
    )
    predictions["student_vs_dtw"] = predictions["geo_student_v1_blend_tvt"] - predictions["dtw_tvt"]
    predictions["student_vs_dwt"] = predictions["geo_student_v1_blend_tvt"] - predictions["dwt_tvt"]
    predictions["student_vs_schema10"] = (
        predictions["geo_student_v1_blend_tvt"] - predictions["schema10_oof_raw"]
    )
    z_values = frame["z"].to_numpy(dtype=float)
    for formation in FORMATIONS:
        pred_col = f"pred_z_minus_surface_{formation}"
        predictions[f"z_minus_surface_hat_{formation}"] = predictions.get(pred_col, np.nan)
        predictions[f"surface_hat_{formation}"] = (
            z_values - predictions[f"z_minus_surface_hat_{formation}"]
        )
        true_surface = z_values - frame[f"z_minus_{formation}_true"].to_numpy(dtype=float)
        predictions[f"surface_unc_{formation}"] = np.abs(
            predictions[f"surface_hat_{formation}"] - true_surface
        )
    return predictions, fold_rows


def run_integration_smoke(
    oof: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    logger: RunLogger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from catboost import CatBoostRegressor

    seed = int(cfg.get("seed", 42))
    iterations = int(cfg.get("integration_iterations", 300))
    feature_columns = [
        column
        for column in [
            "geo_student_v1_true_tvt",
            "geo_student_v1_teacher_tvt",
            "geo_student_v1_blend_tvt",
            "geo_student_v1_true_minus_schema10",
            "geo_student_v1_teacher_minus_schema10",
            "geo_student_v1_blend_minus_schema10",
            "geo_student_v1_minus_flat",
            "student_ensemble_std",
            "student_error_pred",
            "student_vs_pf",
            "student_vs_dtw",
            "student_vs_dwt",
            "student_vs_schema10",
            *(f"surface_hat_{formation}" for formation in FORMATIONS),
            *(f"z_minus_surface_hat_{formation}" for formation in FORMATIONS),
            *(f"surface_unc_{formation}" for formation in FORMATIONS),
        ]
        if column in oof.columns
    ]
    target = oof["tvt_true"].to_numpy(dtype=float) - oof["schema10_oof_raw"].to_numpy(dtype=float)
    pred = np.full(len(oof), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    logger.info("Surface student v1 integration start", features=len(feature_columns), iterations=iterations)
    for fold in sorted(int(item) for item in oof["fold"].dropna().unique()):
        train_idx = np.flatnonzero((oof["fold"].to_numpy(dtype=int) != fold) & np.isfinite(target))
        valid_idx = np.flatnonzero(
            (oof["fold"].to_numpy(dtype=int) == fold) & np.isfinite(target)
        )
        model = CatBoostRegressor(
            iterations=iterations,
            early_stopping_rounds=max(25, min(100, iterations // 4)),
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=8.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed + 7000 + fold,
            allow_writing_files=False,
            verbose=False,
        )
        started = perf_counter()
        model.fit(
            oof.iloc[train_idx][feature_columns],
            target[train_idx],
            eval_set=(oof.iloc[valid_idx][feature_columns], target[valid_idx]),
            use_best_model=True,
            verbose=False,
        )
        pred[valid_idx] = model.predict(oof.iloc[valid_idx][feature_columns])
        fold_frame = oof.iloc[valid_idx].copy()
        fold_plus = (
            fold_frame["schema10_oof_raw"].to_numpy(dtype=float) + pred[valid_idx]
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "valid_rows": int(len(valid_idx)),
                "schema10_rmse": _rmse(
                    fold_frame["schema10_oof_raw"], fold_frame["tvt_true"]
                ),
                "plus_student_v1_rmse": _rmse(fold_plus, fold_frame["tvt_true"]),
                "duration_sec": float(perf_counter() - started),
            }
        )
        logger.metric(
            "Surface student v1 integration fold RMSE",
            fold=fold,
            schema10_rmse=fold_rows[-1]["schema10_rmse"],
            plus_student_v1_rmse=fold_rows[-1]["plus_student_v1_rmse"],
            duration=format_duration(perf_counter() - started),
        )
    out = oof.copy()
    out["schema10_plus_student_v1_residual"] = pred
    out["schema10_plus_student_v1_raw"] = out["schema10_oof_raw"] + pred
    schema_rmse = _rmse(out["schema10_oof_raw"], out["tvt_true"])
    plus_rmse = _rmse(out["schema10_plus_student_v1_raw"], out["tvt_true"])
    metrics = {
        "feature_columns": feature_columns,
        "folds": fold_rows,
        "schema10_raw_rmse": schema_rmse,
        "schema10_plus_student_v1_rmse": plus_rmse,
        "gain_rmse": schema_rmse - plus_rmse,
        "schema10_wells": _well_summary(out, "schema10_oof_raw"),
        "schema10_plus_student_v1_wells": _well_summary(out, "schema10_plus_student_v1_raw"),
    }
    integration_path = output_dir / "integration_oof.parquet"
    try:
        out.to_parquet(integration_path, index=False)
    except Exception:
        integration_path = output_dir / "integration_oof.csv"
        out.to_csv(integration_path, index=False, float_format="%.6f")
    metrics["integration_artifact"] = str(integration_path)
    return out, metrics


def summarize_oof(oof: pd.DataFrame, integration_metrics: dict[str, Any]) -> dict[str, Any]:
    output_cols = [
        "geo_student_v1_true_tvt",
        "geo_student_v1_teacher_tvt",
        "geo_student_v1_blend_tvt",
    ]
    metrics: dict[str, Any] = {
        "rows": int(len(oof)),
        "wells": int(oof["well"].nunique()),
        "schema10_raw_rmse": _rmse(oof["schema10_oof_raw"], oof["tvt_true"]),
        "teacher_vs_true_rmse": _rmse(oof["geo_teacher_tvt"], oof["tvt_true"]),
        "integration": integration_metrics,
    }
    for column in output_cols:
        metrics[column] = {
            "rmse": _rmse(oof[column], oof["tvt_true"]),
            **_well_summary(oof, column),
        }
    error = np.abs(
        oof["geo_student_v1_blend_tvt"].to_numpy(dtype=float)
        - oof["tvt_true"].to_numpy(dtype=float)
    )
    metrics["uncertainty_error_rank_corr"] = _rank_corr(oof["student_error_pred"], error)
    metrics["ensemble_std_error_rank_corr"] = _rank_corr(oof["student_ensemble_std"], error)
    return metrics


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_empty_"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    feature_count: int,
    fold_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
) -> None:
    plus = metrics["integration"]
    lines = [
        "# Surface Student v1 Report",
        "",
        "Alignment-rich CatBoost student trained on production-available fold-safe features. "
        "`kg_path_*` direct-solver/privileged features are excluded.",
        "",
        "## Summary",
        "",
        f"- Rows: `{metrics['rows']}`",
        f"- Wells: `{metrics['wells']}`",
        f"- Feature count: `{feature_count}`",
        f"- Schema10 raw RMSE: `{metrics['schema10_raw_rmse']:.6f}`",
        f"- Teacher true RMSE: `{metrics['teacher_vs_true_rmse']:.6f}`",
        f"- v1 true-head RMSE: `{metrics['geo_student_v1_true_tvt']['rmse']:.6f}`",
        f"- v1 teacher-head RMSE: `{metrics['geo_student_v1_teacher_tvt']['rmse']:.6f}`",
        f"- v1 blend RMSE: `{metrics['geo_student_v1_blend_tvt']['rmse']:.6f}`",
        f"- Schema10 + v1 quick RMSE: `{plus['schema10_plus_student_v1_rmse']:.6f}`",
        f"- Quick gain vs schema10 raw: `{plus['gain_rmse']:.6f}`",
        f"- Student uncertainty/error rank corr: `{metrics['uncertainty_error_rank_corr']:.6f}`",
        "",
        "## Fold Feature Build",
        "",
        _markdown_table(fold_rows, ["fold", "valid_rows", "valid_wells", "duration_sec"]),
        "",
        "## Head Fold Metrics",
        "",
        _markdown_table(
            head_rows[:60],
            ["target", "fold", "valid_rows", "rmse", "best_iteration", "duration_sec"],
        ),
        "",
        "## Integration Fold Metrics",
        "",
        _markdown_table(
            plus["folds"],
            ["fold", "valid_rows", "schema10_rmse", "plus_student_v1_rmse", "duration_sec"],
        ),
        "",
    ]
    (output_dir / "surface_student_v1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_surface_student_v1(
    *,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    max_wells: int | None = None,
) -> dict[str, Any]:
    logger = RunLogger()
    cfg = config.get("surface_student_v1", {})
    seed = int(cfg.get("seed", config.get("seed", 42)))
    prepared_config = _prepare_feature_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = resolve_train_dir(data_dir, prepared_config)
    train_paths = horizontal_files(train_dir, limit=max_wells)
    logger.log(
        "RUN",
        "Surface student v1 started",
        wells=len(train_paths),
        output_dir=output_dir,
        direct_path_features=prepared_config["features"].get("include_direct_path_features"),
    )
    with logger.step("Build surface student v1 feature frame", wells=len(train_paths)):
        frame, fold_rows = build_fold_safe_feature_frame(
            train_paths=train_paths,
            config=prepared_config,
            seed=seed,
            logger=logger,
        )
    teacher_path = Path(cfg.get("teacher_rows_path", "artifacts/surface_teacher/teacher_rows.csv"))
    schema_model_path = Path(
        cfg.get("schema10_model_path", "artifacts/clearml/ed4d9dc6c7cb479881f087fee1217253/model.pkl")
    )
    schema_config_path = Path(
        cfg.get("schema10_config_path", "artifacts/clearml/ed4d9dc6c7cb479881f087fee1217253/config.yml")
    )
    frame = attach_schema10_oof(
        frame,
        model_path=schema_model_path,
        config_path=schema_config_path,
        data_dir=data_dir,
        logger=logger,
    )
    with logger.step("Attach surface teacher rows", path=teacher_path):
        frame = attach_teacher_rows(frame, load_teacher_rows(teacher_path))
    with logger.step("Build surface student v1 derived features", rows=len(frame)):
        frame = add_v1_derived_features(frame)
        feature_columns = select_v1_feature_columns(frame)
        if not feature_columns:
            raise ValueError("No safe surface-student v1 features selected")
        (output_dir / "feature_columns.json").write_text(
            json.dumps(feature_columns, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Surface student v1 feature columns selected",
            features=len(feature_columns),
            sample=",".join(feature_columns[:12]),
        )
    with logger.step("Train surface student v1 heads", targets=8, features=len(feature_columns)):
        oof, head_rows = train_heads_oof(
            frame,
            feature_columns=feature_columns,
            cfg=cfg,
            output_dir=output_dir,
            logger=logger,
        )
    with logger.step("Run surface student v1 integration smoke", rows=len(oof)):
        _integration_oof, integration_metrics = run_integration_smoke(
            oof,
            cfg=cfg,
            output_dir=output_dir,
            logger=logger,
        )
    prediction_artifact = _write_prediction_artifact(oof, output_dir)
    metrics = summarize_oof(oof, integration_metrics)
    metrics.update(
        {
            "feature_count": int(len(feature_columns)),
            "feature_columns_path": str(output_dir / "feature_columns.json"),
            "prediction_artifact": prediction_artifact,
            "integration_artifact": integration_metrics.get("integration_artifact"),
            "config": cfg,
        }
    )
    (output_dir / "surface_student_v1_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir=output_dir,
        metrics=metrics,
        feature_count=len(feature_columns),
        fold_rows=fold_rows,
        head_rows=head_rows,
    )
    logger.log(
        "DONE",
        "Surface student v1 complete",
        rows=len(oof),
        feature_count=len(feature_columns),
        blend_rmse=metrics["geo_student_v1_blend_tvt"]["rmse"],
        integration_gain=metrics["integration"]["gain_rmse"],
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train alignment-rich surface student v1.")
    parser.add_argument("--config", type=Path, default=Path("configs/surface_student_v1.yml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-wells", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or resolve_data_dir(config)
    cfg = config.get("surface_student_v1", {})
    output_dir = args.output_dir or Path(cfg.get("output_dir", "artifacts/surface_student_v1"))
    max_wells = args.max_wells
    if max_wells is None and cfg.get("max_wells") is not None:
        max_wells = int(cfg["max_wells"])
    metrics = train_surface_student_v1(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        max_wells=max_wells,
    )
    print(
        "Surface student v1 complete | "
        f"rows={metrics['rows']} blend_rmse={metrics['geo_student_v1_blend_tvt']['rmse']:.6f} "
        f"integration_gain={metrics['integration']['gain_rmse']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
