from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .constants import FORMATIONS
from .io import well_name
from .numeric import flat_tvt_prediction
from .spatial import KaggleTopContext
from .surface_teacher import build_geo_teacher_for_well
from .validation import grouped_well_folds, mask_validation_surfaces

SURFACE_STUDENT_TARGETS: tuple[str, ...] = (
    "target_geo_teacher_delta_last",
    *(f"target_surface_resid_{formation}" for formation in FORMATIONS),
)
METADATA_COLUMNS: tuple[str, ...] = (
    "id",
    "well",
    "row_index",
    "tvt_true",
    "geo_teacher_tvt",
    "geo_teacher_delta_last",
    "geo_teacher_conf",
    "last_known_tvt",
    "flat_tvt",
)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _safe_mean(values: np.ndarray, default: float = np.nan) -> float:
    finite = _finite(np.asarray(values, dtype=float))
    return float(np.mean(finite)) if len(finite) else float(default)


def _safe_std(values: np.ndarray, default: float = np.nan) -> float:
    finite = _finite(np.asarray(values, dtype=float))
    return float(np.std(finite)) if len(finite) else float(default)


def _robust_slope(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float(default)
    xx = x[mask]
    yy = y[mask]
    dx = xx - np.median(xx)
    dy = yy - np.median(yy)
    denom = float(np.dot(dx, dx))
    if denom <= 1e-9:
        return float(default)
    slope = float(np.dot(dx, dy) / denom)
    return slope if np.isfinite(slope) else float(default)


def _rolling(values: np.ndarray, window: int, method: str) -> np.ndarray:
    series = pd.Series(values, dtype=float).interpolate(limit_direction="both").bfill().ffill()
    roll = series.rolling(int(window), center=True, min_periods=1)
    if method == "std":
        return roll.std().fillna(0.0).to_numpy(dtype=float)
    return roll.mean().to_numpy(dtype=float)


def _typewell_features(
    typewell_df: pd.DataFrame | None,
    query_tvt: np.ndarray,
    observed_gr: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(query_tvt)
    if typewell_df is None or "TVT" not in typewell_df.columns or "GR" not in typewell_df.columns:
        return {
            "typewell_present": np.zeros(n, dtype=float),
            "typewell_gr_at_flat": np.full(n, np.nan, dtype=float),
            "typewell_gr_diff": np.full(n, np.nan, dtype=float),
        }
    tvt = pd.to_numeric(typewell_df["TVT"], errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(typewell_df["GR"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(tvt) & np.isfinite(gr)
    if int(mask.sum()) < 2:
        return {
            "typewell_present": np.zeros(n, dtype=float),
            "typewell_gr_at_flat": np.full(n, np.nan, dtype=float),
            "typewell_gr_diff": np.full(n, np.nan, dtype=float),
        }
    order = np.argsort(tvt[mask])
    tvt_sorted = tvt[mask][order]
    gr_sorted = gr[mask][order]
    unique_tvt, inverse = np.unique(tvt_sorted, return_inverse=True)
    counts = np.bincount(inverse)
    gr_unique = np.bincount(inverse, weights=gr_sorted) / counts
    interp = np.interp(query_tvt, unique_tvt, gr_unique, left=gr_unique[0], right=gr_unique[-1])
    return {
        "typewell_present": np.ones(n, dtype=float),
        "typewell_gr_at_flat": interp,
        "typewell_gr_diff": observed_gr - interp,
    }


def build_surface_student_features_for_well(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame | None,
    config: dict[str, Any],
    *,
    context: KaggleTopContext | None = None,
    well: str | None = None,
) -> pd.DataFrame:
    """Build test-available student features for hidden rows only."""

    n = len(horizontal_df)
    if n == 0:
        return pd.DataFrame()
    student_cfg = config.get("surface_student", {})
    tail_rows = int(student_cfg.get("tail_rows", 384))
    rolling_windows = [int(item) for item in student_cfg.get("rolling_windows", [5, 25, 101])]

    df = mask_validation_surfaces(horizontal_df)
    md = pd.to_numeric(df.get("MD"), errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df.get("X", pd.Series(np.zeros(n))), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.get("Y", pd.Series(np.zeros(n))), errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df.get("Z"), errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(df.get("GR", pd.Series(np.full(n, np.nan))), errors="coerce").to_numpy(dtype=float)
    tvt_input = pd.to_numeric(
        df.get("TVT_input", pd.Series(np.full(n, np.nan))),
        errors="coerce",
    ).to_numpy(dtype=float)

    known = np.isfinite(tvt_input)
    hidden_idx = np.flatnonzero(~known)
    if len(hidden_idx) == 0:
        return pd.DataFrame()
    known_before = np.flatnonzero(known & (np.arange(n) < int(hidden_idx[0])))
    if len(known_before) == 0:
        known_before = np.flatnonzero(known)
    if len(known_before) == 0:
        return pd.DataFrame()

    first_hidden = int(hidden_idx[0])
    last_idx = int(known_before[-1])
    first_known = int(known_before[0])
    last_tvt = float(tvt_input[last_idx])
    flat_tvt = flat_tvt_prediction(md, tvt_input, tail_rows)
    tail_start = max(first_known, last_idx - tail_rows + 1)
    tail_mask = np.zeros(n, dtype=bool)
    tail_mask[tail_start : last_idx + 1] = True
    tail_mask &= known
    tail_md = md[tail_mask]
    tail_tvt = tvt_input[tail_mask]
    tail_gr = gr[tail_mask]
    tail_slope = _robust_slope(tail_md, tail_tvt, default=0.0)
    if int(tail_mask.sum()) >= 4:
        tail_curvature = _robust_slope(tail_md[1:], np.diff(tail_tvt), default=0.0)
    else:
        tail_curvature = 0.0

    h = hidden_idx
    row = pd.DataFrame(
        {
            "row_index": h.astype(int),
            "idx_frac": h / max(n - 1, 1),
            "hidden_frac": np.linspace(0.0, 1.0, len(h)) if len(h) > 1 else np.zeros(len(h)),
            "well_n_rows": np.full(len(h), n, dtype=float),
            "hidden_n_rows": np.full(len(h), len(h), dtype=float),
            "known_count": np.full(len(h), int(known.sum()), dtype=float),
            "md": md[h],
            "x": x[h],
            "y": y[h],
            "z": z[h],
            "gr": gr[h],
            "gr_isna": (~np.isfinite(gr[h])).astype(float),
            "last_known_tvt": np.full(len(h), last_tvt, dtype=float),
            "last_known_md": np.full(len(h), md[last_idx], dtype=float),
            "last_known_z": np.full(len(h), z[last_idx], dtype=float),
            "md_from_last": md[h] - md[last_idx],
            "z_from_last": z[h] - z[last_idx],
            "x_from_last": x[h] - x[last_idx],
            "y_from_last": y[h] - y[last_idx],
            "xy_dist_from_last": np.sqrt((x[h] - x[last_idx]) ** 2 + (y[h] - y[last_idx]) ** 2),
            "flat_tvt": flat_tvt[h],
            "flat_minus_last": flat_tvt[h] - last_tvt,
            "tail_tvt_slope": np.full(len(h), tail_slope, dtype=float),
            "tail_tvt_curvature": np.full(len(h), tail_curvature, dtype=float),
            "tail_gr_mean": np.full(len(h), _safe_mean(tail_gr), dtype=float),
            "tail_gr_std": np.full(len(h), _safe_std(tail_gr), dtype=float),
            "tail_gr_nan_frac": np.full(len(h), float(np.mean(~np.isfinite(tail_gr))) if len(tail_gr) else np.nan, dtype=float),
            "hidden_gr_nan_frac": np.full(len(h), float(np.mean(~np.isfinite(gr[h]))), dtype=float),
            "known_before_hidden": np.full(len(h), first_hidden, dtype=float),
        }
    )
    gr_interp = pd.Series(gr, dtype=float).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
    for window in rolling_windows:
        row[f"gr_roll_mean_{window}"] = _rolling(gr_interp, window, "mean")[h]
        row[f"gr_roll_std_{window}"] = _rolling(gr_interp, window, "std")[h]
    row["gr_diff1"] = np.gradient(gr_interp)[h]
    row["gr_diff2"] = np.gradient(np.gradient(gr_interp))[h]
    row["gr_tail_delta"] = gr_interp[h] - _safe_mean(tail_gr)

    for name, values in _typewell_features(typewell_df, flat_tvt[h], gr_interp[h]).items():
        row[name] = values

    if context is not None:
        formation_hat, formation_dist = context.impute_formations(
            np.column_stack([x[h], y[h]]),
            self_well=well,
        )
        row["spatial_surface_dist"] = formation_dist
        for idx, formation in enumerate(FORMATIONS):
            row[f"spatial_surface_hat_{formation}"] = formation_hat[:, idx]
            row[f"z_minus_spatial_surface_hat_{formation}"] = z[h] - formation_hat[:, idx]
    else:
        row["spatial_surface_dist"] = np.nan
        for formation in FORMATIONS:
            row[f"spatial_surface_hat_{formation}"] = np.nan
            row[f"z_minus_spatial_surface_hat_{formation}"] = np.nan
    return row


def _build_student_table_for_path(
    path: Path,
    config: dict[str, Any],
    *,
    context: KaggleTopContext | None,
) -> pd.DataFrame:
    well = well_name(path)
    df = pd.read_csv(path)
    typewell_path = path.with_name(path.name.replace("__horizontal_well.csv", "__typewell.csv"))
    typewell_df = pd.read_csv(typewell_path) if typewell_path.exists() else None
    features = build_surface_student_features_for_well(
        df,
        typewell_df,
        config,
        context=context,
        well=well,
    )
    if features.empty:
        return features
    teacher = build_geo_teacher_for_well(df, typewell_df, config)
    merged = features.merge(teacher, on="row_index", how="inner")
    if merged.empty:
        return merged
    row_idx = merged["row_index"].to_numpy(dtype=int)
    z = pd.to_numeric(df["Z"], errors="coerce").to_numpy(dtype=float)
    merged.insert(0, "well", well)
    merged.insert(1, "id", [f"{well}_{int(idx)}" for idx in row_idx])
    merged["tvt_true"] = pd.to_numeric(df.loc[row_idx, "TVT"], errors="coerce").to_numpy(dtype=float)
    merged["target_geo_teacher_delta_last"] = merged["geo_teacher_delta_last"]
    for formation in FORMATIONS:
        merged[f"target_surface_resid_{formation}"] = (
            pd.to_numeric(df.loc[row_idx, formation], errors="coerce").to_numpy(dtype=float)
            - z[row_idx]
        )
    return merged


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    forbidden = set(METADATA_COLUMNS) | set(SURFACE_STUDENT_TARGETS)
    forbidden.update(
        {
            "teacher_surface_agreement",
            "teacher_surface_spread",
            "teacher_surface_fit_rmse_min",
            "teacher_surface_fit_rmse_gap",
            "teacher_missing_surfaces",
            "teacher_hidden_rows",
            "teacher_total_rows",
            "teacher_surface_rmse_spread",
            "teacher_surface_count",
            "teacher_error",
            "teacher_abs_error",
        }
    )
    forbidden.update(f"z_minus_{formation}_true" for formation in FORMATIONS)
    return [
        column
        for column in frame.columns
        if column not in forbidden
        and not column.startswith("geo_teacher_")
        and not column.startswith("teacher_")
        and not column.endswith("_true")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _fit_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    params: dict[str, Any],
    seed: int,
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
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)] if len(X_valid) > 0 else None,
        **fit_kwargs,
    )
    return model


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    err = pred[mask] - true[mask]
    return float(np.sqrt(np.mean(err * err)))


def _write_predictions(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    parquet_path = output_dir / "oof_predictions.parquet"
    csv_path = output_dir / "oof_predictions.csv"
    try:
        frame.to_parquet(parquet_path, index=False)
        return {"path": str(parquet_path), "format": "parquet", "parquet_written": True}
    except Exception as exc:
        frame.to_csv(csv_path, index=False, float_format="%.6f")
        (output_dir / "oof_predictions.parquet.unavailable.txt").write_text(
            f"Parquet engine unavailable or write failed: {exc}\n"
            f"Wrote CSV fallback: {csv_path}\n",
            encoding="utf-8",
        )
        return {
            "path": str(csv_path),
            "format": "csv",
            "parquet_written": False,
            "parquet_error": str(exc),
        }


def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}"


def _build_split_frame(
    paths: list[Path],
    config: dict[str, Any],
    *,
    context: KaggleTopContext | None,
    fold_id: int,
    split_name: str,
    progress_interval: int,
) -> pd.DataFrame:
    started = perf_counter()
    parts: list[pd.DataFrame] = []
    rows = 0
    total = len(paths)
    print(
        "Surface student table start | "
        f"fold={fold_id} split={split_name} wells={total}",
        flush=True,
    )
    for current, path in enumerate(paths, start=1):
        table = _build_student_table_for_path(path, config, context=context)
        if not table.empty:
            rows += len(table)
            parts.append(table)
        elapsed = perf_counter() - started
        if current == 1 or current == total or current % max(progress_interval, 1) == 0:
            rate = rows / elapsed if elapsed > 0 else 0.0
            eta = elapsed / current * (total - current) if current > 0 else 0.0
            print(
                "Surface student table progress | "
                f"fold={fold_id} split={split_name} current={current} total={total} "
                f"well={well_name(path)} rows={rows} rows_per_sec={rate:.1f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                flush=True,
            )
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    elapsed = perf_counter() - started
    rate = rows / elapsed if elapsed > 0 else 0.0
    print(
        "Surface student table complete | "
        f"fold={fold_id} split={split_name} wells={total} rows={rows} "
        f"rows_per_sec={rate:.1f} duration={_format_duration(elapsed)}",
        flush=True,
    )
    return frame


def train_surface_student(
    *,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    max_wells: int | None = None,
) -> dict[str, Any]:
    student_cfg = config.get("surface_student", {})
    seed = int(student_cfg.get("seed", config.get("seed", 42)))
    n_splits = int(student_cfg.get("n_splits", config.get("validation", {}).get("n_splits", 5)))
    model_cfg = dict(student_cfg.get("model", {}))
    model_name = str(model_cfg.get("name", "catboost")).lower()
    if model_name not in {"catboost", "cat"}:
        raise ValueError(f"Surface student v0 is CatBoost-only, got model.name={model_name!r}")
    params = dict(model_cfg.get("params", {}))
    targets = list(student_cfg.get("targets") or SURFACE_STUDENT_TARGETS)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"), key=well_name)
    if max_wells is not None:
        paths = paths[: int(max_wells)]
    if not paths:
        raise ValueError(f"No train wells found in {train_dir}")
    folds = grouped_well_folds(paths, n_splits=min(n_splits, len(paths)), seed=seed)
    progress_interval = int(
        student_cfg.get("progress_interval", config.get("features", {}).get("progress_interval", 25))
    )

    oof_parts: list[pd.DataFrame] = []
    all_feature_columns: list[str] | None = None
    started = perf_counter()
    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        train_paths = [paths[int(idx)] for idx in train_idx]
        valid_paths = [paths[int(idx)] for idx in valid_idx]
        fold_started = perf_counter()
        print(
            "Surface student fold start | "
            f"fold={fold_id} total_folds={len(folds)} "
            f"train_wells={len(train_paths)} valid_wells={len(valid_paths)} "
            f"targets={len(targets)}",
            flush=True,
        )
        context = None
        if bool(student_cfg.get("spatial_impute", True)):
            context_started = perf_counter()
            print(
                "Surface student context start | "
                f"fold={fold_id} context_wells={len(train_paths)}",
                flush=True,
            )
            context = KaggleTopContext(train_paths, config)
            print(
                "Surface student context complete | "
                f"fold={fold_id} duration={_format_duration(perf_counter() - context_started)}",
                flush=True,
            )
        train_frame = _build_split_frame(
            train_paths,
            config,
            context=context,
            fold_id=fold_id,
            split_name="train",
            progress_interval=progress_interval,
        )
        valid_frame = _build_split_frame(
            valid_paths,
            config,
            context=context,
            fold_id=fold_id,
            split_name="valid",
            progress_interval=progress_interval,
        )
        if train_frame.empty or valid_frame.empty:
            raise ValueError(f"Fold {fold_id} produced empty train/valid frame")
        feature_columns = _feature_columns(train_frame)
        if all_feature_columns is None:
            all_feature_columns = feature_columns
        else:
            all_feature_columns = sorted(set(all_feature_columns).union(feature_columns))
        for column in all_feature_columns:
            if column not in train_frame.columns:
                train_frame[column] = np.nan
            if column not in valid_frame.columns:
                valid_frame[column] = np.nan
        feature_columns = list(all_feature_columns)
        fold_pred = valid_frame[list(METADATA_COLUMNS)].copy()
        fold_pred["fold"] = fold_id
        for target in targets:
            if target not in train_frame.columns:
                continue
            y_train = train_frame[target].to_numpy(dtype=float)
            y_valid = valid_frame[target].to_numpy(dtype=float)
            valid_train = np.isfinite(y_train)
            valid_valid = np.isfinite(y_valid)
            if int(valid_train.sum()) < 32 or int(valid_valid.sum()) == 0:
                fold_pred[target.replace("target_", "pred_")] = np.nan
                print(
                    "Surface student target skipped | "
                    f"fold={fold_id} target={target} "
                    f"train_rows={int(valid_train.sum())} valid_rows={int(valid_valid.sum())}",
                    flush=True,
                )
                continue
            target_started = perf_counter()
            print(
                "Surface student target start | "
                f"fold={fold_id} target={target} features={len(feature_columns)} "
                f"train_rows={int(valid_train.sum())} valid_rows={int(valid_valid.sum())}",
                flush=True,
            )
            model = _fit_catboost(
                train_frame.loc[valid_train, feature_columns],
                y_train[valid_train],
                valid_frame.loc[valid_valid, feature_columns],
                y_valid[valid_valid],
                params,
                seed + 1000 * fold_id + len(target),
            )
            _save_model(model, model_dir / target / f"fold_{fold_id}.pkl")
            pred = np.full(len(valid_frame), np.nan, dtype=float)
            pred[valid_valid] = model.predict(valid_frame.loc[valid_valid, feature_columns])
            fold_pred[target.replace("target_", "pred_")] = pred
            target_rmse = _rmse(pred, y_valid)
            best_iteration = getattr(model, "get_best_iteration", lambda: None)()
            print(
                "Surface student target complete | "
                f"fold={fold_id} target={target} rmse={target_rmse:.6f} "
                f"best_iteration={best_iteration} "
                f"duration={_format_duration(perf_counter() - target_started)}",
                flush=True,
            )
        fold_pred["geo_student_delta_last"] = fold_pred.get(
            "pred_geo_teacher_delta_last", np.nan
        )
        fold_pred["geo_student_tvt"] = (
            fold_pred["last_known_tvt"] + fold_pred["geo_student_delta_last"]
        )
        fold_pred["geo_student_minus_flat"] = fold_pred["geo_student_tvt"] - fold_pred["flat_tvt"]
        fold_pred["geo_student_uncertainty"] = np.abs(
            fold_pred["geo_student_tvt"] - fold_pred["geo_teacher_tvt"]
        )
        z_values = valid_frame["z"].to_numpy(dtype=float)
        for formation in FORMATIONS:
            pred_col = f"pred_surface_resid_{formation}"
            if pred_col in fold_pred.columns:
                fold_pred[f"surface_hat_{formation}"] = z_values + fold_pred[pred_col]
                true_surface = z_values + valid_frame[f"target_surface_resid_{formation}"].to_numpy(dtype=float)
                fold_pred[f"surface_unc_{formation}"] = np.abs(
                    fold_pred[f"surface_hat_{formation}"] - true_surface
                )
            else:
                fold_pred[f"surface_hat_{formation}"] = np.nan
                fold_pred[f"surface_unc_{formation}"] = np.nan
        fold_pred["student_vs_pf_ancc"] = np.nan
        fold_pred["student_vs_dtw"] = np.nan
        fold_pred["student_vs_dwt"] = np.nan
        fold_pred["student_vs_schema10"] = np.nan
        oof_parts.append(fold_pred)
        elapsed = perf_counter() - started
        print(
            "Surface student fold complete | "
            f"fold={fold_id} valid_rows={len(valid_frame)} "
            f"geo_rmse={_rmse(fold_pred['geo_student_tvt'].to_numpy(float), fold_pred['geo_teacher_tvt'].to_numpy(float)):.6f} "
            f"fold_duration={_format_duration(perf_counter() - fold_started)} "
            f"elapsed={_format_duration(elapsed)}",
            flush=True,
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    print(
        "Surface student write outputs | "
        f"rows={len(oof)} output_dir={output_dir}",
        flush=True,
    )
    prediction_artifact = _write_predictions(oof, output_dir)
    teacher_rmse = _rmse(oof["geo_student_tvt"].to_numpy(float), oof["geo_teacher_tvt"].to_numpy(float))
    true_rmse = _rmse(oof["geo_student_tvt"].to_numpy(float), oof["tvt_true"].to_numpy(float))
    teacher_true_rmse = _rmse(oof["geo_teacher_tvt"].to_numpy(float), oof["tvt_true"].to_numpy(float))
    metrics = {
        "rows": int(len(oof)),
        "wells": int(oof["well"].nunique()),
        "folds": int(len(folds)),
        "model_backend": "catboost",
        "targets": targets,
        "feature_count": int(len(all_feature_columns or [])),
        "features": all_feature_columns or [],
        "prediction_artifact": prediction_artifact,
        "student_vs_teacher_rmse": teacher_rmse,
        "student_vs_true_rmse": true_rmse,
        "teacher_vs_true_rmse": teacher_true_rmse,
        "mean_uncertainty": float(np.nanmean(oof["geo_student_uncertainty"].to_numpy(float))),
    }
    (output_dir / "surface_student_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    _write_report(output_dir, metrics, oof)
    return metrics


def _write_report(output_dir: Path, metrics: dict[str, Any], oof: pd.DataFrame) -> None:
    by_well = (
        oof.assign(err=oof["geo_student_tvt"] - oof["geo_teacher_tvt"])
        .groupby("well")
        .agg(rows=("well", "size"), rmse=("err", lambda x: float(np.sqrt(np.nanmean(np.asarray(x) ** 2)))))
        .reset_index()
        .sort_values("rmse", ascending=False)
    )
    lines = [
        "# Surface Student v0 Report",
        "",
        "CatBoost student trained only on test-available inputs plus fold-safe spatial imputation.",
        "",
        "## Summary",
        "",
        f"- Rows: `{metrics['rows']}`",
        f"- Wells: `{metrics['wells']}`",
        f"- Folds: `{metrics['folds']}`",
        f"- Model backend: `{metrics['model_backend']}`",
        f"- Feature count: `{metrics['feature_count']}`",
        f"- Student vs teacher RMSE: `{metrics['student_vs_teacher_rmse']:.6f}`",
        f"- Student vs true RMSE: `{metrics['student_vs_true_rmse']:.6f}`",
        f"- Teacher vs true RMSE: `{metrics['teacher_vs_true_rmse']:.6f}`",
        f"- Mean OOF uncertainty: `{metrics['mean_uncertainty']:.6f}`",
        f"- Prediction artifact: `{metrics['prediction_artifact']['path']}`",
        f"- Parquet written: `{metrics['prediction_artifact']['parquet_written']}`",
        "",
        "## Worst Wells vs Teacher",
        "",
        "| well | rows | RMSE |",
        "| --- | ---: | ---: |",
    ]
    for _, row in by_well.head(10).iterrows():
        lines.append(f"| `{row['well']}` | `{int(row['rows'])}` | `{float(row['rmse']):.6f}` |")
    lines.append("")
    (output_dir / "surface_student_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fold-safe surface student v0.")
    parser.add_argument("--config", type=Path, default=Path("configs/surface_student_gbm.yml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-wells", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    student_cfg = config.get("surface_student", {})
    data_dir = args.data_dir or Path(config.get("data", {}).get("data_dir") or "data")
    output_dir = args.output_dir or Path(student_cfg.get("output_dir", "artifacts/surface_student"))
    max_wells = args.max_wells
    if max_wells is None and student_cfg.get("max_wells") is not None:
        max_wells = int(student_cfg["max_wells"])
    metrics = train_surface_student(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        max_wells=max_wells,
    )
    print(
        "Surface student complete | "
        f"rows={metrics['rows']} student_vs_teacher_rmse={metrics['student_vs_teacher_rmse']:.6f} "
        f"student_vs_true_rmse={metrics['student_vs_true_rmse']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
