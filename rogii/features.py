from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .constants import FORMATION_ORDER
from .io import typewell_path, well_name
from .numeric import (
    as_float_array,
    centered_rolling,
    flat_tvt_prediction,
    safe_gradient,
    tail_slope,
    tail_stat,
)
from .runlog import RunLogger, format_duration
from .spatial import KaggleTopContext
from .top_signals import build_kaggle_top_signal_features

FEATURE_CACHE_SCHEMA_VERSION = 7


@dataclass(frozen=True)
class WellFeatures:
    well: str
    features: pd.DataFrame
    flat_prediction: np.ndarray
    target: np.ndarray | None
    target_mask: np.ndarray


def choose_prediction_baseline(
    config: dict[str, Any],
    flat_pred: np.ndarray,
    last_tvt: float,
) -> np.ndarray:
    baseline = str(config["features"].get("prediction_baseline", "flat_tvt")).lower()
    if baseline in {"flat", "flat_tvt"}:
        return flat_pred
    if baseline in {"last_known", "last_known_tvt"}:
        return np.full(len(flat_pred), float(last_tvt), dtype=float)
    raise ValueError(
        "features.prediction_baseline must be 'flat_tvt' or 'last_known_tvt'."
    )


def feature_cache_path(
    horizontal_path: Path,
    config: dict[str, Any],
    train: bool,
    context_key: str | None = None,
) -> Path | None:
    cache_cfg = config["features"].get("cache") or {}
    if not cache_cfg.get("enabled", False):
        return None

    type_path = typewell_path(horizontal_path)
    stats = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "horizontal": {
            "name": horizontal_path.name,
            "size": horizontal_path.stat().st_size,
            "mtime_ns": horizontal_path.stat().st_mtime_ns,
        },
        "typewell": None,
        "train": bool(train),
        "context_key": context_key,
        "features": config.get("features", {}),
        "target_rows": config.get("data", {}).get("target_rows", "hidden_only"),
    }
    if type_path is not None:
        stats["typewell"] = {
            "name": type_path.name,
            "size": type_path.stat().st_size,
            "mtime_ns": type_path.stat().st_mtime_ns,
        }
    payload = json.dumps(stats, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    cache_dir = Path(cache_cfg.get("dir", "artifacts/feature_cache"))
    return cache_dir / f"{well_name(horizontal_path)}_{digest}.pkl"


def read_typewell_features(
    path: Path | None, horizontal_gr: np.ndarray, flat_pred: np.ndarray
) -> dict[str, np.ndarray | float]:
    n = len(horizontal_gr)
    if path is None:
        return {
            "typewell_tvt_min": 0.0,
            "typewell_tvt_max": 0.0,
            "typewell_tvt_range": 0.0,
            "typewell_gr_mean": 0.0,
            "typewell_gr_std": 0.0,
            "typewell_nearest_tvt_by_gr": np.zeros(n, dtype=float),
            "typewell_nearest_gr_diff": np.zeros(n, dtype=float),
            "typewell_nearest_geology_code": np.full(n, -1.0, dtype=float),
            "typewell_nearest_tvt_minus_flat": np.zeros(n, dtype=float),
        }

    typewell = pd.read_csv(path)
    tvt = as_float_array(typewell.get("TVT", pd.Series(dtype=float)))
    gr = as_float_array(typewell.get("GR", pd.Series(dtype=float)))
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if valid.sum() == 0:
        return read_typewell_features(None, horizontal_gr, flat_pred)

    tvt_valid = tvt[valid]
    gr_valid = gr[valid]
    geology = typewell.get("Geology")
    if geology is None:
        geology_codes = np.full(len(typewell), -1.0, dtype=float)
    else:
        geology_codes = (
            geology.astype(str).map(FORMATION_ORDER).fillna(-1).to_numpy(dtype=float)
        )
    geology_valid = geology_codes[valid]

    order = np.argsort(gr_valid)
    gr_sorted = gr_valid[order]
    tvt_sorted = tvt_valid[order]
    geology_sorted = geology_valid[order]

    positions = np.searchsorted(gr_sorted, horizontal_gr, side="left")
    left = np.clip(positions - 1, 0, len(gr_sorted) - 1)
    right = np.clip(positions, 0, len(gr_sorted) - 1)
    choose_right = np.abs(gr_sorted[right] - horizontal_gr) < np.abs(
        gr_sorted[left] - horizontal_gr
    )
    nearest = np.where(choose_right, right, left)
    nearest_tvt = tvt_sorted[nearest]
    nearest_gr = gr_sorted[nearest]
    nearest_geology = geology_sorted[nearest]

    return {
        "typewell_tvt_min": float(np.nanmin(tvt_valid)),
        "typewell_tvt_max": float(np.nanmax(tvt_valid)),
        "typewell_tvt_range": float(np.nanmax(tvt_valid) - np.nanmin(tvt_valid)),
        "typewell_gr_mean": float(np.nanmean(gr_valid)),
        "typewell_gr_std": float(np.nanstd(gr_valid)),
        "typewell_nearest_tvt_by_gr": nearest_tvt,
        "typewell_nearest_gr_diff": np.abs(nearest_gr - horizontal_gr),
        "typewell_nearest_geology_code": nearest_geology,
        "typewell_nearest_tvt_minus_flat": nearest_tvt - flat_pred,
    }


def build_target_mask(df: pd.DataFrame, target_rows: str) -> np.ndarray:
    if target_rows not in {"all", "hidden_only"}:
        raise ValueError("data.target_rows must be 'hidden_only' or 'all'")

    if "TVT" not in df.columns:
        return np.zeros(len(df), dtype=bool)

    has_target = pd.to_numeric(df["TVT"], errors="coerce").notna().to_numpy()
    if target_rows == "all":
        return has_target
    if "TVT_input" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    hidden = pd.to_numeric(df["TVT_input"], errors="coerce").isna().to_numpy()
    return has_target & hidden


def build_well_features(
    horizontal_path: Path,
    config: dict[str, Any],
    train: bool,
    top_context: KaggleTopContext | None = None,
    logger: RunLogger | None = None,
) -> WellFeatures:
    context_key = getattr(top_context, "context_key", None)
    cache_path = feature_cache_path(horizontal_path, config, train, context_key)
    profile_stages = bool(config["features"].get("profile_stages", False))
    if cache_path is not None and cache_path.is_file():
        try:
            with cache_path.open("rb") as file:
                cached = pickle.load(file)
            if isinstance(cached, WellFeatures):
                return cached
        except Exception as exc:
            if logger is not None:
                logger.warn("Ignoring feature cache", path=cache_path, error=exc)

    stage_started_at = perf_counter()
    df = pd.read_csv(horizontal_path)
    if logger is not None and profile_stages:
        logger.info(
            "Feature stage",
            stage="well.read_horizontal",
            well=well_name(horizontal_path),
            duration_sec=perf_counter() - stage_started_at,
        )
    n = len(df)
    well = well_name(horizontal_path)
    idx = np.arange(n, dtype=float)

    md = as_float_array(df.get("MD", pd.Series(np.arange(n))))
    x = as_float_array(df.get("X", pd.Series(np.zeros(n))))
    y = as_float_array(df.get("Y", pd.Series(np.zeros(n))))
    z = as_float_array(df.get("Z", pd.Series(np.zeros(n))))
    gr = as_float_array(df.get("GR", pd.Series(np.zeros(n))), default=np.nan)
    tvt_input = as_float_array(
        df.get("TVT_input", pd.Series(np.full(n, np.nan))), default=np.nan
    )

    tail_windows = [int(item) for item in config["features"]["tail_windows"]]
    rolling_windows = [int(item) for item in config["features"]["rolling_windows"]]
    flat_pred = flat_tvt_prediction(md, tvt_input, max(tail_windows))

    known = np.isfinite(tvt_input)
    known_idx = np.flatnonzero(known)
    if len(known_idx):
        first_known = int(known_idx[0])
        last_known = int(known_idx[-1])
        anchor_tvt = np.full(
            n,
            float(np.nanmedian(tvt_input[known_idx[-max(tail_windows) :]])),
            dtype=float,
        )
        first_tvt = float(tvt_input[first_known])
        last_tvt = float(tvt_input[last_known])
    else:
        first_known = 0
        last_known = 0
        anchor_tvt = np.zeros(n, dtype=float)
        first_tvt = 0.0
        last_tvt = 0.0

    base_pred = choose_prediction_baseline(config, flat_pred, last_tvt)

    prev_known_idx = (
        pd.Series(np.where(known, idx, np.nan))
        .ffill()
        .bfill()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    prev_known_tvt = (
        pd.Series(tvt_input)
        .ffill()
        .bfill()
        .fillna(float(np.nanmedian(flat_pred)))
        .to_numpy(dtype=float)
    )
    prev_known_md = np.interp(prev_known_idx, idx, md)
    prev_known_z = np.interp(prev_known_idx, idx, z)
    hidden_idx = np.flatnonzero(~known)
    hidden_frac = np.zeros(n, dtype=float)
    if len(hidden_idx):
        hidden_frac[hidden_idx] = np.arange(len(hidden_idx), dtype=float) / max(
            len(hidden_idx) - 1, 1
        )

    gr_fill_value = float(np.nanmean(gr)) if np.isfinite(gr).any() else 0.0
    gr_filled = (
        pd.Series(gr, dtype=float)
        .interpolate(limit_direction="both")
        .fillna(gr_fill_value)
    )
    md_delta = pd.Series(md, dtype=float).diff().replace(0.0, np.nan)
    dzdmd = (pd.Series(z, dtype=float).diff() / md_delta).fillna(0.0)
    dxdmd = (pd.Series(x, dtype=float).diff() / md_delta).fillna(0.0)
    dydmd = (pd.Series(y, dtype=float).diff() / md_delta).fillna(0.0)

    features: dict[str, np.ndarray | float] = {
        "idx": idx,
        "idx_frac": idx / max(n - 1, 1),
        "well_n_rows": float(n),
        "known_count": float(len(known_idx)),
        "known_frac": float(len(known_idx) / max(n, 1)),
        "md": md,
        "x": x,
        "y": y,
        "z": z,
        "gr": gr,
        "flat_tvt": flat_pred,
        "baseline_tvt": base_pred,
        "baseline_minus_flat": base_pred - flat_pred,
        "tvt_input_isna": (~known).astype(float),
        "first_known_idx": float(first_known),
        "last_known_idx": float(last_known),
        "first_known_tvt": first_tvt,
        "last_known_tvt": last_tvt,
        "anchor_tvt": anchor_tvt,
        "idx_from_last_known": idx - float(last_known),
        "md_from_start": md - md[0],
        "md_from_last_known": md - md[last_known],
        "z_from_last_known": z - z[last_known],
        "x_from_last_known": x - x[last_known],
        "y_from_last_known": y - y[last_known],
        "frac": hidden_frac,
        "frac2": hidden_frac**2,
        "sqrt_frac": np.sqrt(hidden_frac),
        "xy_dist_from_last_known": np.sqrt(
            (x - x[last_known]) ** 2 + (y - y[last_known]) ** 2
        ),
        "dzdmd": dzdmd.to_numpy(dtype=float),
        "dxdmd": dxdmd.to_numpy(dtype=float),
        "dydmd": dydmd.to_numpy(dtype=float),
        "prev_known_tvt": prev_known_tvt,
        "idx_from_prev_known": idx - prev_known_idx,
        "md_from_prev_known": md - prev_known_md,
        "z_from_prev_known": z - prev_known_z,
        "gr_from_well_mean": gr - float(np.nanmean(gr)),
        "gr_grad_md": safe_gradient(gr, md),
        "z_grad_md": safe_gradient(z, md),
        "x_grad_md": safe_gradient(x, md),
        "y_grad_md": safe_gradient(y, md),
    }

    for window in tail_windows:
        features[f"tail_tvt_median_{window}"] = tail_stat(
            tvt_input, known, window, "median"
        )
        features[f"tail_tvt_mean_{window}"] = tail_stat(
            tvt_input, known, window, "mean"
        )
        features[f"tail_tvt_std_{window}"] = tail_stat(tvt_input, known, window, "std")
        features[f"tail_tvt_slope_md_{window}"] = tail_slope(
            md, tvt_input, known, window
        )
        features[f"tail_z_slope_md_{window}"] = tail_slope(
            md, z, np.isfinite(md) & np.isfinite(z), window
        )
        features[f"tail_gr_mean_{window}"] = tail_stat(
            gr, np.isfinite(gr), window, "mean"
        )
        features[f"tail_gr_std_{window}"] = tail_stat(
            gr, np.isfinite(gr), window, "std"
        )

    for window in rolling_windows:
        gr_mean = centered_rolling(gr, window, "mean")
        gr_std = centered_rolling(gr, window, "std")
        features[f"gr_roll_mean_{window}"] = gr_mean
        features[f"gr_roll_std_{window}"] = gr_std
        features[f"gr_roll_min_{window}"] = centered_rolling(gr, window, "min")
        features[f"gr_roll_max_{window}"] = centered_rolling(gr, window, "max")
        features[f"gr_minus_roll_mean_{window}"] = gr - gr_mean
        z_roll_mean = centered_rolling(z, window, "mean")
        features[f"z_roll_mean_{window}"] = z_roll_mean
        features[f"z_minus_roll_mean_{window}"] = z - z_roll_mean

    for window in [5, 21, 51, 101]:
        rolled = gr_filled.rolling(window, center=True, min_periods=1)
        features[f"grm{window}"] = rolled.mean().to_numpy(dtype=float)
        features[f"grs{window}"] = rolled.std().fillna(0.0).to_numpy(dtype=float)
    for lag in [1, 5, 15, 30]:
        features[f"glag{lag}"] = gr_filled.shift(lag).bfill().to_numpy(dtype=float)
        features[f"glead{lag}"] = gr_filled.shift(-lag).ffill().to_numpy(dtype=float)
    features["gr_d1"] = gr_filled.diff().fillna(0.0).to_numpy(dtype=float)
    features["gr_d2"] = gr_filled.diff().diff().fillna(0.0).to_numpy(dtype=float)
    features["gr_env"] = (
        gr_filled.rolling(21, center=True, min_periods=1).max().to_numpy(dtype=float)
    )
    features["gr_nrg"] = np.sqrt(
        np.maximum(
            (gr_filled**2)
            .rolling(21, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float),
            0.0,
        )
    )

    if config["features"].get("include_typewell", True):
        stage_started_at = perf_counter()
        features.update(
            read_typewell_features(typewell_path(horizontal_path), gr, flat_pred)
        )
        if logger is not None and profile_stages:
            logger.info(
                "Feature stage",
                stage="well.typewell",
                well=well,
                duration_sec=perf_counter() - stage_started_at,
            )

    if config["features"].get("include_kaggle_top_signals", False):
        stage_started_at = perf_counter()
        features.update(
            build_kaggle_top_signal_features(
                df,
                horizontal_path,
                top_context,
                md,
                x,
                y,
                z,
                gr,
                tvt_input,
                flat_pred,
                config,
                train,
                logger,
            )
        )
        if logger is not None and profile_stages:
            logger.info(
                "Feature stage",
                stage="well.top_signals_total",
                well=well,
                duration_sec=perf_counter() - stage_started_at,
            )

    stage_started_at = perf_counter()
    feature_frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
    if logger is not None and profile_stages:
        logger.info(
            "Feature stage",
            stage="well.materialize_frame",
            well=well,
            columns=len(feature_frame.columns),
            duration_sec=perf_counter() - stage_started_at,
        )
    target_mask = (
        build_target_mask(df, config["data"].get("target_rows", "hidden_only"))
        if train
        else np.zeros(n, dtype=bool)
    )
    target = (
        as_float_array(df["TVT"], default=np.nan)
        if train and "TVT" in df.columns
        else None
    )
    well_features = WellFeatures(
        well=well,
        features=feature_frame,
        flat_prediction=base_pred,
        target=target,
        target_mask=target_mask,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as file:
            pickle.dump(well_features, file)
    return well_features


def build_training_table(
    paths: list[Path],
    config: dict[str, Any],
    top_context: KaggleTopContext | None = None,
    logger: RunLogger | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_parts: list[pd.DataFrame] = []
    residual_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    flat_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    loaded_rows = 0
    started_at = perf_counter()
    progress_interval = int(config.get("features", {}).get("progress_interval") or 25)
    progress_interval = max(1, progress_interval)

    if logger is not None:
        logger.info("Feature table progress", current=0, total=len(paths), rows=0)

    for i, path in enumerate(paths, start=1):
        if logger is not None and (i == 1 or i % progress_interval == 0):
            logger.info(
                "Build well features",
                current=i,
                total=len(paths),
                well=well_name(path),
            )
        wf = build_well_features(
            path, config, train=True, top_context=top_context, logger=logger
        )
        mask = wf.target_mask
        if wf.target is None or not mask.any():
            continue
        feature_parts.append(wf.features.loc[mask].astype("float32"))
        true_values = wf.target[mask]
        flat_values = wf.flat_prediction[mask]
        residual_parts.append((true_values - flat_values).astype("float32"))
        group_parts.append(np.full(mask.sum(), wf.well))
        flat_parts.append(flat_values.astype("float32"))
        true_parts.append(true_values.astype("float32"))
        loaded_rows += int(mask.sum())
        if logger is not None and (
            i == 1 or i % progress_interval == 0 or i == len(paths)
        ):
            elapsed = perf_counter() - started_at
            eta = elapsed / max(i, 1) * max(len(paths) - i, 0)
            logger.info(
                "Loaded train wells",
                current=i,
                total=len(paths),
                rows=loaded_rows,
                elapsed=format_duration(elapsed),
                eta=format_duration(eta),
            )

    if not feature_parts:
        raise ValueError(
            "No training rows were built. Check data.target_rows and train files."
        )

    X = pd.concat(feature_parts, axis=0, ignore_index=True)
    residual = np.concatenate(residual_parts)
    groups = np.concatenate(group_parts)
    flat = np.concatenate(flat_parts)
    y_true = np.concatenate(true_parts)
    return X, residual, groups, flat, y_true
