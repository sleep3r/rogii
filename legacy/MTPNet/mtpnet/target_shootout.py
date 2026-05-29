"""Experiment 1: Target Formulation Shootout.

Compares five TVT target formulations using LightGBM + GroupKFold(5) to find
the easiest target before building a DL model.

Formulations
------------
T0  raw TVT                             base = 0
T1  TVT + Z  (= plane_coord, Exp 0)     base = -Z   → pred_TVT = pred_T1 - Z
T2  TVT - linear_bridge(MD)             base = bridge  (global linear on known tail)
T3  TVT - linear(MD, Z)                 base = bridge  (2-D linear on known tail)
T4  TVT - last_slope_bridge(MD)         base = bridge  (local slope from last known pt)

Schema safety
-------------
All features are asserted free of FORBIDDEN_INFERENCE_COLUMNS before training.
Forbidden: TVT, Geology, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA.

GO condition
------------
Any T1–T4 beats T0 by mean_well_rmse >= 0.15 ft OOF
AND improves tail-class RMSE (if tail labels available).
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .io import discover_wells, load_well
from .schema_safe import FORBIDDEN_INFERENCE_COLUMNS, assert_schema_safe_columns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_NAMES = ("T0", "T1", "T2", "T3", "T4")

TARGET_DESCRIPTIONS = {
    "T0": "TVT (raw)",
    "T1": "TVT + Z  (plane_coord)",
    "T2": "TVT - linear_bridge(MD)",
    "T3": "TVT - linear(MD, Z)",
    "T4": "TVT - last_slope_bridge(MD)",
}

FEATURE_COLS: list[str] = [
    # Position
    "MD", "X", "Y", "Z",
    # Trajectory
    "dx_dmd", "dy_dmd", "dz_dmd",
    "azi_sin", "azi_cos",
    "inclination", "dogleg",
    # GR local
    "gr_local_mean", "gr_local_std", "gr_deriv", "gr_nan_frac",
    # GR global
    "gr_well_mean", "gr_well_std",
    # Known-tail stats
    "last_known_tvt", "known_slope", "known_std", "known_n",
    # Position relative to known tail
    "dist_from_last_known", "hidden_progress",
    # Typewell neighbour stats
    "tw_tvt_mean", "tw_tvt_std", "tw_tvt_range", "tw_n",
    "tw_gr_mean", "tw_gr_std",
    "tw_gr_at_est",
]


@dataclass(frozen=True)
class ShootoutConfig:
    n_folds: int = 5
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 64
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    seed: int = 42
    k_wells: int = -1
    n_jobs: int = -1
    go_min_improvement_ft: float = 0.15
    tail_classes_path: Path | None = None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _safe_gradient(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """np.gradient with fallback for constant x."""
    dx = np.diff(x)
    if np.all(np.abs(dx) < 1e-9):
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values.astype(np.float64), x.astype(np.float64))


def _rolling_stats(
    arr: np.ndarray, window: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rolling mean, std, nan_frac for arr."""
    s = pd.Series(arr)
    kw = dict(window=window, center=True, min_periods=1)
    mean = s.rolling(**kw).mean().to_numpy(dtype=np.float64)
    std = s.rolling(**kw).std().to_numpy(dtype=np.float64)
    nan_frac = s.isna().rolling(**kw).mean().to_numpy(dtype=np.float64)
    std = np.where(np.isfinite(std), std, 0.0)
    return mean, std, nan_frac


def build_well_features(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> pd.DataFrame:
    """Return one feature row per HIDDEN row (TVT_input is NaN)."""
    n = len(horizontal)
    md = horizontal["MD"].to_numpy(dtype=np.float64)
    x = horizontal["X"].to_numpy(dtype=np.float64)
    y = horizontal["Y"].to_numpy(dtype=np.float64)
    z = horizontal["Z"].to_numpy(dtype=np.float64)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    tvt_true = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    known_mask = np.isfinite(tvt_input)

    # -- Trajectory derivatives ------------------------------------------------
    dx_dmd = _safe_gradient(x, md)
    dy_dmd = _safe_gradient(y, md)
    dz_dmd = _safe_gradient(z, md)

    # -- Azimuth ---------------------------------------------------------------
    azi = np.arctan2(dy_dmd, dx_dmd)
    azi_sin = np.sin(azi)
    azi_cos = np.cos(azi)

    # -- Inclination proxy (angle from vertical) --------------------------------
    horiz = np.sqrt(dx_dmd**2 + dy_dmd**2)
    incl = np.arctan2(horiz, np.abs(dz_dmd) + 1e-9)

    # -- Dogleg severity -------------------------------------------------------
    ddx = _safe_gradient(dx_dmd, md)
    ddy = _safe_gradient(dy_dmd, md)
    ddz = _safe_gradient(dz_dmd, md)
    dogleg = np.sqrt(ddx**2 + ddy**2 + ddz**2)

    # -- GR stats --------------------------------------------------------------
    gr_filled = np.where(np.isfinite(gr_raw), gr_raw, np.nanmean(gr_raw) if np.isfinite(gr_raw).any() else 0.0)
    gr_mean, gr_std, gr_nan_frac = _rolling_stats(gr_raw, window=32)
    gr_deriv = _safe_gradient(gr_filled, md)
    gr_well_mean = float(np.nanmean(gr_raw)) if np.isfinite(gr_raw).any() else 0.0
    gr_well_std = float(np.nanstd(gr_raw)) if np.isfinite(gr_raw).any() else 0.0

    # -- Known-tail stats ------------------------------------------------------
    if known_mask.sum() >= 2:
        md_k = md[known_mask]
        tvt_k = tvt_input[known_mask]
        last_known_md = float(md_k[-1])
        last_known_tvt = float(tvt_k[-1])
        n_use = max(2, int(len(md_k) * 0.2))
        coef = np.polyfit(md_k[-n_use:], tvt_k[-n_use:], 1)
        known_slope = float(coef[0])
        residuals = tvt_k[-n_use:] - np.polyval(coef, md_k[-n_use:])
        known_std = float(np.std(residuals))
        known_n = int(known_mask.sum())
    elif known_mask.sum() == 1:
        last_known_md = float(md[known_mask][0])
        last_known_tvt = float(tvt_input[known_mask][0])
        known_slope = 0.0
        known_std = 0.0
        known_n = 1
    else:
        last_known_md = float(md[0])
        last_known_tvt = float(np.nanmean(tvt_true)) if np.isfinite(tvt_true).any() else 0.0
        known_slope = 0.0
        known_std = 0.0
        known_n = 0

    dist_from_last_known = md - last_known_md
    hidden_len = float(md[-1]) - last_known_md
    hidden_progress = dist_from_last_known / max(hidden_len, 1.0)

    # -- Typewell stats --------------------------------------------------------
    tw_tvt_raw = pd.to_numeric(typewell["TVT"], errors="coerce").dropna().to_numpy(
        dtype=np.float64
    )
    tw_gr_raw = pd.to_numeric(typewell["GR"], errors="coerce").dropna().to_numpy(
        dtype=np.float64
    )
    if len(tw_tvt_raw) > 1:
        tw_tvt_mean = float(np.mean(tw_tvt_raw))
        tw_tvt_std = float(np.std(tw_tvt_raw))
        tw_tvt_range = float(tw_tvt_raw.max() - tw_tvt_raw.min())
        tw_n = int(len(tw_tvt_raw))
    else:
        tw_tvt_mean = tw_tvt_std = tw_tvt_range = float("nan")
        tw_n = 0
    if len(tw_gr_raw) > 1:
        tw_gr_mean = float(np.mean(tw_gr_raw))
        tw_gr_std = float(np.std(tw_gr_raw))
    else:
        tw_gr_mean = tw_gr_std = float("nan")

    # Estimated TVT at each row (linear anchor from known tail)
    est_tvt = last_known_tvt + known_slope * dist_from_last_known

    if len(tw_tvt_raw) >= 2 and len(tw_gr_raw) >= 2:
        min_len = min(len(tw_tvt_raw), len(tw_gr_raw))
        order = np.argsort(tw_tvt_raw[:min_len])
        tw_gr_at_est = np.interp(
            est_tvt,
            tw_tvt_raw[:min_len][order],
            tw_gr_raw[:min_len][order],
        )
    else:
        tw_gr_at_est = np.full(n, float("nan"))

    # -- Assemble all rows (filter to hidden below) ----------------------------
    feat = pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "dx_dmd": dx_dmd,
            "dy_dmd": dy_dmd,
            "dz_dmd": dz_dmd,
            "azi_sin": azi_sin,
            "azi_cos": azi_cos,
            "inclination": incl,
            "dogleg": dogleg,
            "gr_local_mean": gr_mean,
            "gr_local_std": gr_std,
            "gr_deriv": gr_deriv,
            "gr_nan_frac": gr_nan_frac,
            "gr_well_mean": gr_well_mean,
            "gr_well_std": gr_well_std,
            "last_known_tvt": last_known_tvt,
            "known_slope": known_slope,
            "known_std": known_std,
            "known_n": known_n,
            "dist_from_last_known": dist_from_last_known,
            "hidden_progress": hidden_progress,
            "tw_tvt_mean": tw_tvt_mean,
            "tw_tvt_std": tw_tvt_std,
            "tw_tvt_range": tw_tvt_range,
            "tw_n": tw_n,
            "tw_gr_mean": tw_gr_mean,
            "tw_gr_std": tw_gr_std,
            "tw_gr_at_est": tw_gr_at_est,
            # metadata
            "_well_id": well_id,
            "_tvt_true": tvt_true,
            "_is_hidden": ~known_mask,
        }
    )
    # keep hidden rows only
    feat = feat[feat["_is_hidden"]].reset_index(drop=True)
    return feat


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------

def _linear_bridge_md(horizontal: pd.DataFrame, target_md: np.ndarray) -> np.ndarray:
    """Fit a global linear (MD → TVT_input) on the last 20% of known rows."""
    known = horizontal[horizontal["TVT_input"].notna()]
    if len(known) < 2:
        return np.zeros(len(target_md))
    md_k = known["MD"].to_numpy(dtype=np.float64)
    tvt_k = known["TVT_input"].to_numpy(dtype=np.float64)
    n_use = max(2, int(len(md_k) * 0.2))
    coef = np.polyfit(md_k[-n_use:], tvt_k[-n_use:], 1)
    return np.polyval(coef, target_md)


def _linear_bridge_md_z(horizontal: pd.DataFrame, target_md: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    """Fit linear (intercept, MD, Z → TVT_input) on all known rows (lstsq)."""
    known = horizontal[horizontal["TVT_input"].notna()]
    if len(known) < 3:
        # fallback: only MD
        return _linear_bridge_md(horizontal, target_md)
    md_k = known["MD"].to_numpy(dtype=np.float64)
    z_k = known["Z"].to_numpy(dtype=np.float64)
    tvt_k = known["TVT_input"].to_numpy(dtype=np.float64)
    X_k = np.column_stack([np.ones(len(known)), md_k, z_k])
    coef, _, _, _ = np.linalg.lstsq(X_k, tvt_k, rcond=None)
    X_h = np.column_stack([np.ones(len(target_md)), target_md, target_z])
    return X_h @ coef


def _last_slope_bridge(horizontal: pd.DataFrame, target_md: np.ndarray) -> np.ndarray:
    """Extend last known TVT_input point using local slope (last 2 known rows)."""
    known = horizontal[horizontal["TVT_input"].notna()]
    if len(known) < 2:
        if len(known) == 1:
            return np.full(len(target_md), float(known["TVT_input"].iloc[0]))
        return np.zeros(len(target_md))
    md_k = known["MD"].to_numpy(dtype=np.float64)
    tvt_k = known["TVT_input"].to_numpy(dtype=np.float64)
    last_md = float(md_k[-1])
    last_tvt = float(tvt_k[-1])
    dmd = float(md_k[-1] - md_k[-2])
    slope = float((tvt_k[-1] - tvt_k[-2]) / max(dmd, 1e-6))
    return last_tvt + slope * (target_md - last_md)


def build_all_targets(
    feat_df: pd.DataFrame,
    horizontal_by_well: dict[str, pd.DataFrame],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Return {target_name: (target_array, base_array)} for all rows in feat_df.
    pred_TVT = model_pred + base
    """
    n = len(feat_df)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    tvt = feat_df["_tvt_true"].to_numpy(dtype=np.float64)
    z = feat_df["Z"].to_numpy(dtype=np.float64)
    md = feat_df["MD"].to_numpy(dtype=np.float64)
    well_ids = feat_df["_well_id"].to_numpy()

    # T0: raw TVT
    results["T0"] = (tvt.copy(), np.zeros(n))

    # T1: TVT + Z (= plane_coord from Exp 0; Z is elevation so TVT+Z ≈ const)
    # Spec calls this T1 = TVT - Z where Z is "depth" (positive convention).
    # In our data Z is elevation (negative), so TVT + Z = TVT - depth = plane_coord.
    # Reconstruction: pred_TVT = pred_T1 - Z_row
    results["T1"] = (tvt + z, -z)

    # T2–T4 computed per well
    base_t2 = np.empty(n)
    base_t3 = np.empty(n)
    base_t4 = np.empty(n)

    for wid, h_df in horizontal_by_well.items():
        mask = well_ids == wid
        if not mask.any():
            continue
        target_md = md[mask]
        target_z = z[mask]
        base_t2[mask] = _linear_bridge_md(h_df, target_md)
        base_t3[mask] = _linear_bridge_md_z(h_df, target_md, target_z)
        base_t4[mask] = _last_slope_bridge(h_df, target_md)

    results["T2"] = (tvt - base_t2, base_t2)
    results["T3"] = (tvt - base_t3, base_t3)
    results["T4"] = (tvt - base_t4, base_t4)

    return results


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _lgb_params(cfg: ShootoutConfig) -> dict[str, Any]:
    return dict(
        objective="regression",
        metric="rmse",
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        feature_fraction=cfg.colsample_bytree,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )


def run_cv(
    feat_df: pd.DataFrame,
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: ShootoutConfig,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]], dict[str, np.ndarray]]:
    """
    GroupKFold cross-validation for all targets.

    Returns
    -------
    oof_preds : {target_name: array of pred_TVT for all rows}
    fold_metrics : {target_name: [rmse_fold0, …]}
    importances : {target_name: feature_importances summed over folds}
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required. Install with: uv sync --extra dev"
        ) from exc

    assert_schema_safe_columns(FEATURE_COLS, context="target_shootout features")

    X = feat_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    well_ids = feat_df["_well_id"].to_numpy()
    tvt_true = feat_df["_tvt_true"].to_numpy(dtype=np.float64)

    # Group-K-Fold by well_id (no sklearn dependency)
    unique_wells = np.unique(well_ids)
    rng = np.random.default_rng(cfg.seed)
    shuffled_wells = rng.permutation(unique_wells)
    fold_assignments = {w: int(i % cfg.n_folds) for i, w in enumerate(shuffled_wells)}
    row_fold = np.array([fold_assignments[w] for w in well_ids], dtype=np.int64)

    splits = [
        (np.where(row_fold != f)[0], np.where(row_fold == f)[0])
        for f in range(cfg.n_folds)
    ]

    oof_preds: dict[str, np.ndarray] = {
        name: np.full(len(feat_df), np.nan) for name in TARGET_NAMES
    }
    fold_metrics: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    importances: dict[str, np.ndarray] = {
        name: np.zeros(len(FEATURE_COLS)) for name in TARGET_NAMES
    }

    lgb_params = _lgb_params(cfg)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(
            f"  [shootout] fold {fold_idx + 1}/{cfg.n_folds} "
            f"train={len(train_idx)} val={len(val_idx)}",
            flush=True,
        )
        X_train, X_val = X[train_idx], X[val_idx]

        for name in TARGET_NAMES:
            target_arr, base_arr = targets[name]
            y_train = target_arr[train_idx]
            base_val = base_arr[val_idx]

            # Replace NaN targets (should not happen, but guard)
            valid_train = np.isfinite(y_train)
            if valid_train.sum() < 10:
                continue

            dtrain = lgb.Dataset(X_train[valid_train], label=y_train[valid_train])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = lgb.train(
                    lgb_params,
                    dtrain,
                    num_boost_round=cfg.n_estimators,
                    valid_sets=[dtrain],
                    callbacks=[lgb.log_evaluation(period=-1)],
                )

            pred_target = booster.predict(X_val)
            pred_tvt = pred_target + base_val

            oof_preds[name][val_idx] = pred_tvt

            valid_val = np.isfinite(tvt_true[val_idx])
            if valid_val.sum() > 0:
                rmse = float(
                    np.sqrt(
                        np.mean((pred_tvt[valid_val] - tvt_true[val_idx][valid_val]) ** 2)
                    )
                )
            else:
                rmse = float("nan")
            fold_metrics[name].append(rmse)

            importances[name] += booster.feature_importance(importance_type="gain")

    return oof_preds, fold_metrics, importances


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _well_rmse(
    well_ids: np.ndarray,
    pred_tvt: np.ndarray,
    tvt_true: np.ndarray,
) -> dict[str, float]:
    """Compute per-well RMSE and aggregate statistics."""
    per_well: list[float] = []
    for wid in np.unique(well_ids):
        mask = (well_ids == wid) & np.isfinite(pred_tvt) & np.isfinite(tvt_true)
        if mask.sum() < 1:
            continue
        rmse = float(np.sqrt(np.mean((pred_tvt[mask] - tvt_true[mask]) ** 2)))
        per_well.append(rmse)
    if not per_well:
        return {
            "mean_well_rmse": float("nan"),
            "p50_well_rmse": float("nan"),
            "p90_well_rmse": float("nan"),
            "p95_well_rmse": float("nan"),
            "worst_well_rmse": float("nan"),
        }
    arr = np.array(per_well)
    return {
        "mean_well_rmse": float(np.mean(arr)),
        "p50_well_rmse": float(np.percentile(arr, 50)),
        "p90_well_rmse": float(np.percentile(arr, 90)),
        "p95_well_rmse": float(np.percentile(arr, 95)),
        "worst_well_rmse": float(np.max(arr)),
    }


def compute_metrics(
    feat_df: pd.DataFrame,
    oof_preds: dict[str, np.ndarray],
    fold_metrics: dict[str, list[float]],
    tail_df: pd.DataFrame | None,
    cfg: ShootoutConfig,
) -> dict[str, Any]:
    """Compute all metrics and GO verdict."""
    tvt_true = feat_df["_tvt_true"].to_numpy(dtype=np.float64)
    well_ids = feat_df["_well_id"].to_numpy()

    metrics: dict[str, Any] = {"config": asdict(cfg)}
    target_metrics: dict[str, Any] = {}

    for name in TARGET_NAMES:
        pred = oof_preds[name]
        valid = np.isfinite(pred) & np.isfinite(tvt_true)
        row_rmse = (
            float(np.sqrt(np.mean((pred[valid] - tvt_true[valid]) ** 2)))
            if valid.sum() > 0
            else float("nan")
        )
        well_stats = _well_rmse(well_ids, pred, tvt_true)
        fold_rmses = fold_metrics[name]
        tm: dict[str, Any] = {
            "row_rmse": row_rmse,
            **well_stats,
            "fold_rmses": fold_rmses,
            "fold_mean": float(np.mean(fold_rmses)) if fold_rmses else float("nan"),
            "fold_std": float(np.std(fold_rmses)) if fold_rmses else float("nan"),
        }

        # Tail-class breakdown
        if tail_df is not None:
            tail_map = dict(zip(tail_df["well_id"], tail_df["tail_class"]))
            feat_tail = np.array([tail_map.get(w, "unknown") for w in well_ids])
            tail_rmses: dict[str, float] = {}
            for tc in np.unique(feat_tail):
                mask = (feat_tail == tc) & np.isfinite(pred) & np.isfinite(tvt_true)
                if mask.sum() < 5:
                    continue
                tail_rmses[str(tc)] = float(
                    np.sqrt(np.mean((pred[mask] - tvt_true[mask]) ** 2))
                )
            tm["tail_class_rmse"] = tail_rmses

        target_metrics[name] = tm

    metrics["targets"] = target_metrics

    # GO verdict
    t0_mean = target_metrics["T0"]["mean_well_rmse"]
    best_name = "T0"
    best_improvement = 0.0
    for name in ("T1", "T2", "T3", "T4"):
        imp = t0_mean - target_metrics[name]["mean_well_rmse"]
        if imp > best_improvement:
            best_improvement = imp
            best_name = name

    go_by_rmse = best_improvement >= cfg.go_min_improvement_ft

    # Tail improvement
    go_by_tail = False
    if tail_df is not None and best_name != "T0":
        t0_tail = target_metrics["T0"].get("tail_class_rmse", {})
        best_tail = target_metrics[best_name].get("tail_class_rmse", {})
        common = set(t0_tail) & set(best_tail)
        if common:
            n_improved = sum(
                1 for tc in common if best_tail[tc] < t0_tail[tc]
            )
            go_by_tail = n_improved > len(common) // 2
    else:
        go_by_tail = None  # type: ignore[assignment]

    verdict = "GO" if go_by_rmse else "NO-GO"

    metrics["verdict"] = verdict
    metrics["go_by_rmse"] = go_by_rmse
    metrics["go_by_tail"] = go_by_tail
    metrics["best_target"] = best_name
    metrics["best_improvement_ft"] = best_improvement
    metrics["go_threshold_ft"] = cfg.go_min_improvement_ft
    metrics["t0_mean_well_rmse"] = t0_mean
    metrics["best_mean_well_rmse"] = target_metrics[best_name]["mean_well_rmse"]

    return metrics


# ---------------------------------------------------------------------------
# Output: figures
# ---------------------------------------------------------------------------

def _write_figures(
    output_dir: Path,
    feat_df: pd.DataFrame,
    oof_preds: dict[str, np.ndarray],
    metrics: dict[str, Any],
    importances: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tvt_true = feat_df["_tvt_true"].to_numpy(dtype=np.float64)
    target_metrics = metrics["targets"]

    # -- 1. Mean well RMSE bar chart ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(TARGET_NAMES)
    vals = [target_metrics[n]["mean_well_rmse"] for n in names]
    colors = ["#c0392b" if n == "T0" else "#2980b9" for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Mean well RMSE (ft)")
    ax.set_title("Target Shootout — Mean Well OOF RMSE")
    ax.axhline(
        target_metrics["T0"]["mean_well_rmse"],
        color="#c0392b",
        linestyle="--",
        linewidth=0.8,
        label="T0 baseline",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "mean_well_rmse.png", dpi=150)
    plt.close(fig)

    # -- 2. Row RMSE bar chart -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    vals_row = [target_metrics[n]["row_rmse"] for n in names]
    bars = ax.bar(names, vals_row, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals_row):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Row RMSE (ft)")
    ax.set_title("Target Shootout — Row-level OOF RMSE")
    fig.tight_layout()
    fig.savefig(figures_dir / "row_rmse.png", dpi=150)
    plt.close(fig)

    # -- 3. Per-well RMSE distribution per target (violin) --------------------
    well_ids = feat_df["_well_id"].to_numpy()
    per_well_data: dict[str, list[float]] = {}
    for name in TARGET_NAMES:
        pred = oof_preds[name]
        pw: list[float] = []
        for wid in np.unique(well_ids):
            mask = (well_ids == wid) & np.isfinite(pred) & np.isfinite(tvt_true)
            if mask.sum() < 1:
                continue
            pw.append(
                float(np.sqrt(np.mean((pred[mask] - tvt_true[mask]) ** 2)))
            )
        per_well_data[name] = pw

    fig, ax = plt.subplots(figsize=(10, 5))
    data_list = [per_well_data[n] for n in TARGET_NAMES]
    parts = ax.violinplot(data_list, showmedians=True)
    ax.set_xticks(range(1, len(TARGET_NAMES) + 1))
    ax.set_xticklabels(TARGET_NAMES)
    ax.set_ylabel("Per-well OOF RMSE (ft)")
    ax.set_title("Target Shootout — Per-well RMSE Distribution")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "per_well_rmse_violin.png", dpi=150)
    plt.close(fig)

    # -- 4. Fold RMSE per target ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in TARGET_NAMES:
        fold_rmses = target_metrics[name]["fold_rmses"]
        ax.plot(
            range(1, len(fold_rmses) + 1),
            fold_rmses,
            marker="o",
            label=name,
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (ft)")
    ax.set_title("Target Shootout — Fold RMSE by Target")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fold_rmse.png", dpi=150)
    plt.close(fig)

    # -- 5. Feature importance for T0 and best target -------------------------
    best_name = metrics["best_target"]
    targets_to_plot = list(dict.fromkeys(["T0", best_name]))
    fig, axes = plt.subplots(1, len(targets_to_plot), figsize=(7 * len(targets_to_plot), 6))
    if len(targets_to_plot) == 1:
        axes = [axes]
    for ax, name in zip(axes, targets_to_plot):
        imp = importances[name]
        order = np.argsort(imp)[::-1][:20]
        ax.barh(
            [FEATURE_COLS[i] for i in order][::-1],
            imp[order][::-1],
            color="#2980b9",
        )
        ax.set_xlabel("Importance (sum over folds)")
        ax.set_title(f"Feature Importance — {name}")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # -- 6. Tail-class RMSE (if available) ------------------------------------
    if "tail_class_rmse" in target_metrics["T0"]:
        tail_classes = sorted(target_metrics["T0"]["tail_class_rmse"].keys())
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.8 / len(TARGET_NAMES)
        for i, name in enumerate(TARGET_NAMES):
            tc_rmses = target_metrics[name].get("tail_class_rmse", {})
            x = np.arange(len(tail_classes))
            vals_tc = [tc_rmses.get(tc, float("nan")) for tc in tail_classes]
            ax.bar(
                x + i * bar_width,
                vals_tc,
                width=bar_width,
                label=name,
            )
        ax.set_xticks(np.arange(len(tail_classes)) + bar_width * (len(TARGET_NAMES) - 1) / 2)
        ax.set_xticklabels(tail_classes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("OOF RMSE (ft)")
        ax.set_title("Target Shootout — Tail-class RMSE")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "tail_class_rmse.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _write_report(output_dir: Path, metrics: dict[str, Any], cfg: ShootoutConfig) -> None:
    target_metrics = metrics["targets"]
    lines = [
        "# TARGET_SHOOTOUT_V0",
        "",
        "## Question",
        "Which target formulation (T0–T4) leads to the best OOF TVT RMSE using LightGBM?",
        "",
        "## Config",
        f"```json\n{json.dumps(_json_clean(asdict(cfg)), indent=2)}\n```",
        "",
        f"## VERDICT: {metrics['verdict']}",
        "",
        "Best target: **{}** (mean_well_rmse improvement over T0: **{:.3f} ft**, threshold: {:.2f} ft)".format(
            metrics["best_target"],
            metrics["best_improvement_ft"],
            metrics["go_threshold_ft"],
        ),
        "",
        "## Results Table",
        "",
        "| Target | Description | row_rmse | mean_well_rmse | p50 | p90 | p95 | worst |",
        "|--------|-------------|----------|----------------|-----|-----|-----|-------|",
    ]
    for name in TARGET_NAMES:
        tm = target_metrics[name]
        desc = TARGET_DESCRIPTIONS[name]

        def fmt(v: Any) -> str:
            return f"{v:.3f}" if v is not None and np.isfinite(float(v)) else "—"

        lines.append(
            f"| **{name}** | {desc} "
            f"| {fmt(tm['row_rmse'])} "
            f"| {fmt(tm['mean_well_rmse'])} "
            f"| {fmt(tm['p50_well_rmse'])} "
            f"| {fmt(tm['p90_well_rmse'])} "
            f"| {fmt(tm['p95_well_rmse'])} "
            f"| {fmt(tm['worst_well_rmse'])} |"
        )

    lines += [
        "",
        "## Fold RMSE",
        "",
        "| Target | " + " | ".join(f"Fold {i+1}" for i in range(cfg.n_folds)) + " | Mean |",
        "|--------|" + " ---- |" * (cfg.n_folds + 1),
    ]
    for name in TARGET_NAMES:
        tm = target_metrics[name]
        fold_vals = " | ".join(
            f"{v:.3f}" if np.isfinite(v) else "—" for v in tm["fold_rmses"]
        )
        lines.append(
            f"| {name} | {fold_vals} | {tm['fold_mean']:.3f} |"
        )

    if "tail_class_rmse" in target_metrics["T0"]:
        tail_classes = sorted(target_metrics["T0"]["tail_class_rmse"].keys())
        lines += [
            "",
            "## Tail-class RMSE",
            "",
            "| Tail class | " + " | ".join(TARGET_NAMES) + " |",
            "|------------|" + " --- |" * len(TARGET_NAMES),
        ]
        for tc in tail_classes:
            row = f"| {tc} |"
            for name in TARGET_NAMES:
                v = target_metrics[name].get("tail_class_rmse", {}).get(tc)
                row += f" {v:.3f} |" if v is not None and np.isfinite(v) else " — |"
            lines.append(row)

    lines += [
        "",
        "## Interpretation",
        "- GO → build DL/segmentation in the winning coordinate.",
        "- If T1 wins → use plane_coord (TVT+Z) as the target.",
        "- If T2/T3/T4 wins → use bridge-residual inpainting / masked residual path.",
        "- NO-GO → target transform is not the lever; investigate feature set or architecture.",
        "",
        "## Figures",
        "- [Mean well RMSE](figures/mean_well_rmse.png)",
        "- [Row RMSE](figures/row_rmse.png)",
        "- [Per-well violin](figures/per_well_rmse_violin.png)",
        "- [Fold RMSE](figures/fold_rmse.png)",
        "- [Feature importance](figures/feature_importance.png)",
        "- [Tail-class RMSE](figures/tail_class_rmse.png)",
    ]

    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_target_shootout(
    data_dir: Path,
    output_dir: Path,
    cfg: ShootoutConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tail classes (optional)
    tail_df: pd.DataFrame | None = None
    if cfg.tail_classes_path is not None and Path(cfg.tail_classes_path).exists():
        tail_df = pd.read_csv(cfg.tail_classes_path)
        print(
            f"[shootout] Loaded {len(tail_df)} tail labels from {cfg.tail_classes_path}",
            flush=True,
        )

    # Discover wells
    data_cfg = DataConfig(data_dir=data_dir, k_wells=cfg.k_wells)
    wells = discover_wells(data_cfg)
    print(f"[shootout] Processing {len(wells)} wells ...", flush=True)

    # Build features and collect horizontal DataFrames
    all_feat: list[pd.DataFrame] = []
    horizontal_by_well: dict[str, pd.DataFrame] = {}

    for i, wp in enumerate(wells, 1):
        if i % 100 == 0 or i == 1 or i == len(wells):
            print(f"[shootout] features {i}/{len(wells)}: {wp.well_id}", flush=True)
        horizontal, typewell = load_well(wp)
        feat = build_well_features(wp.well_id, horizontal, typewell)
        if len(feat) == 0:
            continue
        all_feat.append(feat)
        horizontal_by_well[wp.well_id] = horizontal

    if not all_feat:
        raise RuntimeError("No hidden rows found across all wells.")

    feat_df = pd.concat(all_feat, ignore_index=True)
    print(
        f"[shootout] Total hidden rows: {len(feat_df):,} "
        f"across {feat_df['_well_id'].nunique()} wells",
        flush=True,
    )

    # Build targets
    print("[shootout] Building targets ...", flush=True)
    targets = build_all_targets(feat_df, horizontal_by_well)

    # Cross-validation
    print("[shootout] Running 5-fold CV (LightGBM) ...", flush=True)
    oof_preds, fold_metrics, importances = run_cv(feat_df, targets, cfg)

    # Save OOF predictions
    oof_df = feat_df[["_well_id", "MD", "_tvt_true"]].copy()
    oof_df = oof_df.rename(columns={"_well_id": "well_id", "_tvt_true": "tvt_true"})
    for name in TARGET_NAMES:
        oof_df[f"pred_{name}"] = oof_preds[name]
    oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)

    # Metrics
    print("[shootout] Computing metrics ...", flush=True)
    metrics = compute_metrics(feat_df, oof_preds, fold_metrics, tail_df, cfg)

    print("\n=== TARGET SHOOTOUT RESULTS ===")
    print(json.dumps(_json_clean(metrics), indent=2))
    print(f"\nVERDICT: {metrics['verdict']}")
    print(
        f"Best target: {metrics['best_target']} "
        f"(improvement: {metrics['best_improvement_ft']:.3f} ft over T0)"
    )

    # Serialise metrics
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )

    # Figures
    print("[shootout] Writing figures ...", flush=True)
    _write_figures(output_dir, feat_df, oof_preds, metrics, importances)

    # Report
    _write_report(output_dir, metrics, cfg)
    print(f"[shootout] Reports saved to: {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 1: Target Formulation Shootout"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/target_shootout_v0"),
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--k-wells", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--tail-classes-path",
        type=Path,
        default=None,
        help="Optional path to well tail audit CSV (well_id, tail_class).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_target_shootout(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cfg=ShootoutConfig(
            n_folds=args.n_folds,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            k_wells=args.k_wells,
            seed=args.seed,
            n_jobs=args.n_jobs,
            tail_classes_path=args.tail_classes_path,
        ),
    )
