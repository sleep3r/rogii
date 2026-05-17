from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .runlog import RunLogger


def make_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    if config["model"].get("name") != "hist_gradient_boosting":
        raise ValueError(
            "Only model.name=hist_gradient_boosting is currently supported."
        )
    params = dict(config["model"].get("params", {}))
    params["random_state"] = seed
    return HistGradientBoostingRegressor(**params)


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def shuffled_group_folds(
    groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_splits = max(2, min(n_splits, len(unique_groups)))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        valid_groups = set(unique_groups[fold::n_splits])
        valid_idx = np.array([group in valid_groups for group in groups])
        train_idx = ~valid_idx
        folds.append((train_idx, valid_idx))
    return folds


def apply_postprocess(
    flat: np.ndarray,
    residual: np.ndarray,
    config: dict[str, Any],
    residual_weight: float | None = None,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    clip_value = config["postprocess"].get("residual_clip")
    if clip_value not in (None, ""):
        clip = float(clip_value)
        residual = np.clip(residual, -clip, clip)
    if residual_weight is None:
        configured = config["postprocess"].get("residual_weight", 1.0)
        residual_weight = 1.0 if configured == "auto" else float(configured)
    weight = float(residual_weight)
    return np.asarray(flat, dtype=float) + weight * residual


def tune_residual_weight(
    flat: np.ndarray,
    residual_pred: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, list[dict[str, float]]]:
    configured = config["postprocess"].get("residual_weight", 1.0)
    if configured != "auto":
        weight = float(configured)
        pred = apply_postprocess(flat, residual_pred, config, residual_weight=weight)
        return weight, [{"weight": weight, "rmse": rmse(pred, y_true)}]

    grid = config["postprocess"].get("residual_weight_grid") or [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    scores: list[dict[str, float]] = []
    for weight in grid:
        weight = float(weight)
        pred = apply_postprocess(flat, residual_pred, config, residual_weight=weight)
        scores.append({"weight": weight, "rmse": rmse(pred, y_true)})
    best = min(scores, key=lambda item: item["rmse"])
    return float(best["weight"]), scores


def run_cv(
    X: pd.DataFrame,
    residual: np.ndarray,
    groups: np.ndarray,
    flat: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    seed: int,
    logger: RunLogger | None = None,
) -> dict[str, Any]:
    validation = config["validation"]
    unique_wells = np.array(sorted(set(groups)))
    max_wells = validation.get("max_wells")
    if max_wells not in (None, "") and len(unique_wells) > int(max_wells):
        rng = np.random.default_rng(seed)
        selected = set(rng.choice(unique_wells, size=int(max_wells), replace=False))
        cv_mask = np.array([group in selected for group in groups])
    else:
        cv_mask = np.ones(len(groups), dtype=bool)

    X_cv = X.loc[cv_mask].reset_index(drop=True)
    residual_cv = residual[cv_mask]
    groups_cv = groups[cv_mask]
    flat_cv = flat[cv_mask]
    y_true_cv = y_true[cv_mask]

    unique_cv_groups = sorted(set(groups_cv))
    if len(unique_cv_groups) < 2:
        if logger is not None:
            logger.warn("Skipping CV", reason="need at least two wells")
        return {"enabled": False, "reason": "not enough groups"}

    folds = shuffled_group_folds(groups_cv, int(validation.get("n_splits", 5)), seed)
    oof_residual = np.zeros(len(X_cv), dtype=float)
    fold_ids = np.zeros(len(X_cv), dtype=int)
    fold_metrics: list[dict[str, Any]] = []

    if logger is not None:
        logger.info(
            "Prepared CV", rows=len(X_cv), wells=len(unique_cv_groups), folds=len(folds)
        )

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        if logger is not None:
            fold_context = logger.step(
                f"CV fold {fold_id}/{len(folds)}",
                train_rows=int(train_idx.sum()),
                valid_rows=int(valid_idx.sum()),
            )
        else:
            fold_context = nullcontext()
        with fold_context:
            model = make_model(config, seed + fold_id)
            model.fit(X_cv.loc[train_idx], residual_cv[train_idx])
            residual_pred = model.predict(X_cv.loc[valid_idx])
        oof_residual[valid_idx] = residual_pred
        fold_ids[valid_idx] = fold_id

    if logger is not None:
        logger.info(
            "Tuning residual blend",
            candidates=len(config["postprocess"].get("residual_weight_grid") or []),
        )
    best_weight, weight_scores = tune_residual_weight(
        flat_cv, oof_residual, y_true_cv, config
    )
    oof_pred = apply_postprocess(
        flat_cv, oof_residual, config, residual_weight=best_weight
    )
    for fold_id in range(1, len(folds) + 1):
        valid_idx = fold_ids == fold_id
        fold_rmse = rmse(oof_pred[valid_idx], y_true_cv[valid_idx])
        fold_metrics.append(
            {
                "fold": fold_id,
                "rmse": fold_rmse,
                "valid_rows": int(valid_idx.sum()),
                "valid_wells": int(len(set(groups_cv[valid_idx]))),
            }
        )
        if logger is not None:
            logger.metric(
                "CV fold RMSE", fold=fold_id, weight=f"{best_weight:g}", rmse=fold_rmse
            )

    overall_rmse = rmse(oof_pred, y_true_cv)
    flat_rmse = rmse(flat_cv, y_true_cv)
    if logger is not None:
        logger.metric(
            "CV summary",
            rmse=overall_rmse,
            flat_rmse=flat_rmse,
            best_residual_weight=f"{best_weight:g}",
        )
    return {
        "enabled": True,
        "rmse": overall_rmse,
        "flat_rmse": flat_rmse,
        "best_residual_weight": best_weight,
        "residual_weight_scores": weight_scores,
        "rows": int(len(X_cv)),
        "wells": int(len(unique_cv_groups)),
        "folds": fold_metrics,
    }
