from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .runlog import RunLogger


class ResidualModel:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ResidualModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class WrappedRegressor(ResidualModel):
    estimator: Any

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WrappedRegressor":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)


class ResidualStackRegressor(ResidualModel):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.base_models: list[ResidualModel] = []
        self.base_names: list[str] = []
        self.weights: np.ndarray | None = None
        self.blender: Ridge | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ResidualStackRegressor":
        base_specs = self.config.get("base_models") or []
        if not base_specs:
            raise ValueError("model.name=stack requires model.base_models.")

        base_predictions = []
        self.base_models = []
        self.base_names = []
        for index, spec in enumerate(base_specs):
            name = str(spec.get("name") or f"model_{index}")
            model = make_single_model(spec, self.seed + index + 1)
            model.fit(X, y)
            pred = model.predict(X)
            self.base_models.append(model)
            self.base_names.append(name)
            base_predictions.append(pred)

        train_stack = np.vstack(base_predictions).T
        blend_cfg = self.config.get("blend", {})
        method = str(blend_cfg.get("method", "weighted")).lower()
        if method == "ridge":
            alpha = float(blend_cfg.get("alpha", 1.0))
            self.blender = Ridge(alpha=alpha, fit_intercept=True)
            self.blender.fit(train_stack, y)
            self.weights = None
        elif method in {"hill_climb", "hill-climb", "hillclimb"}:
            self.weights = fit_hill_climb_weights(
                train_stack,
                y,
                iterations=int(blend_cfg.get("iterations", 200)),
                alpha_grid=blend_cfg.get("alpha_grid"),
            )
            self.blender = None
        elif method == "weighted":
            weights = blend_cfg.get("weights")
            if weights is None:
                weights_array = np.ones(len(self.base_models), dtype=float)
            else:
                weights_array = np.asarray(weights, dtype=float)
            if len(weights_array) != len(self.base_models):
                raise ValueError(
                    "model.blend.weights length must match model.base_models length."
                )
            total = float(np.sum(weights_array))
            if abs(total) <= 1e-12:
                raise ValueError("model.blend.weights must not sum to zero.")
            self.weights = weights_array / total
            self.blender = None
        else:
            raise ValueError(
                "model.blend.method must be 'weighted', 'ridge', or 'hill_climb'."
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.base_models:
            raise RuntimeError("ResidualStackRegressor is not fitted.")
        stack = np.vstack([model.predict(X) for model in self.base_models]).T
        if self.blender is not None:
            return np.asarray(self.blender.predict(stack), dtype=float)
        if self.weights is None:
            raise RuntimeError("ResidualStackRegressor has no fitted blend weights.")
        return stack @ self.weights


def fit_hill_climb_weights(
    stack: np.ndarray,
    y: np.ndarray,
    iterations: int,
    alpha_grid: list[float] | None,
) -> np.ndarray:
    if stack.ndim != 2 or stack.shape[1] == 0:
        raise ValueError("Hill-climb blend requires a non-empty prediction stack.")

    y = np.asarray(y, dtype=float)
    n_models = stack.shape[1]
    single_scores = [rmse(stack[:, idx], y) for idx in range(n_models)]
    best_model = int(np.argmin(single_scores))
    weights = np.zeros(n_models, dtype=float)
    weights[best_model] = 1.0
    best_pred = stack[:, best_model].copy()
    best_score = single_scores[best_model]

    grid = alpha_grid or [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    grid = [float(alpha) for alpha in grid if 0.0 < float(alpha) < 1.0]
    if not grid:
        return weights

    for _ in range(max(int(iterations), 0)):
        candidate_score = best_score
        candidate_pred = best_pred
        candidate_weights = weights
        for model_idx in range(n_models):
            model_pred = stack[:, model_idx]
            for alpha in grid:
                pred = (1.0 - alpha) * best_pred + alpha * model_pred
                score = rmse(pred, y)
                if score + 1e-12 < candidate_score:
                    blended_weights = (1.0 - alpha) * weights.copy()
                    blended_weights[model_idx] += alpha
                    candidate_score = score
                    candidate_pred = pred
                    candidate_weights = blended_weights
        if candidate_score + 1e-12 >= best_score:
            break
        best_score = candidate_score
        best_pred = candidate_pred
        weights = candidate_weights

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Hill-climb blend produced invalid weights.")
    return weights / total


def make_lightgbm(params: dict[str, Any], seed: int) -> WrappedRegressor:
    from lightgbm import LGBMRegressor

    defaults = {
        "objective": "regression",
        "n_estimators": 700,
        "learning_rate": 0.035,
        "num_leaves": 96,
        "max_depth": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "min_child_samples": 50,
        "n_jobs": -1,
        "verbosity": -1,
    }
    defaults.update(params)
    defaults["random_state"] = seed
    return WrappedRegressor(LGBMRegressor(**defaults))


def make_xgboost(params: dict[str, Any], seed: int) -> WrappedRegressor:
    from xgboost import XGBRegressor

    defaults = {
        "objective": "reg:squarederror",
        "n_estimators": 650,
        "learning_rate": 0.035,
        "max_depth": 6,
        "min_child_weight": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }
    defaults.update(params)
    defaults["random_state"] = seed
    return WrappedRegressor(XGBRegressor(**defaults))


def make_catboost(params: dict[str, Any], seed: int) -> WrappedRegressor:
    from catboost import CatBoostRegressor

    defaults = {
        "loss_function": "RMSE",
        "iterations": 900,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.9,
        "allow_writing_files": False,
        "verbose": False,
    }
    defaults.update(params)
    defaults["random_seed"] = seed
    return WrappedRegressor(CatBoostRegressor(**defaults))


def make_hist_gradient_boosting(params: dict[str, Any], seed: int) -> WrappedRegressor:
    params = dict(params)
    params["random_state"] = seed
    return WrappedRegressor(HistGradientBoostingRegressor(**params))


def make_single_model(config: dict[str, Any], seed: int) -> ResidualModel:
    name = str(config.get("name", "lightgbm")).lower()
    params = dict(config.get("params") or {})
    if name in {"lightgbm", "lgbm", "lgb"}:
        return make_lightgbm(params, seed)
    if name in {"xgboost", "xgb"}:
        return make_xgboost(params, seed)
    if name in {"catboost", "cat"}:
        return make_catboost(params, seed)
    if name in {"hist_gradient_boosting", "hgb"}:
        return make_hist_gradient_boosting(params, seed)
    raise ValueError(f"Unsupported model.name={name!r}.")


def make_model(config: dict[str, Any], seed: int) -> ResidualModel:
    model_cfg = config["model"]
    name = str(model_cfg.get("name", "lightgbm")).lower()
    if name == "stack":
        return ResidualStackRegressor(model_cfg, seed)
    return make_single_model(model_cfg, seed)


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
