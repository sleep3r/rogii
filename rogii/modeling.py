from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .runlog import RunLogger


class ResidualModel:
    def fit(self, X: pd.DataFrame, y: np.ndarray, **_: Any) -> "ResidualModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class WrappedRegressor(ResidualModel):
    estimator: Any
    fit_options: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        **_: Any,
    ) -> "WrappedRegressor":
        options = dict(self.fit_options)
        kind = str(options.pop("kind", "")).lower()
        early_stopping_rounds = options.pop("early_stopping_rounds", None)

        if X_valid is None or y_valid is None:
            self.estimator.fit(X, y)
            return self

        if kind == "lightgbm":
            import lightgbm as lgb

            callbacks = list(options.pop("callbacks", []))
            if early_stopping_rounds not in (None, ""):
                callbacks.append(
                    lgb.early_stopping(int(early_stopping_rounds), verbose=False)
                )
            self.estimator.fit(
                X,
                y,
                eval_set=[(X_valid, y_valid)],
                callbacks=callbacks,
                **options,
            )
        elif kind == "xgboost":
            self.estimator.fit(
                X,
                y,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                **options,
            )
        elif kind == "catboost":
            self.estimator.fit(
                X,
                y,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                **options,
            )
        else:
            self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)


class EnsembleRegressor(ResidualModel):
    """Grouped OOF ensemble used by both local training and Kaggle inference."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.model_config = config["model"]
        self.seed = seed
        self.base_names: list[str] = []
        self.fold_models: list[list[ResidualModel]] = []
        self.weights: np.ndarray | None = None
        self.oof_residual_: np.ndarray | None = None
        self.oof_stack_: np.ndarray | None = None
        self.metrics_: dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        logger: RunLogger | None = None,
        **_: Any,
    ) -> "EnsembleRegressor":
        if groups is None:
            raise ValueError("Grouped well ids are required for ensemble training.")

        base_specs = self.model_config.get("base_models") or []
        if not base_specs:
            raise ValueError("model.base_models must contain at least one model.")

        n_splits = int(self.config["validation"].get("n_splits", 5))
        folds = shuffled_group_folds(groups, n_splits, self.seed)
        y = np.asarray(y, dtype=float)
        oof_stack = np.zeros((len(X), len(base_specs)), dtype=float)
        base_metrics: list[dict[str, Any]] = []
        self.fold_models = []
        self.base_names = []

        for model_idx, spec in enumerate(base_specs):
            name = model_spec_name(spec, model_idx)
            self.base_names.append(name)
            fold_models: list[ResidualModel] = []
            fold_scores = []
            if logger is not None:
                logger.info(
                    "OOF base model",
                    model=name,
                    index=model_idx + 1,
                    total=len(base_specs),
                    folds=len(folds),
                )

            for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
                model = make_single_model(
                    spec,
                    self.seed + 1000 * (model_idx + 1) + fold_id,
                )
                context = (
                    logger.step(
                        "OOF model fold",
                        model=name,
                        fold=fold_id,
                        train_rows=int(train_idx.sum()),
                        valid_rows=int(valid_idx.sum()),
                    )
                    if logger is not None
                    else nullcontext()
                )
                with context:
                    model.fit(
                        X.loc[train_idx],
                        y[train_idx],
                        X_valid=X.loc[valid_idx],
                        y_valid=y[valid_idx],
                    )
                    pred = model.predict(X.loc[valid_idx])
                oof_stack[valid_idx, model_idx] = pred
                fold_scores.append(rmse(pred, y[valid_idx]))
                fold_models.append(model)

            self.fold_models.append(fold_models)
            model_oof_rmse = rmse(oof_stack[:, model_idx], y)
            base_metrics.append(
                {
                    "name": name,
                    "oof_rmse": model_oof_rmse,
                    "fold_rmse": [float(score) for score in fold_scores],
                }
            )
            if logger is not None:
                logger.metric("OOF base RMSE", model=name, rmse=model_oof_rmse)

        self.oof_stack_ = oof_stack
        blend_cfg = self.model_config.get("blend") or {}
        self.weights = fit_hill_climb_weights(
            oof_stack,
            y,
            iterations=int(blend_cfg.get("iterations", 1000)),
            alpha_grid=blend_cfg.get("alpha_grid"),
        )
        self.oof_residual_ = oof_stack @ self.weights

        for item, weight in zip(base_metrics, self.weights, strict=True):
            item["weight"] = float(weight)
        self.metrics_ = {
            "type": "ensemble",
            "n_splits": len(folds),
            "base_models": base_metrics,
            "blend_method": "hill_climb",
            "blend_weights": dict(
                zip(self.base_names, [float(w) for w in self.weights], strict=True)
            ),
            "oof_rmse": rmse(self.oof_residual_, y),
        }
        if logger is not None:
            logger.metric(
                "OOF ensemble RMSE",
                rmse=self.metrics_["oof_rmse"],
                nonzero_weights=int(np.sum(self.weights > 1e-9)),
            )
        return self

    def predict_base_stack(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fold_models:
            raise RuntimeError("EnsembleRegressor is not fitted.")
        columns = []
        for fold_models in self.fold_models:
            fold_pred = np.vstack([model.predict(X) for model in fold_models])
            columns.append(np.mean(fold_pred, axis=0))
        return np.vstack(columns).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("EnsembleRegressor has no blend weights.")
        return self.predict_base_stack(X) @ self.weights


def model_spec_name(spec: dict[str, Any], index: int) -> str:
    explicit = spec.get("id") or spec.get("label")
    if explicit not in (None, ""):
        return str(explicit)
    name = str(spec.get("name", "model")).lower()
    return f"{name}_{index + 1}"


def fit_hill_climb_weights(
    stack: np.ndarray,
    y: np.ndarray,
    iterations: int = 1000,
    alpha_grid: list[float] | None = None,
) -> np.ndarray:
    if stack.ndim != 2 or stack.shape[1] == 0:
        raise ValueError("Hill-climb blend requires a non-empty prediction stack.")

    y = np.asarray(y, dtype=float)
    n_models = stack.shape[1]
    single_scores = [rmse(stack[:, idx], y) for idx in range(n_models)]
    best_model = int(np.argmin(single_scores))
    weights = np.zeros(n_models, dtype=float)
    weights[best_model] = 1.0
    best_score = single_scores[best_model]
    steps = alpha_grid or [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    steps = [float(step) for step in steps if float(step) > 0.0]
    moves = 0

    for step in steps:
        improved = True
        while improved and moves < max(int(iterations), 1):
            improved = False
            candidate_score = best_score
            candidate_weights = weights
            for model_idx in range(n_models):
                new_weights = weights.copy()
                new_weights[model_idx] += step
                new_weights /= float(new_weights.sum())
                score = rmse(stack @ new_weights, y)
                if score + 1e-12 < candidate_score:
                    candidate_score = score
                    candidate_weights = new_weights
                    improved = True
            if improved:
                weights = candidate_weights
                best_score = candidate_score
                moves += 1

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Hill-climb blend produced invalid weights.")
    return weights / total


def make_lightgbm(params: dict[str, Any], seed: int) -> WrappedRegressor:
    from lightgbm import LGBMRegressor

    params = dict(params)
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
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
    return WrappedRegressor(
        LGBMRegressor(**defaults),
        {"kind": "lightgbm", "early_stopping_rounds": early_stopping_rounds},
    )


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
    return WrappedRegressor(XGBRegressor(**defaults), {"kind": "xgboost"})


def make_catboost(params: dict[str, Any], seed: int) -> WrappedRegressor:
    from catboost import CatBoostRegressor

    params = dict(params)
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
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
    if early_stopping_rounds not in (None, ""):
        defaults["od_type"] = "Iter"
        defaults["od_wait"] = int(early_stopping_rounds)
    defaults.update(params)
    defaults["random_seed"] = seed
    return WrappedRegressor(CatBoostRegressor(**defaults), {"kind": "catboost"})


def make_single_model(config: dict[str, Any], seed: int) -> ResidualModel:
    name = str(config.get("name", "lightgbm")).lower()
    params = dict(config.get("params") or {})
    if name in {"lightgbm", "lgbm", "lgb"}:
        return make_lightgbm(params, seed)
    if name in {"xgboost", "xgb"}:
        return make_xgboost(params, seed)
    if name in {"catboost", "cat"}:
        return make_catboost(params, seed)
    raise ValueError(f"Unsupported base model: {name!r}.")


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def shuffled_group_folds(
    groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.array(sorted(set(groups)))
    if len(unique_groups) < 2:
        raise ValueError("Grouped OOF training needs at least two wells.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_splits = min(max(2, n_splits), len(unique_groups))
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
    features: pd.DataFrame | None = None,
    groups: np.ndarray | None = None,
    notebook_blend: dict[str, float] | None = None,
    smoothing: dict[str, float] | None = None,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    clip_value = config["postprocess"].get("residual_clip")
    if clip_value not in (None, ""):
        clip = float(clip_value)
        residual = np.clip(residual, -clip, clip)

    if residual_weight is None:
        residual_weight = float(config["postprocess"].get("residual_weight", 1.0))
    pred = np.asarray(flat, dtype=float) + float(residual_weight) * residual
    pred = apply_notebook_blend(pred, config, features, notebook_blend)
    return apply_smoothing(pred, groups, smoothing)


def notebook_blend_candidates(config: dict[str, Any]) -> list[dict[str, float] | None]:
    blend_cfg = config["postprocess"].get("notebook_blend") or {}
    if not blend_cfg.get("enabled", False):
        return [None]

    candidates = blend_cfg.get("candidates") or []
    if candidates:
        parsed: list[dict[str, float] | None] = []
        for candidate in candidates:
            if candidate is None or candidate.get("enabled", True) is False:
                parsed.append(None)
                continue
            parsed.append(
                {
                    "alpha": float(candidate.get("alpha", 1.0)),
                    "tau": float(candidate.get("tau", 0.0)),
                    "w_pf": float(candidate.get("w_pf", 0.0)),
                }
            )
        return parsed

    alpha_grid = blend_cfg.get("alpha_grid") or [blend_cfg.get("alpha", 1.0)]
    tau_grid = blend_cfg.get("tau_grid") or [blend_cfg.get("tau", 0.0)]
    w_pf_grid = blend_cfg.get("w_pf_grid") or [blend_cfg.get("w_pf", 0.0)]
    return [
        {"alpha": float(alpha), "tau": float(tau), "w_pf": float(w_pf)}
        for alpha, tau, w_pf in product(alpha_grid, tau_grid, w_pf_grid)
    ]


def smoothing_candidates(config: dict[str, Any]) -> list[dict[str, float] | None]:
    smoothing_cfg = config["postprocess"].get("smoothing") or {}
    if not smoothing_cfg.get("enabled", False):
        return [None]
    candidates = smoothing_cfg.get("candidates") or [
        {"enabled": False},
        {"window": 17, "polyorder": 3},
    ]
    parsed: list[dict[str, float] | None] = []
    for candidate in candidates:
        if candidate is None or candidate.get("enabled", True) is False:
            parsed.append(None)
        else:
            parsed.append(
                {
                    "window": int(candidate.get("window", 17)),
                    "polyorder": int(candidate.get("polyorder", 3)),
                }
            )
    return parsed


def apply_notebook_blend(
    pred: np.ndarray,
    config: dict[str, Any],
    features: pd.DataFrame | None,
    params: dict[str, float] | None,
) -> np.ndarray:
    blend_cfg = config["postprocess"].get("notebook_blend") or {}
    if not blend_cfg.get("enabled", False) or features is None:
        return pred
    if params is None:
        params = {
            "alpha": float(blend_cfg.get("alpha", 1.0)),
            "tau": float(blend_cfg.get("tau", 0.0)),
            "w_pf": float(blend_cfg.get("w_pf", 0.0)),
        }

    pf_column = str(blend_cfg.get("pf_column", "kg_pf_ancc_tvt"))
    required = {"last_known_tvt", "md_since", pf_column}
    if not required.issubset(features.columns):
        return pred

    last = features["last_known_tvt"].to_numpy(dtype=float)
    md_since = features["md_since"].to_numpy(dtype=float)
    pf_tvt = features[pf_column].to_numpy(dtype=float)
    valid = np.isfinite(last) & np.isfinite(md_since) & np.isfinite(pf_tvt)
    if not valid.any():
        return pred

    alpha = float(params.get("alpha", 1.0))
    tau = float(params.get("tau", 0.0))
    w_pf = float(np.clip(params.get("w_pf", 0.0), 0.0, 1.0))
    model_delta = pred - last
    pf_delta = pf_tvt - last
    delta = (1.0 - w_pf) * model_delta + w_pf * pf_delta
    if tau > 0:
        delta = delta * (1.0 - np.exp(-np.maximum(md_since, 0.0) / tau))

    blended = pred.copy()
    blended[valid] = last[valid] + alpha * delta[valid]
    return blended


def apply_smoothing(
    pred: np.ndarray, groups: np.ndarray | None, params: dict[str, float] | None
) -> np.ndarray:
    if params is None or groups is None:
        return pred
    window = int(params.get("window", 17))
    polyorder = int(params.get("polyorder", 3))
    smoothed = np.asarray(pred, dtype=float).copy()
    group_array = np.asarray(groups)
    for group in pd.unique(group_array):
        idx = np.flatnonzero(group_array == group)
        width = min(window, len(idx))
        if width % 2 == 0:
            width -= 1
        if width >= polyorder + 2:
            smoothed[idx] = savgol_filter(smoothed[idx], width, polyorder)
    return smoothed


def tune_postprocess(
    flat: np.ndarray,
    residual_pred: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    features: pd.DataFrame,
    groups: np.ndarray,
) -> tuple[
    float, list[dict[str, float]], dict[str, float] | None, dict[str, float] | None
]:
    residual_grid = config["postprocess"].get("residual_weight_grid") or [
        config["postprocess"].get("residual_weight", 1.0)
    ]
    scores: list[dict[str, float]] = []
    for weight, blend, smoothing in product(
        residual_grid,
        notebook_blend_candidates(config),
        smoothing_candidates(config),
    ):
        weight = float(weight)
        pred = apply_postprocess(
            flat,
            residual_pred,
            config,
            residual_weight=weight,
            features=features,
            groups=groups,
            notebook_blend=blend,
            smoothing=smoothing,
        )
        score = {"weight": weight, "rmse": rmse(pred, y_true)}
        if blend is not None:
            score.update({f"blend_{key}": float(value) for key, value in blend.items()})
        if smoothing is not None:
            score.update(
                {f"smooth_{key}": float(value) for key, value in smoothing.items()}
            )
        scores.append(score)

    best = min(scores, key=lambda item: item["rmse"])
    best_blend = {
        key.replace("blend_", ""): best[key]
        for key in ("blend_alpha", "blend_tau", "blend_w_pf")
        if key in best
    }
    best_smoothing = {
        key.replace("smooth_", ""): best[key]
        for key in ("smooth_window", "smooth_polyorder")
        if key in best
    }
    return float(best["weight"]), scores, best_blend or None, best_smoothing or None


def evaluate_oof_predictions(
    residual_pred: np.ndarray,
    X: pd.DataFrame,
    groups: np.ndarray,
    flat: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    logger: RunLogger | None = None,
) -> dict[str, Any]:
    candidate_count = (
        max(len(config["postprocess"].get("residual_weight_grid") or []), 1)
        * max(len(notebook_blend_candidates(config)), 1)
        * max(len(smoothing_candidates(config)), 1)
    )
    if logger is not None:
        logger.info("Tuning postprocess", candidates=candidate_count)
    best_weight, weight_scores, best_notebook_blend, best_smoothing = tune_postprocess(
        flat,
        residual_pred,
        y_true,
        config,
        X,
        groups,
    )
    oof_pred = apply_postprocess(
        flat,
        residual_pred,
        config,
        residual_weight=best_weight,
        features=X,
        groups=groups,
        notebook_blend=best_notebook_blend,
        smoothing=best_smoothing,
    )
    baseline_rmse = rmse(flat, y_true)
    overall_rmse = rmse(oof_pred, y_true)
    if logger is not None:
        logger.metric(
            "OOF postprocessed RMSE",
            rmse=overall_rmse,
            baseline_rmse=baseline_rmse,
            best_residual_weight=f"{best_weight:g}",
            best_notebook_blend=best_notebook_blend,
            best_smoothing=best_smoothing,
        )
    return {
        "enabled": True,
        "rmse": overall_rmse,
        "baseline": config["features"].get("prediction_baseline", "last_known_tvt"),
        "baseline_rmse": baseline_rmse,
        "best_residual_weight": best_weight,
        "best_notebook_blend": best_notebook_blend,
        "best_smoothing": best_smoothing,
        "postprocess_scores": weight_scores,
        "rows": int(len(X)),
        "wells": int(len(set(groups))),
    }
