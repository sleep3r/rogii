from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .runlog import RunLogger, format_duration


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


@dataclass(frozen=True)
class PostprocessBasisStats:
    gram: np.ndarray
    target_dot: np.ndarray
    target_norm: float
    rows: int


class EnsembleRegressor(ResidualModel):
    """Grouped OOF ensemble used by both local training and Kaggle inference."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.model_config = config["model"]
        self.seed = seed
        self.base_names: list[str] = []
        self.fold_models: list[list[ResidualModel]] = []
        self.final_models: list[ResidualModel] = []
        self.weights: np.ndarray | None = None
        self.oof_residual_: np.ndarray | None = None
        self.oof_stack_: np.ndarray | None = None
        self.final_iteration_overrides_: dict[str, int] = {}
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
                backend = model_backend_summary(model)
                if logger is not None and backend:
                    logger.info(
                        "Model backend config",
                        model=name,
                        fold=fold_id,
                        **backend,
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
                score = rmse(pred, y[valid_idx])
                fold_scores.append(score)
                if logger is not None:
                    fold_baseline_rmse = rmse(
                        np.zeros_like(y[valid_idx], dtype=float),
                        y[valid_idx],
                    )
                    logger.metric(
                        "OOF fold RMSE",
                        model=name,
                        fold=fold_id,
                        rmse=score,
                        baseline_rmse=fold_baseline_rmse,
                        delta_vs_baseline=score - fold_baseline_rmse,
                    )
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
            allow_negative_weights=bool(
                blend_cfg.get("allow_negative_weights", False)
            ),
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

    def set_oof_results(
        self,
        oof_stack: np.ndarray,
        y: np.ndarray,
        base_names: list[str],
        base_metrics: list[dict[str, Any]],
        iteration_counts: dict[str, list[int]],
        logger: RunLogger | None = None,
    ) -> None:
        self.base_names = list(base_names)
        self.oof_stack_ = np.asarray(oof_stack, dtype=float)
        blend_cfg = self.model_config.get("blend") or {}
        self.weights = fit_hill_climb_weights(
            self.oof_stack_,
            y,
            iterations=int(blend_cfg.get("iterations", 1000)),
            alpha_grid=blend_cfg.get("alpha_grid"),
            allow_negative_weights=bool(
                blend_cfg.get("allow_negative_weights", False)
            ),
        )
        self.oof_residual_ = self.oof_stack_ @ self.weights
        self.final_iteration_overrides_ = {
            name: int(np.median(values))
            for name, values in iteration_counts.items()
            if values
        }

        for item, weight in zip(base_metrics, self.weights, strict=True):
            item["weight"] = float(weight)
            override = self.final_iteration_overrides_.get(str(item["name"]))
            if override is not None:
                item["final_iterations"] = int(override)

        self.metrics_ = {
            "type": "ensemble",
            "n_splits": int(
                max(
                    (len(item.get("fold_rmse", [])) for item in base_metrics), default=0
                )
            ),
            "base_models": base_metrics,
            "blend_method": "hill_climb",
            "blend_weights": dict(
                zip(self.base_names, [float(w) for w in self.weights], strict=True)
            ),
            "final_iteration_overrides": self.final_iteration_overrides_,
            "oof_rmse": rmse(self.oof_residual_, y),
        }
        if logger is not None:
            logger.metric(
                "OOF ensemble RMSE",
                rmse=self.metrics_["oof_rmse"],
                nonzero_weights=int(np.sum(self.weights > 1e-9)),
            )

    def fit_final(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        logger: RunLogger | None = None,
    ) -> "EnsembleRegressor":
        if self.weights is None or not self.base_names:
            raise RuntimeError("OOF blend weights must be fitted before final models.")

        base_specs = self.model_config.get("base_models") or []
        self.final_models = []
        y = np.asarray(y, dtype=float)
        for model_idx, spec in enumerate(base_specs):
            name = model_spec_name(spec, model_idx)
            final_spec = final_model_spec(
                spec,
                self.final_iteration_overrides_.get(name),
            )
            model = make_single_model(
                final_spec,
                self.seed + 10_000 + 1000 * (model_idx + 1),
            )
            context = (
                logger.step(
                    "Final model",
                    model=name,
                    rows=len(X),
                    features=len(X.columns),
                    iterations=self.final_iteration_overrides_.get(name),
                )
                if logger is not None
                else nullcontext()
            )
            with context:
                model.fit(X, y)
            self.final_models.append(model)
        self.metrics_["final_models"] = {
            "count": len(self.final_models),
            "strategy": "full_context",
        }
        return self

    def use_fold_average_models(self) -> None:
        if self.weights is None or not self.base_names:
            raise RuntimeError(
                "OOF blend weights must be fitted before fold averaging."
            )
        if not self.fold_models:
            raise RuntimeError("Fold-average inference requires stored fold models.")
        self.final_models = []
        self.metrics_["final_models"] = {
            "count": int(sum(len(models) for models in self.fold_models)),
            "strategy": "fold_average",
        }

    def predict_base_stack(self, X: pd.DataFrame) -> np.ndarray:
        final_models = getattr(self, "final_models", [])
        if final_models:
            return np.vstack([model.predict(X) for model in final_models]).T
        fold_models = getattr(self, "fold_models", [])
        if not fold_models:
            raise RuntimeError("EnsembleRegressor is not fitted.")
        columns = []
        for models in fold_models:
            fold_pred = np.vstack([model.predict(X) for model in models])
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
    allow_negative_weights: bool = False,
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
    deltas: list[float] = []
    for step in steps:
        deltas.append(step)
        if allow_negative_weights:
            deltas.append(-step)
    moves = 0

    for delta in deltas:
        improved = True
        while improved and moves < max(int(iterations), 1):
            improved = False
            candidate_score = best_score
            candidate_weights = weights
            for model_idx in range(n_models):
                new_weights = weights.copy()
                new_weights[model_idx] += delta
                total = float(new_weights.sum())
                if abs(total) < 1e-12:
                    continue
                new_weights = new_weights / total
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
    if abs(total) < 1e-12:
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

    params = dict(params)
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
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
    if early_stopping_rounds not in (None, ""):
        defaults["early_stopping_rounds"] = int(early_stopping_rounds)
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


def model_backend_summary(model: ResidualModel) -> dict[str, Any]:
    estimator = getattr(model, "estimator", None)
    if estimator is None or not hasattr(estimator, "get_params"):
        return {}
    params = estimator.get_params()
    keys = (
        "task_type",
        "devices",
        "device_type",
        "device",
        "gpu_platform_id",
        "gpu_device_id",
        "gpu_use_dp",
        "tree_method",
        "max_bin",
        "n_jobs",
    )
    return {key: params.get(key) for key in keys if params.get(key) is not None}


def fitted_iteration_count(model: ResidualModel, model_name: str) -> int | None:
    estimator = getattr(model, "estimator", None)
    if estimator is None:
        return None

    tree_count = getattr(estimator, "tree_count_", None)
    if tree_count not in (None, ""):
        count = int(tree_count)
        return count if count > 0 else None

    if model_name in {"lightgbm", "lgbm", "lgb"}:
        best_iteration = getattr(estimator, "best_iteration_", None)
        if best_iteration not in (None, ""):
            count = int(best_iteration)
            return count if count > 0 else None

    best_iteration = getattr(estimator, "best_iteration", None)
    if best_iteration not in (None, ""):
        count = int(best_iteration) + 1
        return count if count > 0 else None
    return None


def final_model_spec(
    spec: dict[str, Any], iteration_count: int | None
) -> dict[str, Any]:
    final_spec = deepcopy(spec)
    params = dict(final_spec.get("params") or {})
    params.pop("early_stopping_rounds", None)
    name = str(final_spec.get("name", "lightgbm")).lower()
    if iteration_count is not None and iteration_count > 0:
        if name in {"catboost", "cat"}:
            params["iterations"] = int(iteration_count)
        else:
            params["n_estimators"] = int(iteration_count)
    final_spec["params"] = params
    return final_spec


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
    pf_columns = notebook_pf_columns(config)

    candidates = blend_cfg.get("candidates") or []
    if candidates:
        parsed: list[dict[str, float] | None] = []
        for candidate in candidates:
            if candidate is None or candidate.get("enabled", True) is False:
                parsed.append(None)
                continue
            parsed_candidate = {
                "alpha": float(candidate.get("alpha", 1.0)),
                "tau": float(candidate.get("tau", 0.0)),
            }
            if has_multi_pf_blend(blend_cfg):
                for column in pf_columns:
                    key = pf_weight_key(column)
                    parsed_candidate[key] = float(candidate.get(key, 0.0))
            else:
                parsed_candidate["w_pf"] = float(candidate.get("w_pf", 0.0))
            if notebook_pf_weight_sum(parsed_candidate, blend_cfg, pf_columns) <= (
                notebook_pf_total_max(blend_cfg) + 1e-12
            ):
                parsed.append(parsed_candidate)
        return parsed

    alpha_grid = grid_values(blend_cfg, "alpha", blend_cfg.get("alpha", 1.0))
    tau_grid = grid_values(blend_cfg, "tau", blend_cfg.get("tau", 0.0))
    if has_multi_pf_blend(blend_cfg):
        weight_grids = [
            (
                pf_weight_key(column),
                grid_values(blend_cfg, pf_weight_key(column), 0.0),
            )
            for column in pf_columns
        ]
        blend_options: list[dict[str, float]] = []
        for alpha, tau in product(alpha_grid, tau_grid):
            for weights in product(*(values for _, values in weight_grids)):
                option = {"alpha": float(alpha), "tau": float(tau)}
                option.update(
                    {
                        key: float(value)
                        for (key, _values), value in zip(weight_grids, weights)
                    }
                )
                if notebook_pf_weight_sum(option, blend_cfg, pf_columns) <= (
                    notebook_pf_total_max(blend_cfg) + 1e-12
                ):
                    blend_options.append(option)
        return blend_options

    w_pf_grid = grid_values(blend_cfg, "w_pf", blend_cfg.get("w_pf", 0.0))
    return [
        {"alpha": float(alpha), "tau": float(tau), "w_pf": float(w_pf)}
        for alpha, tau, w_pf in product(alpha_grid, tau_grid, w_pf_grid)
    ]


def grid_values(config: dict[str, Any], name: str, default: float) -> list[float]:
    explicit = config.get(f"{name}_grid")
    if explicit:
        return [float(value) for value in explicit]
    range_value = config.get(f"{name}_range")
    if range_value:
        if len(range_value) != 3:
            raise ValueError(f"{name}_range must be [start, stop, step].")
        start, stop, step = [float(value) for value in range_value]
        if step <= 0:
            raise ValueError(f"{name}_range step must be positive.")
        values = []
        current = start
        epsilon = step * 1e-6
        while current <= stop + epsilon:
            values.append(round(current, 10))
            current += step
        return values
    return [float(default)]


def smoothing_candidates(config: dict[str, Any]) -> list[dict[str, float] | None]:
    smoothing_cfg = config["postprocess"].get("smoothing") or {}
    if not smoothing_cfg.get("enabled", False):
        return [None]

    candidates = smoothing_cfg.get("candidates")
    if candidates:
        return parse_smoothing_candidates(candidates)

    return [
        {
            "window": int(smoothing_cfg.get("window", 17)),
            "polyorder": int(smoothing_cfg.get("polyorder", 3)),
        }
    ]


def parse_smoothing_candidates(
    candidates: list[dict[str, Any] | None],
) -> list[dict[str, float] | None]:
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


def required_notebook_pf_column(config: dict[str, Any]) -> str:
    blend_cfg = config["postprocess"].get("notebook_blend") or {}
    if not blend_cfg.get("enabled", False):
        return ""
    pf_column = blend_cfg.get("pf_column")
    if pf_column in (None, ""):
        raise ValueError("postprocess.notebook_blend.pf_column is required.")
    return str(pf_column)


def has_multi_pf_blend(blend_cfg: dict[str, Any]) -> bool:
    return bool(blend_cfg.get("pf_columns"))


def notebook_pf_columns(config: dict[str, Any]) -> list[str]:
    blend_cfg = config["postprocess"].get("notebook_blend") or {}
    if not blend_cfg.get("enabled", False):
        return []
    columns = blend_cfg.get("pf_columns")
    if columns:
        parsed = [str(column) for column in columns if str(column)]
        if not parsed:
            raise ValueError("postprocess.notebook_blend.pf_columns must not be empty.")
        return parsed
    return [required_notebook_pf_column(config)]


def pf_weight_key(column: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in str(column)).strip("_")
    return f"w_{sanitized}"


def notebook_pf_total_max(blend_cfg: dict[str, Any]) -> float:
    return float(blend_cfg.get("w_pf_total_max", 1.0))


def notebook_pf_weight_sum(
    params: dict[str, float],
    blend_cfg: dict[str, Any],
    pf_columns: list[str],
) -> float:
    if has_multi_pf_blend(blend_cfg):
        return float(sum(max(0.0, float(params.get(pf_weight_key(column), 0.0))) for column in pf_columns))
    return float(max(0.0, float(params.get("w_pf", 0.0))))


def notebook_pf_weights(
    params: dict[str, float],
    blend_cfg: dict[str, Any],
    pf_columns: list[str],
) -> np.ndarray:
    if has_multi_pf_blend(blend_cfg):
        weights = np.array(
            [
                max(0.0, float(params.get(pf_weight_key(column), 0.0)))
                for column in pf_columns
            ],
            dtype=float,
        )
    else:
        weights = np.array([max(0.0, float(params.get("w_pf", 0.0)))], dtype=float)
    total = float(weights.sum())
    total_max = notebook_pf_total_max(blend_cfg)
    if total > total_max and total > 0.0:
        weights *= total_max / total
    return weights


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
        }
        if has_multi_pf_blend(blend_cfg):
            for column in notebook_pf_columns(config):
                key = pf_weight_key(column)
                params[key] = float(blend_cfg.get(key, 0.0))
        else:
            params["w_pf"] = float(blend_cfg.get("w_pf", 0.0))

    pf_columns = notebook_pf_columns(config)
    required = {"last_known_tvt", "md_from_last_known", *pf_columns}
    if not required.issubset(features.columns):
        return pred

    last = features["last_known_tvt"].to_numpy(dtype=float)
    md_from_last_known = features["md_from_last_known"].to_numpy(dtype=float)
    pf_tvt_matrix = np.column_stack(
        [features[column].to_numpy(dtype=float) for column in pf_columns]
    )
    valid = (
        np.isfinite(last)
        & np.isfinite(md_from_last_known)
        & np.isfinite(pf_tvt_matrix).all(axis=1)
    )
    if not valid.any():
        return pred

    alpha = float(params.get("alpha", 1.0))
    tau = float(params.get("tau", 0.0))
    pf_weights = notebook_pf_weights(params, blend_cfg, pf_columns)
    w_pf_total = float(pf_weights.sum())
    model_delta = pred - last
    pf_delta = pf_tvt_matrix - last[:, None]
    delta = (1.0 - w_pf_total) * model_delta + pf_delta @ pf_weights
    if tau > 0:
        delta = delta * (1.0 - np.exp(-np.maximum(md_from_last_known, 0.0) / tau))

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


def apply_smoothing_to_basis(
    basis: np.ndarray, groups: np.ndarray | None, params: dict[str, float] | None
) -> np.ndarray:
    if params is None or groups is None:
        return basis
    window = int(params.get("window", 17))
    polyorder = int(params.get("polyorder", 3))
    smoothed = np.asarray(basis, dtype=float).copy()
    group_array = np.asarray(groups)
    for group in pd.unique(group_array):
        idx = np.flatnonzero(group_array == group)
        width = min(window, len(idx))
        if width % 2 == 0:
            width -= 1
        if width >= polyorder + 2:
            smoothed[idx, :] = savgol_filter(
                smoothed[idx, :],
                width,
                polyorder,
                axis=0,
            )
    return smoothed


def postprocess_basis_matrix(
    flat: np.ndarray,
    residual: np.ndarray,
    config: dict[str, Any],
    features: pd.DataFrame,
    blend: dict[str, float] | None,
) -> np.ndarray:
    if blend is None:
        return np.column_stack([flat, residual])

    blend_cfg = config["postprocess"].get("notebook_blend") or {}
    if not blend_cfg.get("enabled", False):
        return np.column_stack([flat, residual])

    pf_columns = notebook_pf_columns(config)
    required = {"last_known_tvt", "md_from_last_known", *pf_columns}
    if not required.issubset(features.columns):
        return np.column_stack([flat, residual])

    last = features["last_known_tvt"].to_numpy(dtype=float)
    md_from_last_known = features["md_from_last_known"].to_numpy(dtype=float)
    pf_tvt_matrix = np.column_stack(
        [features[column].to_numpy(dtype=float) for column in pf_columns]
    )
    valid = (
        np.isfinite(last)
        & np.isfinite(md_from_last_known)
        & np.isfinite(pf_tvt_matrix).all(axis=1)
    )
    if not valid.any():
        return np.column_stack([flat, residual])

    tau = float(blend.get("tau", 0.0))
    decay = np.zeros_like(flat, dtype=float)
    if tau > 0:
        decay[valid] = 1.0 - np.exp(-np.maximum(md_from_last_known[valid], 0.0) / tau)
    else:
        decay[valid] = 1.0

    intercept = np.asarray(flat, dtype=float).copy()
    residual_outside_valid = np.asarray(residual, dtype=float).copy()
    model_delta = np.zeros_like(flat, dtype=float)
    residual_inside_valid = np.zeros_like(flat, dtype=float)
    pf_deltas = np.zeros((len(flat), len(pf_columns)), dtype=float)

    intercept[valid] = last[valid]
    residual_outside_valid[valid] = 0.0
    model_delta[valid] = decay[valid] * (flat[valid] - last[valid])
    residual_inside_valid[valid] = decay[valid] * residual[valid]
    pf_deltas[valid, :] = decay[valid, None] * (
        pf_tvt_matrix[valid, :] - last[valid, None]
    )
    return np.column_stack(
        [
            intercept,
            residual_outside_valid,
            model_delta,
            residual_inside_valid,
            pf_deltas,
        ]
    )


def postprocess_coefficients(
    weight: float,
    blend: dict[str, float] | None,
    basis_width: int,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    if blend is None or basis_width == 2:
        return np.array([1.0, float(weight)], dtype=float)
    alpha = float(blend.get("alpha", 1.0))
    if config is None:
        w_pf = float(np.clip(blend.get("w_pf", 0.0), 0.0, 1.0))
        return np.array(
            [
                1.0,
                float(weight),
                alpha * (1.0 - w_pf),
                alpha * float(weight) * (1.0 - w_pf),
                alpha * w_pf,
            ],
            dtype=float,
        )

    blend_cfg = config["postprocess"].get("notebook_blend") or {}
    pf_columns = notebook_pf_columns(config)
    pf_weights = notebook_pf_weights(blend, blend_cfg, pf_columns)
    w_pf_total = float(pf_weights.sum())
    return np.array(
        [
            1.0,
            float(weight),
            alpha * (1.0 - w_pf_total),
            alpha * float(weight) * (1.0 - w_pf_total),
            *[alpha * value for value in pf_weights],
        ],
        dtype=float,
    )


def smoothing_key(params: dict[str, float] | None) -> tuple[str, int, int]:
    if params is None:
        return ("none", 0, 0)
    return (
        "savgol",
        int(params.get("window", 17)),
        int(params.get("polyorder", 3)),
    )


def blend_basis_key(blend: dict[str, float] | None) -> tuple[str, float]:
    if blend is None:
        return ("none", 0.0)
    return ("notebook", float(blend.get("tau", 0.0)))


def postprocess_basis_stats(
    flat: np.ndarray,
    residual: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    features: pd.DataFrame,
    groups: np.ndarray,
    blend: dict[str, float] | None,
    smoothing: dict[str, float] | None,
) -> PostprocessBasisStats:
    basis = postprocess_basis_matrix(flat, residual, config, features, blend)
    basis = apply_smoothing_to_basis(basis, groups, smoothing)
    return PostprocessBasisStats(
        gram=basis.T @ basis,
        target_dot=basis.T @ y_true,
        target_norm=float(y_true @ y_true),
        rows=int(len(y_true)),
    )


def rmse_from_basis_stats(stats: PostprocessBasisStats, coeffs: np.ndarray) -> float:
    sse = (
        float(coeffs @ stats.gram @ coeffs)
        - 2.0 * float(coeffs @ stats.target_dot)
        + stats.target_norm
    )
    return float(np.sqrt(max(sse, 0.0) / max(stats.rows, 1)))


def postprocess_residual_grid(config: dict[str, Any]) -> list[float]:
    grid = config["postprocess"].get("residual_weight_grid")
    if grid:
        return [float(value) for value in grid]
    return [float(config["postprocess"].get("residual_weight", 1.0))]


def postprocess_search_method(config: dict[str, Any]) -> str:
    return str(config["postprocess"].get("search_method", "grid")).lower()


def postprocess_candidate_count(config: dict[str, Any]) -> int:
    if postprocess_search_method(config) == "optuna":
        return max(1, int(config["postprocess"].get("optuna_trials", 500)))
    return (
        max(len(postprocess_residual_grid(config)), 1)
        * max(len(notebook_blend_candidates(config)), 1)
        * max(len(smoothing_candidates(config)), 1)
    )


def _tune_postprocess_grid(
    flat: np.ndarray,
    residual_pred: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    features: pd.DataFrame,
    groups: np.ndarray,
    logger: RunLogger | None = None,
) -> tuple[
    float, list[dict[str, float]], dict[str, float] | None, dict[str, float] | None
]:
    residual_grid = postprocess_residual_grid(config)
    blend_options = notebook_blend_candidates(config)
    smoothing_options = smoothing_candidates(config)
    total_candidates = (
        max(len(residual_grid), 1)
        * max(len(blend_options), 1)
        * max(len(smoothing_options), 1)
    )
    progress_interval = int(
        config["postprocess"].get("progress_interval") or max(1, total_candidates // 20)
    )
    flat = np.asarray(flat, dtype=float)
    residual_pred = np.asarray(residual_pred, dtype=float)
    clip_value = config["postprocess"].get("residual_clip")
    if clip_value not in (None, ""):
        residual_pred = np.clip(residual_pred, -float(clip_value), float(clip_value))
    y_true = np.asarray(y_true, dtype=float)

    scores: list[dict[str, float]] = []
    best_score = float("inf")
    best: dict[str, float] | None = None
    started_at = perf_counter()
    stats_cache: dict[
        tuple[tuple[str, float], tuple[str, int, int]], PostprocessBasisStats
    ] = {}
    for candidate_idx, (weight, blend, smoothing) in enumerate(
        product(
            residual_grid,
            blend_options,
            smoothing_options,
        ),
        start=1,
    ):
        weight = float(weight)
        basis_key = (blend_basis_key(blend), smoothing_key(smoothing))
        stats = stats_cache.get(basis_key)
        if stats is None:
            stats = postprocess_basis_stats(
                flat,
                residual_pred,
                y_true,
                config,
                features,
                groups,
                blend,
                smoothing,
            )
            stats_cache[basis_key] = stats
        coeffs = postprocess_coefficients(weight, blend, len(stats.target_dot), config)
        score = {"weight": weight, "rmse": rmse_from_basis_stats(stats, coeffs)}
        if blend is not None:
            score.update({f"blend_{key}": float(value) for key, value in blend.items()})
        if smoothing is not None:
            score.update(
                {f"smooth_{key}": float(value) for key, value in smoothing.items()}
            )
        scores.append(score)
        if score["rmse"] < best_score:
            best_score = float(score["rmse"])
            best = score

        if logger is not None and (
            candidate_idx == 1
            or candidate_idx == total_candidates
            or candidate_idx % progress_interval == 0
        ):
            elapsed = perf_counter() - started_at
            rate = candidate_idx / max(elapsed, 1e-9)
            remaining = (total_candidates - candidate_idx) / max(rate, 1e-9)
            best_blend = {
                key.replace("blend_", ""): value
                for key, value in (best or {}).items()
                if key.startswith("blend_")
            }
            best_smooth = {
                key.replace("smooth_", ""): best[key]
                for key in ("smooth_window", "smooth_polyorder")
                if best is not None and key in best
            }
            logger.info(
                "Postprocess tuning progress",
                current=candidate_idx,
                total=total_candidates,
                pct=100.0 * candidate_idx / max(total_candidates, 1),
                elapsed=format_duration(elapsed),
                eta=format_duration(remaining),
                best_rmse=best_score,
                best_weight=best.get("weight") if best is not None else None,
                best_blend=best_blend or None,
                best_smoothing=best_smooth or None,
            )

    if best is None:
        best = min(scores, key=lambda item: item["rmse"])
    best_blend = {
        key.replace("blend_", ""): value
        for key, value in best.items()
        if key.startswith("blend_")
    }
    best_smoothing = {
        key.replace("smooth_", ""): best[key]
        for key in ("smooth_window", "smooth_polyorder")
        if key in best
    }
    return float(best["weight"]), scores, best_blend or None, best_smoothing or None


def optuna_suggest_grid_or_range(
    trial: Any,
    config: dict[str, Any],
    name: str,
    default: float,
) -> float:
    explicit = config.get(f"{name}_grid")
    if explicit:
        return float(
            trial.suggest_categorical(
                name,
                [float(value) for value in explicit],
            )
        )
    range_value = config.get(f"{name}_range")
    if range_value:
        if len(range_value) != 3:
            raise ValueError(f"{name}_range must be [start, stop, step].")
        start, stop, step = [float(value) for value in range_value]
        if step <= 0:
            raise ValueError(f"{name}_range step must be positive.")
        return float(trial.suggest_float(name, start, stop, step=step))
    return float(config.get(name, default))


def _tune_postprocess_optuna(
    flat: np.ndarray,
    residual_pred: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    features: pd.DataFrame,
    groups: np.ndarray,
    logger: RunLogger | None = None,
) -> tuple[
    float, list[dict[str, float]], dict[str, float] | None, dict[str, float] | None
]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "postprocess.search_method=optuna requires the optuna package."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trials = max(1, int(config["postprocess"].get("optuna_trials", 500)))
    seed = int(config["postprocess"].get("optuna_seed", config.get("seed", 42)))
    progress_interval = int(
        config["postprocess"].get("progress_interval") or max(1, trials // 20)
    )
    residual_grid = postprocess_residual_grid(config)
    smoothing_options = smoothing_candidates(config)
    blend_cfg = config["postprocess"].get("notebook_blend") or {}
    blend_enabled = bool(blend_cfg.get("enabled", False))
    pf_columns = notebook_pf_columns(config) if blend_enabled else []

    flat = np.asarray(flat, dtype=float)
    residual_pred = np.asarray(residual_pred, dtype=float)
    clip_value = config["postprocess"].get("residual_clip")
    if clip_value not in (None, ""):
        residual_pred = np.clip(residual_pred, -float(clip_value), float(clip_value))
    y_true = np.asarray(y_true, dtype=float)

    scores: list[dict[str, float]] = []
    best_score = float("inf")
    best: dict[str, float] | None = None
    started_at = perf_counter()
    stats_cache: dict[
        tuple[tuple[str, float], tuple[str, int, int]], PostprocessBasisStats
    ] = {}

    def objective(trial: Any) -> float:
        nonlocal best, best_score

        if len(residual_grid) == 1:
            weight = residual_grid[0]
        else:
            weight = float(trial.suggest_categorical("residual_weight", residual_grid))

        blend: dict[str, float] | None = None
        if blend_enabled:
            blend = {
                "alpha": optuna_suggest_grid_or_range(
                    trial,
                    blend_cfg,
                    "alpha",
                    float(blend_cfg.get("alpha", 1.0)),
                ),
                "tau": optuna_suggest_grid_or_range(
                    trial,
                    blend_cfg,
                    "tau",
                    float(blend_cfg.get("tau", 0.0)),
                ),
            }
            if has_multi_pf_blend(blend_cfg):
                for column in pf_columns:
                    key = pf_weight_key(column)
                    blend[key] = optuna_suggest_grid_or_range(trial, blend_cfg, key, 0.0)
            else:
                blend["w_pf"] = optuna_suggest_grid_or_range(
                    trial,
                    blend_cfg,
                    "w_pf",
                    float(blend_cfg.get("w_pf", 0.0)),
                )
            if notebook_pf_weight_sum(blend, blend_cfg, pf_columns) > (
                notebook_pf_total_max(blend_cfg) + 1e-12
            ):
                return 1e12

        if len(smoothing_options) == 1:
            smoothing = smoothing_options[0]
        else:
            smoothing_idx = int(
                trial.suggest_int("smoothing_idx", 0, len(smoothing_options) - 1)
            )
            smoothing = smoothing_options[smoothing_idx]

        basis_key = (blend_basis_key(blend), smoothing_key(smoothing))
        stats = stats_cache.get(basis_key)
        if stats is None:
            stats = postprocess_basis_stats(
                flat,
                residual_pred,
                y_true,
                config,
                features,
                groups,
                blend,
                smoothing,
            )
            stats_cache[basis_key] = stats
        coeffs = postprocess_coefficients(weight, blend, len(stats.target_dot), config)
        score = {"weight": weight, "rmse": rmse_from_basis_stats(stats, coeffs)}
        if blend is not None:
            score.update({f"blend_{key}": float(value) for key, value in blend.items()})
        if smoothing is not None:
            score.update(
                {f"smooth_{key}": float(value) for key, value in smoothing.items()}
            )
        scores.append(score)
        if score["rmse"] < best_score:
            best_score = float(score["rmse"])
            best = score
        return float(score["rmse"])

    def progress_callback(_study: Any, trial: Any) -> None:
        trial_idx = int(trial.number) + 1
        if logger is None or not (
            trial_idx == 1 or trial_idx == trials or trial_idx % progress_interval == 0
        ):
            return
        elapsed = perf_counter() - started_at
        rate = trial_idx / max(elapsed, 1e-9)
        remaining = (trials - trial_idx) / max(rate, 1e-9)
        best_blend = {
            key.replace("blend_", ""): value
            for key, value in (best or {}).items()
            if key.startswith("blend_")
        }
        best_smooth = {
            key.replace("smooth_", ""): best[key]
            for key in ("smooth_window", "smooth_polyorder")
            if best is not None and key in best
        }
        logger.info(
            "Postprocess optuna progress",
            current=trial_idx,
            total=trials,
            pct=100.0 * trial_idx / max(trials, 1),
            elapsed=format_duration(elapsed),
            eta=format_duration(remaining),
            best_rmse=best_score,
            best_weight=best.get("weight") if best is not None else None,
            best_blend=best_blend or None,
            best_smoothing=best_smooth or None,
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, callbacks=[progress_callback])

    if best is None:
        best = min(scores, key=lambda item: item["rmse"])
    best_blend = {
        key.replace("blend_", ""): value
        for key, value in best.items()
        if key.startswith("blend_")
    }
    best_smoothing = {
        key.replace("smooth_", ""): best[key]
        for key in ("smooth_window", "smooth_polyorder")
        if key in best
    }
    return float(best["weight"]), scores, best_blend or None, best_smoothing or None


def tune_postprocess(
    flat: np.ndarray,
    residual_pred: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    features: pd.DataFrame,
    groups: np.ndarray,
    logger: RunLogger | None = None,
) -> tuple[
    float, list[dict[str, float]], dict[str, float] | None, dict[str, float] | None
]:
    method = postprocess_search_method(config)
    if method == "optuna":
        return _tune_postprocess_optuna(
            flat,
            residual_pred,
            y_true,
            config,
            features,
            groups,
            logger=logger,
        )
    if method != "grid":
        raise ValueError(f"Unsupported postprocess.search_method: {method!r}.")
    return _tune_postprocess_grid(
        flat,
        residual_pred,
        y_true,
        config,
        features,
        groups,
        logger=logger,
    )


def evaluate_oof_predictions(
    residual_pred: np.ndarray,
    X: pd.DataFrame,
    groups: np.ndarray,
    flat: np.ndarray,
    y_true: np.ndarray,
    config: dict[str, Any],
    logger: RunLogger | None = None,
) -> dict[str, Any]:
    candidate_count = postprocess_candidate_count(config)
    if logger is not None:
        logger.info(
            "Tuning postprocess",
            method=postprocess_search_method(config),
            candidates=candidate_count,
        )
    best_weight, weight_scores, best_notebook_blend, best_smoothing = tune_postprocess(
        flat,
        residual_pred,
        y_true,
        config,
        X,
        groups,
        logger=logger,
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
