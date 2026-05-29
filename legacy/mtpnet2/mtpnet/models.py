"""LightGBM/CatBoost multiclass classifier for offset bin prediction.

Design:
  - Both global (1 pred per well) and per-segment (K preds per well) use the
    same classifier interface: train_offset_classifier / predict_offset_proba
  - Output is a soft probability distribution over N_OFFSET_BINS bins
  - Default backend: LightGBM (faster iteration); CatBoost available as fallback
  - Hyperparameters are reasonable defaults; tuning happens in experiment scripts

Usage:
    from mtpnet.models import train_offset_classifier, predict_offset_proba

    model = train_offset_classifier(X_train, y_train)
    proba = predict_offset_proba(model, X_test)   # (N_test, N_BINS)
    bins  = np.argmax(proba, axis=1)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .offsets import N_OFFSET_BINS

# ---------------------------------------------------------------------------
# LightGBM defaults
# ---------------------------------------------------------------------------

LGBM_DEFAULTS: dict[str, Any] = {
    "objective":       "multiclass",
    "num_class":       N_OFFSET_BINS,
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "n_jobs":          -1,
    "verbose":         -1,
    "random_state":    42,
}

# CatBoost defaults (lower iteration count — slower but often better on small data)
CATBOOST_DEFAULTS: dict[str, Any] = {
    "loss_function":   "MultiClass",
    "classes_count":   N_OFFSET_BINS,
    "iterations":      300,
    "learning_rate":   0.05,
    "depth":           6,
    "l2_leaf_reg":     3.0,
    "random_seed":     42,
    "verbose":         0,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_offset_classifier(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    backend: str = "lgbm",
    params: dict | None = None,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> Any:
    """Train a multiclass offset classifier.

    Args:
        X       : (N, F) float32 feature matrix
        y       : (N,) int32 bin labels in [0, N_OFFSET_BINS)
        X_val   : optional validation features for early stopping
        y_val   : optional validation labels for early stopping
        backend : "lgbm" (default) or "catboost"
        params  : override default hyper-parameters (merged, not replaced)
        early_stopping_rounds: 0 to disable
        verbose : print training progress

    Returns:
        Fitted model object (LGBMClassifier or CatBoostClassifier)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if backend == "lgbm":
        return _train_lgbm(
            X, y, X_val, y_val, params, early_stopping_rounds, verbose
        )
    elif backend == "catboost":
        return _train_catboost(
            X, y, X_val, y_val, params, early_stopping_rounds, verbose
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'lgbm' or 'catboost'.")


# ---------------------------------------------------------------------------
# Thin wrapper for native LightGBM booster to expose sklearn-style API
# ---------------------------------------------------------------------------

class _LGBMWrapper:
    """Wraps lgb.Booster to expose predict_proba() + feature_importances_."""

    def __init__(self, booster):
        self.booster = booster
        self.feature_importances_ = booster.feature_importance(importance_type="gain")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        raw = self.booster.predict(X)   # (N, N_BINS) for multiclass
        if raw.ndim == 1:
            # Binary edge case: shouldn't happen but guard it
            raw = np.column_stack([1 - raw, raw])
        return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def _train_lgbm(
    X, y, X_val, y_val, params, early_stopping_rounds, verbose
):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required: pip install lightgbm")

    kw = {**LGBM_DEFAULTS}
    if params:
        kw.update(params)

    # Use native API to avoid sklearn label-encoding issues
    # (the sklearn wrapper breaks when val has bins not seen in train)
    n_estimators = kw.pop("n_estimators", 500)
    kw.pop("random_state", None)
    kw["seed"] = kw.pop("seed", 42) if "seed" in kw else 42
    kw["n_jobs"] = kw.pop("n_jobs", -1)
    kw["verbosity"] = -1

    train_data = lgb.Dataset(X, label=y)
    callbacks = []
    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None and early_stopping_rounds > 0:
        val_data = lgb.Dataset(
            np.asarray(X_val, np.float32),
            label=np.asarray(y_val, np.int32),
            reference=train_data,
        )
        valid_sets.append(val_data)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        if verbose:
            callbacks.append(lgb.log_evaluation(period=100))

    booster = lgb.train(
        kw,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks if callbacks else None,
    )

    # Wrap in a thin object that has .predict_proba() and .feature_importances_
    return _LGBMWrapper(booster)


def _train_catboost(
    X, y, X_val, y_val, params, early_stopping_rounds, verbose
):
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        raise ImportError("catboost is required: pip install catboost")

    kw = {**CATBOOST_DEFAULTS}
    if params:
        kw.update(params)

    model = CatBoostClassifier(**kw)

    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = Pool(np.asarray(X_val, np.float32), label=y_val)

    model.fit(
        Pool(X, label=y),
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
    )
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_offset_proba(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Predict soft probability distribution over offset bins.

    Args:
        model : fitted classifier (LGBMClassifier or CatBoostClassifier)
        X     : (N, F) feature matrix

    Returns:
        (N, N_OFFSET_BINS) float32 probability array (rows sum to ~1)
    """
    X = np.asarray(X, dtype=np.float32)
    proba = model.predict_proba(X)
    return proba.astype(np.float32)


def predict_offset_bin(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Predict the argmax offset bin (hard assignment).

    Returns:
        (N,) int32 array of bin indices
    """
    proba = predict_offset_proba(model, X)
    return np.argmax(proba, axis=1).astype(np.int32)


def predict_offset_value(
    model: Any,
    X: np.ndarray,
    grid: np.ndarray,
    mode: str = "argmax",
) -> np.ndarray:
    """Predict offset value for each sample.

    Args:
        mode : "argmax" (MAP estimate) or "mean" (posterior mean)

    Returns:
        (N,) float32 array of offset values
    """
    proba = predict_offset_proba(model, X)
    if mode == "argmax":
        bins = np.argmax(proba, axis=1)
        return grid[bins].astype(np.float32)
    elif mode == "mean":
        g = np.asarray(grid, dtype=np.float64)
        return (proba.astype(np.float64) @ g).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: Any,
    feature_names: list[str] | None = None,
) -> dict[str, float] | None:
    """Extract feature importance from a fitted model (if available).

    Returns:
        Dict mapping feature_name → importance (or None if not available).
    """
    backend = type(model).__module__.split(".")[0]

    if isinstance(model, _LGBMWrapper) or backend == "lightgbm":
        imp = model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp), key=lambda x: -x[1]))

    elif backend == "catboost":
        try:
            imp = model.get_feature_importance()
            names = feature_names or [f"f{i}" for i in range(len(imp))]
            return dict(sorted(zip(names, imp), key=lambda x: -x[1]))
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str) -> None:
    """Save a fitted model to disk (pickle for LGBM, native for CatBoost)."""
    import pickle
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, _LGBMWrapper):
        model.booster.save_model(path)
        return
    backend = type(model).__module__.split(".")[0]
    if backend == "catboost":
        model.save_model(path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_model(path: str, backend: str = "lgbm") -> Any:
    """Load a model from disk."""
    import pickle
    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("catboost required")
        model = CatBoostClassifier()
        model.load_model(path)
        return model
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
