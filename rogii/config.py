from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "data_dir": None,
        "train_dir": None,
        "test_dir": None,
        "sample_submission": None,
        "max_train_wells": None,
        "max_test_wells": None,
        "target_rows": "hidden_only",
    },
    "features": {
        "tail_windows": [25, 100, 250],
        "rolling_windows": [5, 25, 101],
        "prediction_baseline": "last_known_tvt",
        "cache": {
            "enabled": False,
            "dir": "artifacts/feature_cache",
        },
        "include_typewell": True,
        "include_kaggle_top_signals": True,
        "kaggle_top": {
            "mode": "notebook",
            "beam_configs": [
                [10, 20.0, 144.0, 2, "cons"],
                [10, 8.0, 64.0, 2, "loose"],
                [8, 35.0, 220.0, 1, "vcons"],
                [10, 14.0, 90.0, 5, "sm5"],
                [20, 4.0, 36.0, 3, "vloose"],
                [12, 12.0, 100.0, 3, "mid"],
                [15, 25.0, 180.0, 2, "stiff"],
            ],
            "ncc_windows": [8, 15, 25],
            "ncc_stride": 3,
            "dtw_enabled": True,
            "dtw_max_query_points": 700,
            "dtw_max_ref_points": 700,
            "dtw_radius": 35,
            "dtw_radii": [20, 50, 100, 200],
            "dtw_stochastic_enabled": True,
            "dtw_stochastic_radius": 50,
            "dtw_stochastic_k": 12,
            "dtw_stochastic_temperature": 3.0,
            "dwt_enabled": True,
            "dwt_wavelet": "db4",
            "dwt_level": 3,
            "dwt_radii": [20, 50, 100, 200],
            "spatial_k": 10,
            "dense_k": 20,
            "dense_fetch": 5000,
            "dense_query_chunk": 512,
            "dense_samples_per_well": 60,
            "particle_enabled": True,
            "particle_count": 192,
            "ancc_particle_count": 192,
        },
    },
    "model": {
        "base_models": [
            {
                "id": "lgb",
                "name": "lightgbm",
                "params": {"n_estimators": 700, "learning_rate": 0.035},
            },
            {
                "id": "xgb",
                "name": "xgboost",
                "params": {"n_estimators": 650, "learning_rate": 0.035},
            },
            {
                "id": "cat",
                "name": "catboost",
                "params": {"iterations": 900, "learning_rate": 0.035},
            },
        ],
        "blend": {
            "method": "hill_climb",
            "alpha_grid": [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
            "iterations": 1000,
        },
    },
    "validation": {
        "n_splits": 5,
        "fold_safe_context": True,
        "final_model_strategy": "full_context",
    },
    "postprocess": {
        "progress_interval": 200,
        "residual_weight": 1.0,
        "residual_weight_grid": [0.7, 0.8, 0.9, 1.0, 1.1],
        "residual_clip": 250.0,
        "notebook_blend": {
            "enabled": True,
            "pf_column": "kg_pf_ancc_tvt",
            "alpha": 1.0,
            "tau": 0.0,
            "w_pf": 0.0,
            "candidates": [],
        },
        "smoothing": {
            "enabled": True,
            "candidates": [{"enabled": False}, {"window": 17, "polyorder": 3}],
        },
    },
    "reporting": {
        "compute_train_metrics": True,
    },
    "outputs": {
        "output_dir": "artifacts/stack",
        "submission_path": "submission.csv",
    },
    "runs": {
        "registry_path": "artifacts/runs.csv",
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    else:
        raise FileNotFoundError(f"Config file not found: {path}")

    parent = loaded.pop("inherits", None)
    if parent not in (None, ""):
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        config = load_config(parent_path)
    else:
        config = DEFAULT_CONFIG
    return deep_update(config, loaded)
