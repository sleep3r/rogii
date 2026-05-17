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
        "include_typewell": True,
        "include_kaggle_top_signals": True,
        "kaggle_top": {
            "beam_configs": [
                [20.0, 144.0, 2, "cons"],
                [8.0, 64.0, 2, "loose"],
                [14.0, 90.0, 5, "sm5"],
                [25.0, 180.0, 2, "stiff"],
            ],
            "ncc_windows": [8, 15, 25],
            "ncc_stride": 3,
            "dtw_enabled": True,
            "dtw_max_query_points": 700,
            "dtw_max_ref_points": 700,
            "dtw_radius": 35,
            "spatial_k": 10,
            "dense_k": 20,
            "dense_samples_per_well": 60,
        },
    },
    "model": {
        "name": "hist_gradient_boosting",
        "params": {
            "loss": "squared_error",
            "learning_rate": 0.04,
            "max_iter": 350,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.05,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 25,
        },
    },
    "validation": {
        "enabled": True,
        "n_splits": 5,
        "max_wells": 150,
    },
    "postprocess": {
        "residual_weight": "auto",
        "residual_weight_grid": [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
        "residual_clip": 250.0,
    },
    "reporting": {
        "compute_train_metrics": True,
    },
    "outputs": {
        "output_dir": "artifacts/hgb",
        "submission_path": "submission.csv",
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
