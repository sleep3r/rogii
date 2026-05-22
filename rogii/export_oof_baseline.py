from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .formation_plane_knn import _rmse, json_safe, write_frame
from .io import horizontal_files, resolve_data_dir, resolve_train_dir
from .modeling import apply_postprocess
from .pipeline import (
    assert_fold_context_safe,
    build_top_context,
    shuffled_path_folds,
)
from .features import build_training_table
from .runlog import RunLogger


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_model(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _artifact_path(model_dir: Path, name: str) -> Path:
    path = model_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return path


def _disable_postprocess_extra(config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    post = out.setdefault("postprocess", {})
    post.setdefault("notebook_blend", {})["enabled"] = False
    post.setdefault("smoothing", {})["enabled"] = False
    return out


def _row_ids(groups: np.ndarray, features: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    row_idx = pd.to_numeric(features["idx"], errors="coerce").round().astype(int).to_numpy()
    ids = [f"{well}_{idx}" for well, idx in zip(groups.astype(str), row_idx, strict=False)]
    return ids, row_idx


def build_oof_support_frame(
    *,
    config: dict[str, Any],
    logger: RunLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = resolve_data_dir(config)
    train_dir = resolve_train_dir(data_dir, config)
    train_paths = horizontal_files(train_dir, config["data"].get("max_train_wells"))
    seed = int(config.get("seed", 42))
    folds = shuffled_path_folds(
        train_paths,
        int(config["validation"].get("n_splits", 5)),
        seed,
    )

    feature_parts: list[pd.DataFrame] = []
    support_parts: list[pd.DataFrame] = []
    group_parts: list[np.ndarray] = []
    flat_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    for fold_id, fold_train_paths, fold_valid_paths in folds:
        context = build_top_context(
            fold_train_paths,
            config,
            logger,
            f"Build fold {fold_id} OOF export context",
        )
        assert_fold_context_safe(context, fold_valid_paths, fold_id)
        with logger.step(
            "Build fold OOF export valid table",
            fold=fold_id,
            valid_wells=len(fold_valid_paths),
        ):
            X_valid, _residual, groups_valid, flat_valid, true_valid = build_training_table(
                fold_valid_paths,
                config,
                context,
                logger,
            )
        ids, row_idx = _row_ids(groups_valid, X_valid)
        support_parts.append(
            pd.DataFrame(
                {
                    "id": ids,
                    "well_id": groups_valid.astype(str),
                    "row_idx": row_idx,
                    "TVT": true_valid.astype(float),
                    "flat_tvt": flat_valid.astype(float),
                }
            )
        )
        feature_parts.append(X_valid)
        group_parts.append(groups_valid.astype(str))
        flat_parts.append(flat_valid.astype(float))
        true_parts.append(true_valid.astype(float))

    return (
        pd.concat(support_parts, ignore_index=True),
        pd.concat(feature_parts, ignore_index=True),
        np.concatenate(group_parts),
        np.concatenate(flat_parts),
        np.concatenate(true_parts),
    )


def export_oof_baseline(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    config = _load_config(Path(args.config) if args.config else _artifact_path(model_dir, "config.yml"))
    config.setdefault("data", {})
    config["data"]["data_dir"] = str(args.data_dir)
    config["data"].setdefault("clearml", {})["enabled"] = False
    if args.train_dir:
        config["data"]["train_dir"] = str(args.train_dir)
    if args.num_workers is not None:
        config.setdefault("features", {})["num_workers"] = int(args.num_workers)
    if args.progress_interval is not None:
        config.setdefault("features", {})["progress_interval"] = int(args.progress_interval)

    logger = RunLogger()
    model = _load_model(_artifact_path(model_dir, "model.pkl"))
    if getattr(model, "oof_residual_", None) is None:
        raise RuntimeError("Model artifact does not contain oof_residual_.")
    residual = np.asarray(model.oof_residual_, dtype=float)

    support, features, groups, flat, y_true = build_oof_support_frame(config=config, logger=logger)
    if len(support) != len(residual):
        raise RuntimeError(
            f"OOF row count mismatch: rebuilt={len(support)} model_residual={len(residual)}"
        )

    raw_config = _disable_postprocess_extra(config)
    raw_pred = apply_postprocess(
        flat,
        residual,
        raw_config,
        features=features,
        groups=groups,
    )
    pp_pred = apply_postprocess(
        flat,
        residual,
        config,
        features=features,
        groups=groups,
    )
    support["schema10_oof_raw"] = raw_pred.astype(float)
    support["schema10_oof_pp"] = pp_pred.astype(float)
    support["oof_residual"] = residual.astype(float)

    output = Path(args.output)
    write_frame(support, output)
    metrics = {
        "model_dir": str(model_dir),
        "output": str(output),
        "rows": int(len(support)),
        "wells": int(support["well_id"].nunique()),
        "schema10_oof_raw_rmse": _rmse(support["schema10_oof_raw"].to_numpy(float), y_true),
        "schema10_oof_pp_rmse": _rmse(support["schema10_oof_pp"].to_numpy(float), y_true),
        "flat_tvt_rmse": _rmse(flat, y_true),
    }
    metrics_path = output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(metrics), file, indent=2)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export row-level OOF baseline predictions from a model artifact.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/clearml/c11ac4df327f49f3b91ad293c69bc91e"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/oof_baseline/schema10_oof.parquet"))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = export_oof_baseline(parse_args(argv))
    print(json.dumps(json_safe(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
