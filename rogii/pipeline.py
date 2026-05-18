from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import load_config
from .diagnostics import append_run_registry, git_hash, regression_diagnostics
from .features import FEATURE_CACHE_SCHEMA_VERSION, build_training_table
from .io import (
    horizontal_files,
    resolve_data_dir,
    resolve_sample_submission,
    resolve_test_dir,
    resolve_train_dir,
    well_name,
)
from .modeling import (
    EnsembleRegressor,
    apply_postprocess,
    evaluate_oof_predictions,
    fitted_iteration_count,
    make_single_model,
    model_spec_name,
    rmse,
)
from .runlog import RunLogger
from .spatial import KaggleTopContext, context_well_overlap
from .submission import predict_test, save_outputs


def print_config(config: dict[str, Any], logger: RunLogger) -> None:
    logger.log("CONFIG", "Resolved training config")
    print("=" * 88, flush=True)
    print(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False).rstrip(),
        flush=True,
    )
    print("=" * 88, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ROGII TVT residual model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stack.yml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Override data.data_dir."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override outputs.output_dir."
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=None,
        help="Override outputs.submission_path.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id for registry.")
    parser.add_argument("--notes", default="", help="Optional run notes for registry.")
    parser.add_argument(
        "--public-lb",
        type=float,
        default=None,
        help="Optional public leaderboard score to record.",
    )
    return parser.parse_args()


def default_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"


def shuffled_path_folds(
    paths: list[Path], n_splits: int, seed: int
) -> list[tuple[int, list[Path], list[Path]]]:
    if len(paths) < 2:
        raise ValueError("Fold-safe validation needs at least two train wells.")
    ordered = sorted(paths, key=well_name)
    wells = np.array([well_name(path) for path in ordered])
    unique_wells = np.array(sorted(set(wells)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_wells)
    n_splits = min(max(2, int(n_splits)), len(unique_wells))
    folds = []
    for fold_index in range(n_splits):
        valid_wells = set(unique_wells[fold_index::n_splits])
        train_paths = [path for path in ordered if well_name(path) not in valid_wells]
        valid_paths = [path for path in ordered if well_name(path) in valid_wells]
        folds.append((fold_index + 1, train_paths, valid_paths))
    return folds


def build_top_context(
    paths: list[Path], config: dict[str, Any], logger: RunLogger, label: str
) -> KaggleTopContext | None:
    if not config["features"].get("include_kaggle_top_signals", False):
        return None
    with logger.step(label, train_wells=len(paths)):
        context = KaggleTopContext(paths, config)
    logger.info(
        "Spatial context ready",
        context_key=context.context_key,
        formation_wells=len(getattr(context, "formation_values", [])),
        dense_ancc_points=len(getattr(context, "dense_ancc", [])),
    )
    return context


def assert_fold_context_safe(
    context: KaggleTopContext | None, valid_paths: list[Path], fold_id: int
) -> None:
    if context is None:
        return
    overlap = context_well_overlap(context, valid_paths)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(
            f"Fold {fold_id} context contains validation wells: {sample}"
        )


def align_valid_features(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in X_train.columns if column not in X_valid.columns]
    aligned = X_valid.reindex(columns=X_train.columns)
    if missing:
        aligned.loc[:, missing] = np.nan
    return aligned.astype("float32")


def train_fold_safe_oof(
    model: EnsembleRegressor,
    train_paths: list[Path],
    config: dict[str, Any],
    seed: int,
    logger: RunLogger,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_specs = config["model"].get("base_models") or []
    if not base_specs:
        raise ValueError("model.base_models must contain at least one model.")

    folds = shuffled_path_folds(
        train_paths,
        int(config["validation"].get("n_splits", 5)),
        seed,
    )
    base_names = [model_spec_name(spec, index) for index, spec in enumerate(base_specs)]
    base_metrics = [
        {"name": name, "oof_rmse": None, "fold_rmse": []} for name in base_names
    ]
    iteration_counts: dict[str, list[int]] = {name: [] for name in base_names}
    valid_features: list[pd.DataFrame] = []
    residual_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    flat_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    stack_parts: list[np.ndarray] = []

    logger.info("Prepared fold-safe CV", folds=len(folds), models=len(base_specs))
    for fold_id, fold_train_paths, fold_valid_paths in folds:
        context = build_top_context(
            fold_train_paths,
            config,
            logger,
            f"Build fold {fold_id} Kaggle top context",
        )
        assert_fold_context_safe(context, fold_valid_paths, fold_id)

        with logger.step(
            "Build fold train table",
            fold=fold_id,
            train_wells=len(fold_train_paths),
        ):
            X_train, y_train, _train_groups, _train_flat, _train_true = (
                build_training_table(fold_train_paths, config, context, logger)
            )
        with logger.step(
            "Build fold valid table",
            fold=fold_id,
            valid_wells=len(fold_valid_paths),
        ):
            X_valid, y_valid, groups_valid, flat_valid, true_valid = (
                build_training_table(fold_valid_paths, config, context, logger)
            )
        X_valid_model = align_valid_features(X_train, X_valid)
        fold_stack = np.zeros((len(X_valid_model), len(base_specs)), dtype=float)

        for model_idx, spec in enumerate(base_specs):
            name = base_names[model_idx]
            base_model = make_single_model(
                spec,
                seed + 1000 * (model_idx + 1) + fold_id,
            )
            model_name = str(spec.get("name", "lightgbm")).lower()
            with logger.step(
                "OOF model fold",
                model=name,
                fold=fold_id,
                train_rows=len(X_train),
                valid_rows=len(X_valid_model),
            ):
                base_model.fit(
                    X_train,
                    y_train,
                    X_valid=X_valid_model,
                    y_valid=y_valid,
                )
                pred = base_model.predict(X_valid_model)
            fold_stack[:, model_idx] = pred
            score = rmse(pred, y_valid)
            base_metrics[model_idx]["fold_rmse"].append(float(score))
            iteration_count = fitted_iteration_count(base_model, model_name)
            if iteration_count is not None:
                iteration_counts[name].append(iteration_count)
            logger.metric("OOF fold RMSE", model=name, fold=fold_id, rmse=score)

        valid_features.append(X_valid)
        residual_parts.append(y_valid.astype("float32"))
        group_parts.append(groups_valid)
        flat_parts.append(flat_valid.astype("float32"))
        true_parts.append(true_valid.astype("float32"))
        stack_parts.append(fold_stack)

    X_oof = pd.concat(valid_features, axis=0, ignore_index=True)
    residual_oof = np.concatenate(residual_parts)
    groups_oof = np.concatenate(group_parts)
    flat_oof = np.concatenate(flat_parts)
    true_oof = np.concatenate(true_parts)
    oof_stack = np.vstack(stack_parts)
    for model_idx, name in enumerate(base_names):
        base_metrics[model_idx]["oof_rmse"] = rmse(
            oof_stack[:, model_idx], residual_oof
        )
        logger.metric(
            "OOF base RMSE",
            model=name,
            rmse=base_metrics[model_idx]["oof_rmse"],
        )

    model.set_oof_results(
        oof_stack,
        residual_oof,
        base_names,
        base_metrics,
        iteration_counts,
        logger,
    )
    return X_oof, residual_oof, groups_oof, flat_oof, true_oof


def main() -> None:
    run_started_at = perf_counter()
    args = parse_args()
    run_id = args.run_id or default_run_id()
    logger = RunLogger()
    logger.log("RUN", "ROGII training started", config=args.config, run_id=run_id)

    config = load_config(args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = str(args.data_dir)
    if args.output_dir is not None:
        config["outputs"]["output_dir"] = str(args.output_dir)
    if args.submission is not None:
        config["outputs"]["submission_path"] = str(args.submission)
    print_config(config, logger)
    seed = int(config.get("seed", 42))
    data_dir = resolve_data_dir(config)
    train_dir = resolve_train_dir(data_dir, config)
    test_dir = resolve_test_dir(data_dir, config)
    sample_submission_path = resolve_sample_submission(data_dir, test_dir, config)

    logger.info(
        "Resolved paths",
        data_dir=data_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        sample_submission=sample_submission_path,
    )

    train_paths = horizontal_files(train_dir, config["data"].get("max_train_wells"))
    test_paths = horizontal_files(test_dir, config["data"].get("max_test_wells"))
    logger.info(
        "Discovered wells", train_wells=len(train_paths), test_wells=len(test_paths)
    )

    def apply_cv_selection(cv_metrics: dict[str, Any]) -> None:
        best_weight = cv_metrics.get("best_residual_weight")
        if best_weight is not None:
            config["postprocess"]["residual_weight"] = best_weight
        best_notebook_blend = cv_metrics.get("best_notebook_blend")
        notebook_blend_cfg = config["postprocess"].get("notebook_blend") or {}
        if best_notebook_blend and notebook_blend_cfg.get("enabled", False):
            notebook_blend_cfg.update(best_notebook_blend)
            config["postprocess"]["notebook_blend"] = notebook_blend_cfg
        best_smoothing = cv_metrics.get("best_smoothing")
        smoothing_cfg = config["postprocess"].get("smoothing") or {}
        if best_smoothing and smoothing_cfg.get("enabled", False):
            smoothing_cfg.update(best_smoothing)
            config["postprocess"]["smoothing"] = smoothing_cfg

    metrics: dict[str, Any] = {}
    metrics["run"] = {
        "run_id": run_id,
        "config": str(args.config),
        "git_hash": git_hash(Path(".")),
        "notes": args.notes,
        "public_lb": args.public_lb,
    }
    model = EnsembleRegressor(config, seed)
    fold_safe = bool(config["validation"].get("fold_safe_context", True))
    if fold_safe:
        with logger.step("Train fold-safe OOF ensemble", train_wells=len(train_paths)):
            X_oof, residual_oof, groups_oof, flat_oof, y_true_oof = train_fold_safe_oof(
                model, train_paths, config, seed, logger
            )
    else:
        full_context_for_oof = build_top_context(
            train_paths, config, logger, "Build Kaggle top-solution spatial context"
        )
        with logger.step("Build training table", train_wells=len(train_paths)):
            X_oof, residual_oof, groups_oof, flat_oof, y_true_oof = (
                build_training_table(train_paths, config, full_context_for_oof, logger)
            )
        with logger.step(
            "Train legacy OOF ensemble", rows=len(X_oof), features=len(X_oof.columns)
        ):
            model.fit(X_oof, residual_oof, groups=groups_oof, logger=logger)

    metrics["model"] = model.metrics_
    if model.oof_residual_ is None:
        raise RuntimeError("Ensemble did not produce OOF residual predictions.")
    baseline_name = config["features"].get("prediction_baseline", "flat_tvt")
    logger.metric(
        "OOF table",
        rows=len(X_oof),
        features=len(X_oof.columns),
        baseline=baseline_name,
        baseline_rmse=rmse(flat_oof, y_true_oof),
    )
    with logger.step("Tune OOF postprocess", rows=len(X_oof)):
        metrics["cv"] = evaluate_oof_predictions(
            model.oof_residual_, X_oof, groups_oof, flat_oof, y_true_oof, config, logger
        )
    apply_cv_selection(metrics["cv"])

    final_context = build_top_context(
        train_paths,
        config,
        logger,
        "Build final full-train Kaggle top context",
    )
    with logger.step("Build final training table", train_wells=len(train_paths)):
        X_full, residual_full, _groups_full, _flat_full, _y_true_full = (
            build_training_table(train_paths, config, final_context, logger)
        )
    with logger.step("Train final full-context models", rows=len(X_full)):
        model.fit_final(X_full, residual_full, logger)
    metrics["model"] = model.metrics_

    feature_names = list(X_full.columns)
    metrics["features"] = {
        "count": int(len(feature_names)),
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "context_mode": "fold_safe" if fold_safe else "global",
        "final_model_strategy": config["validation"].get(
            "final_model_strategy", "full_context"
        ),
        "final_context_key": getattr(final_context, "context_key", None),
    }
    metrics["train"] = {
        "rows": int(len(X_oof)),
        "wells": int(len(set(groups_oof))),
        "baseline": baseline_name,
        "baseline_rmse": rmse(flat_oof, y_true_oof),
        "flat_rmse": rmse(flat_oof, y_true_oof),
        "final_rows": int(len(X_full)),
    }
    if config.get("reporting", {}).get("compute_train_metrics", True):
        with logger.step("Evaluate OOF train prediction", rows=len(X_oof)):
            train_pred = apply_postprocess(
                flat_oof,
                model.oof_residual_,
                config,
                features=X_oof,
                groups=groups_oof,
            )
        metrics["train"]["rmse"] = rmse(train_pred, y_true_oof)
        metrics["diagnostics"] = regression_diagnostics(
            train_pred, y_true_oof, groups_oof, X_oof
        )
        logger.metric(
            "Final train summary",
            rmse=metrics["train"]["rmse"],
            baseline=baseline_name,
            baseline_rmse=metrics["train"]["baseline_rmse"],
            wells=metrics["train"]["wells"],
            mean_well_rmse=metrics["diagnostics"].get("mean_well_rmse"),
            p90_well_rmse=metrics["diagnostics"].get("p90_well_rmse"),
        )
    else:
        metrics["train"]["rmse"] = None
        metrics["train"]["skipped_model_train_rmse"] = True
        logger.info(
            "Skipped final train RMSE", reason="reporting.compute_train_metrics=false"
        )

    with logger.step("Predict test", test_wells=len(test_paths)):
        submission = predict_test(
            model,
            test_paths,
            sample_submission_path,
            config,
            feature_names,
            final_context,
            logger,
        )
    if submission["tvt"].isna().any():
        raise ValueError("Submission contains NaN predictions.")
    submission_path = Path(config["outputs"]["submission_path"])
    with logger.step("Write submission", path=submission_path, rows=len(submission)):
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)

    with logger.step("Save artifacts", output_dir=config["outputs"]["output_dir"]):
        save_outputs(model, feature_names, config, metrics, args.config)
    registry_path = Path(
        config.get("runs", {}).get("registry_path", "artifacts/runs.csv")
    )
    with logger.step("Append run registry", path=registry_path, run_id=run_id):
        append_run_registry(
            registry_path,
            metrics,
            args.config,
            run_id,
            args.public_lb,
            perf_counter() - run_started_at,
            args.notes,
        )
    logger.log("DONE", "Training run complete", total_duration=logger.elapsed())


if __name__ == "__main__":
    main()
