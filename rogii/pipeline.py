from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config
from .features import build_training_table
from .io import (
    horizontal_files,
    resolve_data_dir,
    resolve_sample_submission,
    resolve_test_dir,
    resolve_train_dir,
)
from .modeling import (
    EnsembleRegressor,
    apply_postprocess,
    evaluate_oof_predictions,
    rmse,
)
from .runlog import RunLogger
from .spatial import KaggleTopContext
from .submission import predict_test, save_outputs


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = RunLogger()
    logger.log("RUN", "ROGII training started", config=args.config)

    config = load_config(args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = str(args.data_dir)
    if args.output_dir is not None:
        config["outputs"]["output_dir"] = str(args.output_dir)
    if args.submission is not None:
        config["outputs"]["submission_path"] = str(args.submission)
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

    top_context = None
    if config["features"].get("include_kaggle_top_signals", False):
        with logger.step(
            "Build Kaggle top-solution spatial context", train_wells=len(train_paths)
        ):
            top_context = KaggleTopContext(train_paths, config)
        logger.info(
            "Spatial context ready",
            formation_wells=len(getattr(top_context, "formation_values", [])),
            dense_ancc_points=len(getattr(top_context, "dense_ancc", [])),
        )

    with logger.step("Build training table", train_wells=len(train_paths)):
        X, residual, groups, flat, y_true = build_training_table(
            train_paths, config, top_context, logger
        )
    baseline_name = config["features"].get("prediction_baseline", "flat_tvt")
    logger.metric(
        "Training table",
        rows=len(X),
        features=len(X.columns),
        baseline=baseline_name,
        baseline_train_rmse=rmse(flat, y_true),
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
    model = EnsembleRegressor(config, seed)
    with logger.step("Train OOF ensemble", rows=len(X), features=len(X.columns)):
        model.fit(X, residual, groups=groups, logger=logger)
    metrics["model"] = model.metrics_
    if model.oof_residual_ is None:
        raise RuntimeError("Ensemble did not produce OOF residual predictions.")
    with logger.step("Tune OOF postprocess", rows=len(X)):
        metrics["cv"] = evaluate_oof_predictions(
            model.oof_residual_, X, groups, flat, y_true, config, logger
        )
    apply_cv_selection(metrics["cv"])

    metrics["train"] = {
        "rows": int(len(X)),
        "wells": int(len(set(groups))),
        "baseline": baseline_name,
        "baseline_rmse": rmse(flat, y_true),
        "flat_rmse": rmse(flat, y_true),
    }
    if config.get("reporting", {}).get("compute_train_metrics", True):
        with logger.step("Evaluate OOF train prediction", rows=len(X)):
            train_pred = apply_postprocess(
                flat,
                model.oof_residual_,
                config,
                features=X,
                groups=groups,
            )
        metrics["train"]["rmse"] = rmse(train_pred, y_true)
        logger.metric(
            "Final train summary",
            rmse=metrics["train"]["rmse"],
            baseline=baseline_name,
            baseline_rmse=metrics["train"]["baseline_rmse"],
            wells=metrics["train"]["wells"],
        )
    else:
        metrics["train"]["rmse"] = None
        metrics["train"]["skipped_model_train_rmse"] = True
        logger.info(
            "Skipped final train RMSE", reason="reporting.compute_train_metrics=false"
        )

    feature_names = list(X.columns)
    with logger.step("Predict test", test_wells=len(test_paths)):
        submission = predict_test(
            model,
            test_paths,
            sample_submission_path,
            config,
            feature_names,
            top_context,
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
    logger.log("DONE", "Training run complete", total_duration=logger.elapsed())


if __name__ == "__main__":
    main()
