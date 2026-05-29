from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .clearml_data import apply_data_clearml_overrides, prepare_clearml_data_if_needed
from .config import load_config
from .io import (
    horizontal_files,
    resolve_data_dir,
    resolve_sample_submission,
    resolve_test_dir,
    resolve_train_dir,
)
from .runlog import RunLogger
from .spatial import KaggleTopContext
from .submission import predict_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROGII inference from a trained model artifact."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path. Defaults to <model-dir>/config.yml if present.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/stack"),
        help="Directory containing model.pkl, features.json, and metrics.json.",
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
    parser.add_argument(
        "--b2-config",
        type=Path,
        default=None,
        help="Optional frozen B2 guarded correction config to apply after base inference.",
    )
    parser.add_argument(
        "--data-clearml-enabled", "--data_clearml_enabled", default=None
    )
    parser.add_argument(
        "--data-clearml-project", "--data_clearml_project", default=None
    )
    parser.add_argument("--data-clearml-name", "--data_clearml_name", default=None)
    parser.add_argument(
        "--data-clearml-version", "--data_clearml_version", default=None
    )
    parser.add_argument(
        "--data-clearml-id",
        "--data_clearml_dataset_id",
        dest="data_clearml_dataset_id",
        default=None,
    )
    parser.add_argument("--data-clearml-alias", "--data_clearml_alias", default=None)
    parser.add_argument(
        "--data-clearml-cache-dir", "--data_clearml_cache_dir", default=None
    )
    return parser.parse_args()


def choose_config_path(model_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    artifact_config = model_dir / "config.yml"
    if artifact_config.is_file():
        return artifact_config
    return Path("configs/stack.yml")


def find_model_artifact_dir(root: Path) -> Path | None:
    if (root / "model.pkl").is_file() and (root / "features.json").is_file():
        return root
    if not root.is_dir():
        return None
    candidates = [
        path.parent
        for path in sorted(root.rglob("model.pkl"))
        if (path.parent / "features.json").is_file()
    ]
    return candidates[0] if candidates else None


def resolve_model_dir(model_dir: Path) -> Path:
    resolved = find_model_artifact_dir(model_dir)
    if resolved is not None:
        return resolved

    # Kaggle sometimes displays the dataset as attached while mounting it under
    # a title-derived or nested path. Search all inputs before failing.
    if str(model_dir).startswith("/kaggle/input"):
        resolved = find_model_artifact_dir(Path("/kaggle/input"))
        if resolved is not None:
            return resolved

    return model_dir


def describe_tree(root: Path, limit: int = 80) -> list[str]:
    if not root.exists():
        return [f"{root} [missing]"]
    output: list[str] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        children = []
    for child in children:
        kind = "dir" if child.is_dir() else "file"
        output.append(f"{child} [{kind}]")
        if len(output) >= limit:
            output.append("... truncated ...")
            return output
    for path in sorted(root.rglob("*")):
        if path.parent == root:
            continue
        if len(output) >= limit:
            output.append("... truncated ...")
            break
        kind = "dir" if path.is_dir() else "file"
        output.append(f"{path} [{kind}]")
    return output


def has_model_artifacts(model_dir: Path) -> bool:
    if (model_dir / "model.pkl").is_file() and (model_dir / "features.json").is_file():
        return True
    return False


def load_artifact_config(path: Path) -> dict[str, Any]:
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text) or {}
        if "inherits" not in loaded:
            return loaded
    return load_config(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_bundle(model_dir: Path) -> tuple[Any, list[str], dict[str, Any]]:
    model_path = model_dir / "model.pkl"
    features_path = model_dir / "features.json"
    metrics_path = model_dir / "metrics.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature list not found: {features_path}")

    with model_path.open("rb") as file:
        model = pickle.load(file)
    feature_names = [str(item) for item in load_json(features_path)]
    metrics = load_json(metrics_path) if metrics_path.is_file() else {}
    return model, feature_names, metrics


def align_feature_flags_from_artifact(
    config: dict[str, Any],
    feature_names: list[str],
    logger: RunLogger,
) -> None:
    top_cfg = config.setdefault("features", {}).setdefault("kaggle_top", {})
    robust_markers = (
        "kg_signal_robust",
        "pf_ancc_conf",
        "beam_conf",
        "pf_beam_gap",
        "pf_beam_abs_gap",
        "pf_dtw_gap",
        "pf_dwt_gap",
    )
    needs_robust = any(
        any(str(name).startswith(marker) for marker in robust_markers)
        for name in feature_names
    )
    if needs_robust and not top_cfg.get("robust_expert_enabled", False):
        top_cfg["robust_expert_enabled"] = True
        logger.warn("Enabled robust expert features from artifact schema")

    needs_hmm = any(str(name).startswith("kg_hmm_") for name in feature_names)
    if needs_hmm and not top_cfg.get("hmm_enabled", False):
        top_cfg["hmm_enabled"] = True
        logger.warn("Enabled HMM path features from artifact schema")


def main() -> None:
    args = parse_args()
    logger = RunLogger()
    logger.log("RUN", "ROGII inference started", model_dir=args.model_dir)

    model_dir = resolve_model_dir(args.model_dir)
    if model_dir != args.model_dir:
        logger.info("Resolved nested model artifact dir", model_dir=model_dir)
    if not (model_dir / "model.pkl").is_file():
        logger.warn(
            "Model artifact files not found", requested_model_dir=args.model_dir
        )
        for line in describe_tree(Path("/kaggle/input")):
            logger.warn("Kaggle input tree", path=line)
        for line in describe_tree(args.model_dir):
            logger.warn("Requested model dir tree", path=line)
        raise FileNotFoundError(f"Model artifact not found under {args.model_dir}")
    config_path = choose_config_path(model_dir, args.config)
    config = load_artifact_config(config_path)

    if args.data_dir is not None:
        config.setdefault("data", {})
        config["data"]["data_dir"] = str(args.data_dir)
        config["data"].setdefault("clearml", {})["enabled"] = False
    if args.output_dir is not None:
        config["outputs"]["output_dir"] = str(args.output_dir)
    if args.submission is not None:
        config["outputs"]["submission_path"] = str(args.submission)
    apply_data_clearml_overrides(config, args)
    prepare_clearml_data_if_needed(config, logger)

    with logger.step("Load model artifact", path=model_dir):
        model, feature_names, metrics = load_model_bundle(model_dir)
    align_feature_flags_from_artifact(config, feature_names, logger)
    logger.info(
        "Model artifact ready",
        features=len(feature_names),
        trained_rows=metrics.get("train", {}).get("rows"),
        cv_rmse=metrics.get("cv", {}).get("rmse"),
        residual_weight=config["postprocess"].get("residual_weight"),
    )

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

    output_dir = Path(config["outputs"]["output_dir"])
    b2_metrics: dict[str, Any] | None = None
    if args.b2_config is not None:
        from .formation_b2_inference import run_test_inference

        base_submission_path = output_dir / "base_submission.csv"
        b2_output_dir = output_dir / "b2_guarded"
        with logger.step("Write base submission before B2", path=base_submission_path):
            base_submission_path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(base_submission_path, index=False)
        with logger.step("Apply B2 guarded correction", config=args.b2_config):
            b2_metrics = run_test_inference(
                argparse.Namespace(
                    output_dir=b2_output_dir,
                    config=args.b2_config,
                    data_dir=data_dir,
                    train_dir=train_dir,
                    test_dir=test_dir,
                    base_submission=base_submission_path,
                    base_column="tvt",
                    progress_interval=1,
                )
            )
            submission = pd.read_csv(b2_output_dir / "submission.csv")
        if submission["tvt"].isna().any():
            raise ValueError("B2 submission contains NaN predictions.")

    submission_path = Path(config["outputs"]["submission_path"])
    with logger.step("Write submission", path=submission_path, rows=len(submission)):
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)

    with logger.step("Save inference metadata", output_dir=output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "inference.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "model_dir": str(model_dir),
                    "config": str(config_path),
                    "features": len(feature_names),
                    "submission_rows": int(len(submission)),
                    "source_cv_rmse": metrics.get("cv", {}).get("rmse"),
                    "residual_weight": config["postprocess"].get("residual_weight"),
                    "b2_config": str(args.b2_config) if args.b2_config else None,
                    "b2_metrics": b2_metrics,
                },
                file,
                indent=2,
            )
    logger.log("DONE", "Inference run complete", total_duration=logger.elapsed())


if __name__ == "__main__":
    main()
