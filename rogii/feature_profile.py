from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .features import build_training_table
from .io import horizontal_files, resolve_data_dir, resolve_train_dir, well_name
from .pipeline import build_top_context, shuffled_path_folds
from .runlog import RunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile ROGII feature preparation.")
    parser.add_argument("--config", type=Path, default=Path("configs/stack.yml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-wells", type=int, default=50)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument(
        "--context",
        choices=["fold-train", "fold-valid", "full", "none"],
        default="fold-train",
        help="Which context/table slice to profile.",
    )
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--no-stage-profile", action="store_true")
    return parser.parse_args()


def selected_paths(
    train_paths: list[Path],
    config: dict,
    context_mode: str,
    fold_id: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if context_mode == "none":
        return [], train_paths
    if context_mode == "full":
        return train_paths, train_paths

    folds = shuffled_path_folds(
        train_paths,
        int(config["validation"].get("n_splits", 5)),
        seed,
    )
    if fold_id < 1 or fold_id > len(folds):
        raise ValueError(f"fold-id must be in [1, {len(folds)}], got {fold_id}")
    _, fold_train_paths, fold_valid_paths = folds[fold_id - 1]
    if context_mode == "fold-valid":
        return fold_train_paths, fold_valid_paths
    return fold_train_paths, fold_train_paths


def main() -> None:
    args = parse_args()
    logger = RunLogger()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = str(args.data_dir)
    config["features"]["profile_stages"] = not args.no_stage_profile
    config["features"]["progress_interval"] = max(1, int(args.progress_interval))
    if args.disable_cache:
        config["features"].setdefault("cache", {})["enabled"] = False

    seed = int(config.get("seed", 42))
    data_dir = resolve_data_dir(config)
    train_dir = resolve_train_dir(data_dir, config)
    train_paths = horizontal_files(train_dir, config["data"].get("max_train_wells"))
    context_paths, table_paths = selected_paths(
        train_paths,
        config,
        args.context,
        args.fold_id,
        seed,
    )
    if args.max_wells > 0:
        table_paths = table_paths[: args.max_wells]
    if not table_paths:
        raise ValueError("No wells selected for feature profiling.")

    logger.log(
        "RUN",
        "ROGII feature profiling started",
        config=args.config,
        context=args.context,
        fold=args.fold_id,
        selected_wells=len(table_paths),
        cache_enabled=config["features"].get("cache", {}).get("enabled", False),
    )
    logger.info(
        "Feature profile paths",
        train_dir=train_dir,
        context_wells=len(context_paths),
        table_wells=len(table_paths),
        first_well=well_name(table_paths[0]),
    )

    context = None
    if args.context != "none":
        context = build_top_context(
            context_paths,
            config,
            logger,
            "Build profile Kaggle top context",
        )

    with logger.step("Profile feature table", wells=len(table_paths)):
        X, residual, groups, flat, y_true = build_training_table(
            table_paths,
            config,
            context,
            logger,
        )
    logger.metric(
        "Feature profile summary",
        rows=len(X),
        features=len(X.columns),
        wells=len(set(groups)),
        residual_rows=len(residual),
        flat_rows=len(flat),
        target_rows=len(y_true),
    )
    logger.log("DONE", "Feature profiling complete", total_duration=logger.elapsed())


if __name__ == "__main__":
    main()
