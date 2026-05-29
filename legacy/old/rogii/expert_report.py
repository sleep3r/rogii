from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clearml_data import apply_data_clearml_overrides, prepare_clearml_data_if_needed
from .config import load_config
from .diagnostics import regression_diagnostics
from .features import build_training_table
from .io import horizontal_files, resolve_data_dir, resolve_train_dir
from .pipeline import build_top_context, shuffled_path_folds
from .runlog import RunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank TVT candidate-path experts without training GBM models."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stack.yml"))
    parser.add_argument("--data-dir", "--data_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/expert_report.md"))
    parser.add_argument("--csv", type=Path, default=Path("artifacts/expert_report.csv"))
    parser.add_argument("--json", type=Path, default=Path("artifacts/expert_report.json"))
    parser.add_argument("--top-n", "--top_n", type=int, default=40)
    parser.add_argument("--max-wells", "--max_wells", type=int, default=None)
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Evaluate candidates with one full-train context instead of fold-safe contexts.",
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


def candidate_predictions(
    X: pd.DataFrame,
    flat: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {
        "baseline": np.asarray(flat, dtype=float),
    }

    absolute_names = {
        "flat_tvt",
        "last_known_tvt",
        "anchor_tvt",
        "prev_known_tvt",
        "typewell_nearest_tvt_by_gr",
        "pf_ancc",
        "pf_z",
    }
    excluded_suffixes = (
        "_min",
        "_max",
        "_range",
        "_std",
        "_score",
        "_dist",
        "_velocity",
    )
    for column in X.columns:
        if column in absolute_names or (
            column.endswith("_tvt") and not column.endswith(excluded_suffixes)
        ):
            values = X[column].to_numpy(dtype=float)
            if np.isfinite(values).any():
                candidates[column] = values

    if "last_known_tvt" in X.columns:
        last = X["last_known_tvt"].to_numpy(dtype=float)
        for column in X.columns:
            if column.endswith("_minus_last") or column.endswith("_delta"):
                values = last + X[column].to_numpy(dtype=float)
                if np.isfinite(values).any():
                    candidates[f"{column}__as_tvt"] = values
        if "sig_mean_d" in X.columns:
            values = last + X["sig_mean_d"].to_numpy(dtype=float)
            if np.isfinite(values).any():
                candidates["sig_mean_d__as_tvt"] = values

    if "flat_tvt" in X.columns:
        flat_tvt = X["flat_tvt"].to_numpy(dtype=float)
        for column in X.columns:
            if column.endswith("_minus_flat"):
                values = flat_tvt + X[column].to_numpy(dtype=float)
                if np.isfinite(values).any():
                    candidates[f"{column}__as_tvt"] = values

    return candidates


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def score_candidate(
    name: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    X: pd.DataFrame,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(y_true)
    coverage = float(valid.mean()) if len(valid) else 0.0
    if not valid.any():
        return {
            "expert": name,
            "coverage": coverage,
            "valid_rows": 0,
            "global_rmse": None,
        }

    diagnostics = regression_diagnostics(
        pred[valid],
        y_true[valid],
        groups[valid],
        X.loc[valid].reset_index(drop=True),
    )
    error = pred[valid] - y_true[valid]
    row: dict[str, Any] = {
        "expert": name,
        "coverage": coverage,
        "valid_rows": int(valid.sum()),
        "bias": finite_float(np.nanmean(error)),
        "mae": finite_float(np.nanmean(np.abs(error))),
    }
    row.update(diagnostics)
    return row


def build_fold_safe_table(
    train_paths: list[Path],
    config: dict[str, Any],
    seed: int,
    logger: RunLogger,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_splits = int(config["validation"].get("n_splits", 5))
    folds = shuffled_path_folds(train_paths, n_splits, seed)
    X_parts: list[pd.DataFrame] = []
    residual_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    flat_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    for fold_id, fold_train_paths, fold_valid_paths in folds:
        context = build_top_context(
            fold_train_paths,
            config,
            logger,
            f"Build expert fold {fold_id} context",
        )
        with logger.step(
            "Build expert fold valid table",
            fold=fold_id,
            valid_wells=len(fold_valid_paths),
        ):
            X, residual, groups, flat, y_true = build_training_table(
                fold_valid_paths,
                config,
                context,
                logger,
            )
        X_parts.append(X)
        residual_parts.append(residual)
        group_parts.append(groups)
        flat_parts.append(flat)
        true_parts.append(y_true)
    return (
        pd.concat(X_parts, axis=0, ignore_index=True),
        np.concatenate(residual_parts),
        np.concatenate(group_parts),
        np.concatenate(flat_parts),
        np.concatenate(true_parts),
    )


def markdown_report(
    rows: list[dict[str, Any]],
    config_path: Path,
    fold_safe: bool,
    top_n: int,
) -> str:
    frame = pd.DataFrame(rows).sort_values("global_rmse", na_position="last")
    columns = [
        "expert",
        "global_rmse",
        "mean_well_rmse",
        "p90_well_rmse",
        "worst_well_rmse",
        "coverage",
        "valid_rows",
        "no_typewell_rmse",
        "long_hidden_rmse",
    ]
    visible = frame[[column for column in columns if column in frame.columns]].head(
        top_n
    )
    lines = [
        "# ROGII Candidate Expert Report",
        "",
        f"- Config: `{config_path}`",
        f"- Context: `{'fold-safe' if fold_safe else 'full-train'}`",
        f"- Experts scored: `{len(frame)}`",
        "",
        "Lower is better. `coverage` is the fraction of hidden training rows where "
        "the expert produced a finite TVT prediction.",
        "",
        dataframe_to_markdown(visible),
        "",
    ]
    return "\n".join(lines)


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not np.isfinite(value) else f"{value:.5f}"
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        [markdown_cell(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logger = RunLogger()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = str(args.data_dir)
    if args.max_wells is not None:
        config["data"]["max_train_wells"] = int(args.max_wells)
    apply_data_clearml_overrides(config, args)
    prepare_clearml_data_if_needed(config, logger)
    seed = int(config.get("seed", 42))

    data_dir = resolve_data_dir(config)
    train_dir = resolve_train_dir(data_dir, config)
    train_paths = horizontal_files(train_dir, config["data"].get("max_train_wells"))
    logger.info(
        "Expert report started",
        config=args.config,
        train_wells=len(train_paths),
        context="full" if args.full_context else "fold-safe",
    )

    if args.full_context:
        context = build_top_context(
            train_paths,
            config,
            logger,
            "Build expert full-train context",
        )
        with logger.step("Build expert full training table", train_wells=len(train_paths)):
            X, _residual, groups, flat, y_true = build_training_table(
                train_paths,
                config,
                context,
                logger,
            )
    else:
        with logger.step("Build expert fold-safe table", train_wells=len(train_paths)):
            X, _residual, groups, flat, y_true = build_fold_safe_table(
                train_paths,
                config,
                seed,
                logger,
            )

    candidates = candidate_predictions(X, flat)
    logger.info("Scoring candidate experts", candidates=len(candidates), rows=len(X))
    rows = [
        score_candidate(name, pred, y_true, groups, X)
        for name, pred in sorted(candidates.items())
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            float("inf") if row.get("global_rmse") is None else row["global_rmse"],
            -float(row.get("coverage") or 0.0),
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        markdown_report(rows, args.config, not args.full_context, args.top_n),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    args.json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.metric(
        "Best expert",
        expert=rows[0].get("expert") if rows else None,
        rmse=rows[0].get("global_rmse") if rows else None,
        coverage=rows[0].get("coverage") if rows else None,
    )
    logger.info("Wrote expert report", output=args.output, csv=args.csv, json=args.json)


if __name__ == "__main__":
    main()
