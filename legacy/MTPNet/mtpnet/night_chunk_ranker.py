"""Post-night chunk LambdaRank selector over a precomputed chunk candidate table."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from mtpnet.night_chunk_selector import _json_safe, _path_metrics


@dataclass(frozen=True)
class NightChunkRankerConfig:
    chunk_table_path: Path = Path("artifacts/night/chunk_selector_table.parquet")
    path_bank_path: Path = Path("artifacts/night/path_bank_oof.parquet")
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")
    n_folds: int = 5
    iterations: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 31
    seed: int = 42


def _feature_columns(table: pd.DataFrame) -> list[str]:
    forbidden = {
        "well_id",
        "chunk_id",
        "candidate_col",
        "chunk_mse",
        "chunk_rmse",
        "endpoint_error",
        "start_error",
        "value_cost",
        "target_log_cost",
        "pred_log_cost",
        "ranker_score",
    }
    return [c for c in table.columns if c not in forbidden and pd.api.types.is_numeric_dtype(table[c])]


def _rank_labels(table: pd.DataFrame) -> np.ndarray:
    rank = table.groupby(["well_id", "chunk_id"], sort=False)["chunk_mse"].rank(method="first", ascending=True)
    max_rank = table.groupby(["well_id", "chunk_id"], sort=False)["candidate_id"].transform("count")
    relevance = (max_rank - rank).astype(np.int32)
    return relevance.to_numpy(dtype=np.int32)


def _sort_by_group(x: pd.DataFrame, y: np.ndarray, group_ids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    order = np.argsort(group_ids, kind="stable")
    x_sorted = x.iloc[order]
    y_sorted = y[order]
    groups_sorted = group_ids[order]
    _, counts = np.unique(groups_sorted, return_counts=True)
    return x_sorted, y_sorted, counts.astype(int).tolist()


def _train_oof_ranker(table: pd.DataFrame, config: NightChunkRankerConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = _feature_columns(table)
    x = table[features].replace([np.inf, -np.inf], np.nan)
    y = _rank_labels(table)
    wells = table["well_id"].astype(str).to_numpy()
    state_ids = (table["well_id"].astype(str) + "::" + table["chunk_id"].astype(str)).to_numpy()
    unique_wells = np.unique(wells)
    n_splits = min(config.n_folds, len(unique_wells))
    pred = np.full(len(table), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups=wells)):
        train_x, train_y, train_group = _sort_by_group(x.iloc[train_idx], y[train_idx], state_ids[train_idx])
        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=config.iterations,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            min_child_samples=1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.seed + fold,
            verbose=-1,
        )
        model.fit(train_x, train_y, group=train_group)
        pred[valid_idx] = model.predict(x.iloc[valid_idx])
        fold_rows.append(
            {
                "fold": fold,
                "train_wells": int(len(np.unique(wells[train_idx]))),
                "valid_wells": int(len(np.unique(wells[valid_idx]))),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )
    return pred, fold_rows


def _select_predictions(frame: pd.DataFrame, table: pd.DataFrame, candidate_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    selected = table.sort_values("ranker_score", ascending=False).groupby(["well_id", "chunk_id"], sort=False).first().reset_index()
    lookup = {(str(row.well_id), int(row.chunk_id)): int(row.candidate_id) for row in selected.itertuples(index=False)}
    work = frame[["well_id", "row_idx"]].copy()
    work["chunk_id"] = (work.groupby("well_id", sort=False).cumcount() // _infer_chunk_size(table)).astype(np.int64)
    x = frame[candidate_cols].to_numpy(dtype=np.float64)
    pred = np.full(len(frame), np.nan, dtype=np.float64)
    chosen = np.full(len(frame), -1, dtype=np.int32)
    for (well_id, chunk_id), idx in work.groupby(["well_id", "chunk_id"], sort=False).indices.items():
        cand = lookup.get((str(well_id), int(chunk_id)))
        if cand is None:
            continue
        loc = np.asarray(idx, dtype=np.int64)
        pred[loc] = x[loc, cand]
        chosen[loc] = cand
    return pred, chosen


def _infer_chunk_size(table: pd.DataFrame) -> int:
    counts = table.groupby(["well_id", "chunk_id"], sort=False)["candidate_id"].count()
    candidates = int(table["candidate_id"].nunique())
    # This is only used to reconstruct row chunk ids; infer from first well/chunk table
    # is impossible directly, so use the mission default unless attrs exist.
    return int(table.attrs.get("chunk_size", 512)) if candidates else 512


def _write_report(output_dir: Path, metrics: dict[str, Any], fold_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Chunk LambdaRank Selector",
        "",
        f"Ranker row RMSE: `{metrics['ranker_row_rmse']:.4f}`",
        f"Candidate count: `{metrics['candidate_count']}`",
        "",
        "| fold | train_wells | valid_wells | train_rows | valid_rows |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows:
        lines.append(f"| {row['fold']} | {row['train_wells']} | {row['valid_wells']} | {row['train_rows']} | {row['valid_rows']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This tests within-chunk ranking directly via LambdaRank.",
            "- If this does not improve over cost regression, current features do not expose candidate quality.",
            "",
        ]
    )
    (output_dir / "chunk_ranker_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission(path: Path, metrics: dict[str, Any]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    section = (
        "\n## Post-Night Task D. Chunk LambdaRank Selector\n\n"
        "Artifacts:\n\n"
        "- [x] `artifacts/night/chunk_ranker_oof_predictions.parquet`\n"
        "- [x] `artifacts/night/chunk_ranker_report.md`\n"
        "- [x] `artifacts/night/chunk_ranker_metrics.json`\n\n"
        "Verdict:\n\n"
        "```text\n"
        f"LambdaRank selector RMSE: {metrics['ranker_row_rmse']:.4f}. "
        "This is the first direct ranking objective over chunk candidates.\n"
        "```\n"
    )
    if "## Post-Night Task D. Chunk LambdaRank Selector" not in text:
        text = text.replace("## Final Decision Tree", section + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_chunk_ranker(config: NightChunkRankerConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(config.chunk_table_path)
    candidate_map = table[["candidate_id", "candidate_col"]].drop_duplicates().sort_values("candidate_id")
    candidate_cols = candidate_map["candidate_col"].astype(str).tolist()
    frame = pd.read_parquet(config.path_bank_path, columns=["well_id", "row_idx", "true_tvt"] + candidate_cols)
    ranker_score, fold_rows = _train_oof_ranker(table, config)
    table = table.copy()
    table["ranker_score"] = ranker_score
    table.to_parquet(output_dir / "chunk_ranker_table.parquet", index=False)
    pred, chosen = _select_predictions(frame, table, candidate_cols)
    metrics_path = _path_metrics(frame, pred, chosen)
    pred_frame = frame[["well_id", "row_idx", "true_tvt"]].copy()
    pred_frame["pred_tvt"] = pred
    pred_frame["selected_candidate_id"] = chosen
    pred_frame["candidate"] = "chunk_lambdarank_v1"
    pred_frame.to_parquet(output_dir / "chunk_ranker_oof_predictions.parquet", index=False)
    metrics = {
        "task": "night_chunk_ranker",
        "candidate_count": int(len(candidate_cols)),
        "ranker_row_rmse": float(metrics_path["row_rmse"]),
        "ranker_mean_well_rmse": float(metrics_path["mean_well_rmse"]),
        "ranker_p90_well_rmse": float(metrics_path["p90_well_rmse"]),
        "ranker_p99_well_rmse": float(metrics_path["p99_well_rmse"]),
        "ranker_worst_well_rmse": float(metrics_path["worst_well_rmse"]),
        "ranker_switches": int(metrics_path["switches"]),
        "folds": fold_rows,
    }
    (output_dir / "chunk_ranker_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _write_report(output_dir, metrics, fold_rows)
    _update_mission(config.mission_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a chunk LambdaRank selector over chunk candidate table")
    parser.add_argument("--chunk-table-path", type=Path, default=NightChunkRankerConfig.chunk_table_path)
    parser.add_argument("--path-bank-path", type=Path, default=NightChunkRankerConfig.path_bank_path)
    parser.add_argument("--output-dir", type=Path, default=NightChunkRankerConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightChunkRankerConfig.mission_path)
    parser.add_argument("--n-folds", type=int, default=NightChunkRankerConfig.n_folds)
    parser.add_argument("--iterations", type=int, default=NightChunkRankerConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=NightChunkRankerConfig.learning_rate)
    args = parser.parse_args()
    metrics = run_chunk_ranker(
        NightChunkRankerConfig(
            chunk_table_path=args.chunk_table_path,
            path_bank_path=args.path_bank_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
            n_folds=args.n_folds,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
