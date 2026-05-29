"""Post-night sanity models over chunk candidate features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from mtpnet.night_chunk_selector import _json_safe, _path_metrics


@dataclass(frozen=True)
class NightChunkSanityConfig:
    chunk_table_path: Path = Path("artifacts/night/chunk_selector_table.parquet")
    path_bank_path: Path = Path("artifacts/night/path_bank_oof.parquet")
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")
    chunk_size: int = 512
    n_folds: int = 5
    n_estimators: int = 300
    max_depth: int | None = 10
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


def _train_extratrees_oof(table: pd.DataFrame, config: NightChunkSanityConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = _feature_columns(table)
    x = table[features].replace([np.inf, -np.inf], np.nan)
    y = table["target_log_cost"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    groups = table["well_id"].astype(str).to_numpy()
    n_splits = min(config.n_folds, len(np.unique(groups)))
    pred = np.full(len(table), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups=groups)):
        train_idx = train_idx[valid[train_idx]]
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=2,
                random_state=config.seed + fold,
                n_jobs=-1,
            ),
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        pred[valid_idx] = model.predict(x.iloc[valid_idx])
        folds.append(
            {
                "fold": fold,
                "train_wells": int(len(np.unique(groups[train_idx]))),
                "valid_wells": int(len(np.unique(groups[valid_idx]))),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )
    return pred, folds


def _select_predictions(frame: pd.DataFrame, table: pd.DataFrame, candidate_cols: list[str], score_col: str, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    selected = table.sort_values(score_col).groupby(["well_id", "chunk_id"], sort=False).first().reset_index()
    lookup = {(str(row.well_id), int(row.chunk_id)): int(row.candidate_id) for row in selected.itertuples(index=False)}
    work = frame[["well_id", "row_idx"]].copy()
    work["chunk_id"] = (work.groupby("well_id", sort=False).cumcount() // int(chunk_size)).astype(np.int64)
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


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Chunk Sanity Models",
        "",
        f"ExtraTrees row RMSE: `{metrics['extratrees_row_rmse']:.4f}`",
        f"Chunk oracle RMSE: `{metrics['chunk_oracle_rmse']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- ExtraTrees is a non-boosted sanity check over the same chunk features.",
        "- If it also fails to approach oracle, the missing ingredient is feature signal/self-validation, not just LightGBM objective.",
        "",
    ]
    (output_dir / "chunk_sanity_report.md").write_text("\n".join(lines), encoding="utf-8")


def _update_mission(path: Path, metrics: dict[str, Any]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    section = (
        "\n## Post-Night Task F. Chunk Sanity Models\n\n"
        "Artifacts:\n\n"
        "- [x] `artifacts/night/chunk_sanity_grid.csv`\n"
        "- [x] `artifacts/night/chunk_sanity_extratrees_oof_predictions.parquet`\n"
        "- [x] `artifacts/night/chunk_sanity_report.md`\n\n"
        "Verdict:\n\n"
        "```text\n"
        f"ExtraTrees sanity RMSE: {metrics['extratrees_row_rmse']:.4f}; same-table chunk oracle: {metrics['chunk_oracle_rmse']:.4f}. "
        "If this is still near best-single, current chunk features lack selection signal.\n"
        "```\n"
    )
    if "## Post-Night Task F. Chunk Sanity Models" not in text:
        text = text.replace("## Final Decision Tree", section + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_chunk_sanity_models(config: NightChunkSanityConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(config.chunk_table_path)
    candidate_map = table[["candidate_id", "candidate_col"]].drop_duplicates().sort_values("candidate_id")
    candidate_cols = candidate_map["candidate_col"].astype(str).tolist()
    frame = pd.read_parquet(config.path_bank_path, columns=["well_id", "row_idx", "true_tvt"] + candidate_cols)
    pred_cost, folds = _train_extratrees_oof(table, config)
    table = table.copy()
    table["extratrees_pred_log_cost"] = pred_cost
    table.to_parquet(output_dir / "chunk_sanity_table.parquet", index=False)
    pred, chosen = _select_predictions(frame, table, candidate_cols, "extratrees_pred_log_cost", config.chunk_size)
    oracle_pred, oracle_chosen = _select_predictions(frame, table, candidate_cols, "chunk_mse", config.chunk_size)
    pred_frame = frame[["well_id", "row_idx", "true_tvt"]].copy()
    pred_frame["pred_tvt"] = pred
    pred_frame["selected_candidate_id"] = chosen
    pred_frame["candidate"] = "chunk_extratrees_v1"
    pred_frame.to_parquet(output_dir / "chunk_sanity_extratrees_oof_predictions.parquet", index=False)
    path_metrics = _path_metrics(frame, pred, chosen)
    oracle_metrics = _path_metrics(frame, oracle_pred, oracle_chosen)
    rows = [
        {"model": "extratrees", **path_metrics},
        {"model": "chunk_oracle", **oracle_metrics},
    ]
    grid = pd.DataFrame(rows)
    grid.to_csv(output_dir / "chunk_sanity_grid.csv", index=False)
    metrics = {
        "task": "night_chunk_sanity_models",
        "candidate_count": int(len(candidate_cols)),
        "extratrees_row_rmse": float(path_metrics["row_rmse"]),
        "extratrees_mean_well_rmse": float(path_metrics["mean_well_rmse"]),
        "extratrees_p90_well_rmse": float(path_metrics["p90_well_rmse"]),
        "extratrees_worst_well_rmse": float(path_metrics["worst_well_rmse"]),
        "chunk_oracle_rmse": float(oracle_metrics["row_rmse"]),
        "folds": folds,
    }
    (output_dir / "chunk_sanity_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _write_report(output_dir, metrics)
    _update_mission(config.mission_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sanity models over chunk candidate table")
    parser.add_argument("--chunk-table-path", type=Path, default=NightChunkSanityConfig.chunk_table_path)
    parser.add_argument("--path-bank-path", type=Path, default=NightChunkSanityConfig.path_bank_path)
    parser.add_argument("--output-dir", type=Path, default=NightChunkSanityConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightChunkSanityConfig.mission_path)
    parser.add_argument("--chunk-size", type=int, default=NightChunkSanityConfig.chunk_size)
    parser.add_argument("--n-folds", type=int, default=NightChunkSanityConfig.n_folds)
    parser.add_argument("--n-estimators", type=int, default=NightChunkSanityConfig.n_estimators)
    args = parser.parse_args()
    metrics = run_chunk_sanity_models(
        NightChunkSanityConfig(
            chunk_table_path=args.chunk_table_path,
            path_bank_path=args.path_bank_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
            chunk_size=args.chunk_size,
            n_folds=args.n_folds,
            n_estimators=args.n_estimators,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
