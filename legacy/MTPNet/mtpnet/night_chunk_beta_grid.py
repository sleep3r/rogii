"""Beta-grid chunk selector target audit."""

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

from mtpnet.night_chunk_selector import _feature_columns, _json_safe, _path_metrics


@dataclass(frozen=True)
class NightChunkBetaGridConfig:
    chunk_table_path: Path = Path("artifacts/night/chunk_selector_table.parquet")
    path_bank_path: Path = Path("artifacts/night/path_bank_oof.parquet")
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")
    chunk_size: int = 512
    betas: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0)
    n_folds: int = 5
    iterations: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 31
    seed: int = 42


def make_value_target(table: pd.DataFrame, beta: float) -> pd.Series:
    rows_left = np.expm1(table["feat_rows_left_log"].astype(float)) if "feat_rows_left_log" in table else 0.0
    value = table["chunk_mse"].astype(float) + float(beta) * rows_left * (table["endpoint_error"].astype(float) ** 2)
    value = value.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    return np.log1p(value)


def _train_oof(table: pd.DataFrame, y: np.ndarray, config: NightChunkBetaGridConfig, beta: float) -> np.ndarray:
    features = _feature_columns(table)
    x = table[features].replace([np.inf, -np.inf], np.nan)
    valid = np.isfinite(y)
    groups = table["well_id"].astype(str).to_numpy()
    n_splits = min(config.n_folds, len(np.unique(groups)))
    pred = np.full(len(table), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups=groups)):
        train_idx = train_idx[valid[train_idx]]
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=config.iterations,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            min_child_samples=1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.seed + fold + int(beta * 1000),
            verbose=-1,
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        pred[valid_idx] = model.predict(x.iloc[valid_idx])
    return pred


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


def _write_report(output_dir: Path, grid: pd.DataFrame) -> None:
    lines = [
        "# Chunk Value-Target Beta Grid",
        "",
        "| beta | row_rmse | mean_well | p90 | worst |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in grid.itertuples(index=False):
        lines.append(
            f"| {row.beta:.3f} | {row.row_rmse:.4f} | {row.mean_well_rmse:.4f} | "
            f"{row.p90_well_rmse:.4f} | {row.worst_well_rmse:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This reruns the selector with the intended `chunk_mse + beta * rows_left * endpoint_error^2` target.",
            "- If all betas remain near best-single, endpoint-aware value target is not enough without stronger features.",
            "",
        ]
    )
    (output_dir / "chunk_beta_grid_report.md").write_text("\n".join(lines), encoding="utf-8")


def _update_mission(path: Path, metrics: dict[str, Any]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    section = (
        "\n## Post-Night Task G. Chunk Value-Target Beta Grid\n\n"
        "Artifacts:\n\n"
        "- [x] `artifacts/night/chunk_beta_grid.csv`\n"
        "- [x] `artifacts/night/chunk_beta_best_oof_predictions.parquet`\n"
        "- [x] `artifacts/night/chunk_beta_grid_report.md`\n\n"
        "Verdict:\n\n"
        "```text\n"
        f"Best beta-grid RMSE: {metrics['best_row_rmse']:.4f} at beta={metrics['best_beta']}. "
        "This uses the intended rows-left endpoint-aware value target.\n"
        "```\n"
    )
    if "## Post-Night Task G. Chunk Value-Target Beta Grid" not in text:
        text = text.replace("## Final Decision Tree", section + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_chunk_beta_grid(config: NightChunkBetaGridConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_base = pd.read_parquet(config.chunk_table_path)
    candidate_map = table_base[["candidate_id", "candidate_col"]].drop_duplicates().sort_values("candidate_id")
    candidate_cols = candidate_map["candidate_col"].astype(str).tolist()
    frame = pd.read_parquet(config.path_bank_path, columns=["well_id", "row_idx", "true_tvt"] + candidate_cols)
    rows: list[dict[str, Any]] = []
    best_pred: np.ndarray | None = None
    best_chosen: np.ndarray | None = None
    best_rmse = float("inf")
    best_beta = float("nan")
    for beta in config.betas:
        table = table_base.copy()
        y = make_value_target(table, beta=float(beta)).to_numpy(dtype=np.float64)
        table[f"pred_beta_{beta:g}"] = _train_oof(table, y, config, float(beta))
        pred, chosen = _select_predictions(frame, table, candidate_cols, f"pred_beta_{beta:g}", config.chunk_size)
        path_metrics = _path_metrics(frame, pred, chosen)
        row = {"beta": float(beta), **path_metrics}
        rows.append(row)
        if float(path_metrics["row_rmse"]) < best_rmse:
            best_rmse = float(path_metrics["row_rmse"])
            best_beta = float(beta)
            best_pred = pred
            best_chosen = chosen
    grid = pd.DataFrame(rows).sort_values("row_rmse").reset_index(drop=True)
    grid.to_csv(output_dir / "chunk_beta_grid.csv", index=False)
    if best_pred is not None and best_chosen is not None:
        pred_frame = frame[["well_id", "row_idx", "true_tvt"]].copy()
        pred_frame["pred_tvt"] = best_pred
        pred_frame["selected_candidate_id"] = best_chosen
        pred_frame["candidate"] = f"chunk_beta_grid_best_b{best_beta:g}"
        pred_frame.to_parquet(output_dir / "chunk_beta_best_oof_predictions.parquet", index=False)
    _write_report(output_dir, grid)
    metrics = {
        "task": "night_chunk_beta_grid",
        "betas": [float(v) for v in config.betas],
        "best_beta": best_beta,
        "best_row_rmse": best_rmse,
    }
    (output_dir / "chunk_beta_grid_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _update_mission(config.mission_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run beta grid for chunk value target")
    parser.add_argument("--chunk-table-path", type=Path, default=NightChunkBetaGridConfig.chunk_table_path)
    parser.add_argument("--path-bank-path", type=Path, default=NightChunkBetaGridConfig.path_bank_path)
    parser.add_argument("--output-dir", type=Path, default=NightChunkBetaGridConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightChunkBetaGridConfig.mission_path)
    parser.add_argument("--chunk-size", type=int, default=NightChunkBetaGridConfig.chunk_size)
    parser.add_argument("--betas", type=str, default="0,0.1,0.3,1.0")
    parser.add_argument("--n-folds", type=int, default=NightChunkBetaGridConfig.n_folds)
    parser.add_argument("--iterations", type=int, default=NightChunkBetaGridConfig.iterations)
    args = parser.parse_args()
    metrics = run_chunk_beta_grid(
        NightChunkBetaGridConfig(
            chunk_table_path=args.chunk_table_path,
            path_bank_path=args.path_bank_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
            chunk_size=args.chunk_size,
            betas=tuple(float(v) for v in args.betas.split(",") if v.strip()),
            n_folds=args.n_folds,
            iterations=args.iterations,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
