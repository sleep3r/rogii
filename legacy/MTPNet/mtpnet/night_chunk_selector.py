"""Post-night chunk/state selector over the measured path bank."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class NightChunkSelectorConfig:
    path_bank_path: Path = Path("artifacts/night/path_bank_oof.parquet")
    candidate_summary_path: Path = Path("artifacts/night/path_bank_candidate_summary.csv")
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")
    chunk_size: int = 512
    n_folds: int = 5
    max_candidates: int = 16
    min_coverage_frac: float = 0.999
    exclude_regex: str = r"oracle|truth|target|shuffled|zero"
    beta_endpoint: float = 0.3
    dp_switch_penalty: float = 0.02
    iterations: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 31
    seed: int = 42


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    return float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))) if mask.any() else float("nan")


def _well_metrics(well_ids: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    work = pd.DataFrame({"well_id": well_ids, "sqerr": (pred - y) ** 2})
    rmse = np.sqrt(work.groupby("well_id")["sqerr"].mean())
    return {
        "mean_well_rmse": float(rmse.mean()),
        "p50_well_rmse": float(rmse.quantile(0.50)),
        "p90_well_rmse": float(rmse.quantile(0.90)),
        "p95_well_rmse": float(rmse.quantile(0.95)),
        "p99_well_rmse": float(rmse.quantile(0.99)),
        "worst_well_rmse": float(rmse.max()),
    }


def _choose_columns(summary: pd.DataFrame, config: NightChunkSelectorConfig) -> tuple[list[str], dict[str, float], dict[str, int]]:
    work = summary.copy()
    if "coverage_frac" in work:
        work = work[work["coverage_frac"] >= config.min_coverage_frac]
    if config.exclude_regex:
        pattern = re.compile(config.exclude_regex, re.IGNORECASE)
        mask = ~work["path_col"].astype(str).map(lambda x: bool(pattern.search(x)))
        if "candidate" in work:
            mask &= ~work["candidate"].astype(str).map(lambda x: bool(pattern.search(x)))
        work = work[mask]
    work = work.sort_values("pooled_rmse", na_position="last").head(config.max_candidates)
    cols = [str(v) for v in work["path_col"].tolist()]
    rmse = {str(row.path_col): float(row.pooled_rmse) for row in work.itertuples(index=False)}
    rank = {col: i for i, col in enumerate(cols)}
    return cols, rmse, rank


def _safe_col(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return frame[name].to_numpy(dtype=np.float64)


def build_chunk_candidate_table(
    frame: pd.DataFrame,
    candidate_cols: list[str],
    *,
    chunk_size: int,
    beta_endpoint: float,
    candidate_global_rmse: dict[str, float] | None = None,
    candidate_rank: dict[str, int] | None = None,
) -> pd.DataFrame:
    if not candidate_cols:
        raise ValueError("candidate_cols is empty")
    work = frame[["well_id", "row_idx", "true_tvt"]].copy()
    work["_pos"] = np.arange(len(work), dtype=np.int64)
    work["chunk_id"] = (work.groupby("well_id", sort=False).cumcount() // int(chunk_size)).astype(np.int64)
    x = frame[candidate_cols].to_numpy(dtype=np.float64)
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    bank_median = np.nanmedian(x, axis=1)
    anchor = x[:, 0]
    md = _safe_col(frame, "MD")
    z = _safe_col(frame, "Z")
    gr = _safe_col(frame, "GR")
    rows: list[dict[str, Any]] = []
    for (well_id, chunk_id), idx in work.groupby(["well_id", "chunk_id"], sort=False).indices.items():
        loc = np.asarray(idx, dtype=np.int64)
        true = y[loc]
        med = bank_median[loc]
        anc = anchor[loc]
        chunk_rows = len(loc)
        rows_left = int((work["well_id"].to_numpy() == well_id).sum() - (int(chunk_id) + 1) * int(chunk_size))
        rows_left = max(rows_left, 0)
        z_chunk = z[loc]
        md_chunk = md[loc]
        gr_chunk = gr[loc]
        chunk_context = {
            "feat_chunk_rows": float(chunk_rows),
            "feat_chunk_rows_log": float(np.log1p(chunk_rows)),
            "feat_rows_left_log": float(np.log1p(rows_left)),
            "feat_row_start": float(frame.iloc[loc[0]]["row_idx"]),
            "feat_row_end": float(frame.iloc[loc[-1]]["row_idx"]),
            "feat_md_span": float(np.nanmax(md_chunk) - np.nanmin(md_chunk)) if np.isfinite(md_chunk).any() else np.nan,
            "feat_z_span": float(np.nanmax(z_chunk) - np.nanmin(z_chunk)) if np.isfinite(z_chunk).any() else np.nan,
            "feat_z_mean": float(np.nanmean(z_chunk)) if np.isfinite(z_chunk).any() else np.nan,
            "feat_z_std": float(np.nanstd(z_chunk)) if np.isfinite(z_chunk).any() else np.nan,
            "feat_gr_mean": float(np.nanmean(gr_chunk)) if np.isfinite(gr_chunk).any() else np.nan,
            "feat_gr_std": float(np.nanstd(gr_chunk)) if np.isfinite(gr_chunk).any() else np.nan,
            "feat_gr_valid_frac": float(np.isfinite(gr_chunk).mean()),
            "feat_bank_disagreement": float(np.nanmean(np.nanstd(x[loc], axis=1))),
        }
        for cand_id, col in enumerate(candidate_cols):
            pred = x[loc, cand_id]
            diff = pred - true
            finite = np.isfinite(diff)
            if finite.any():
                chunk_mse = float(np.mean(diff[finite] ** 2))
                endpoint_error = float(diff[finite][-1])
                start_error = float(diff[finite][0])
            else:
                chunk_mse = float("inf")
                endpoint_error = float("inf")
                start_error = float("inf")
            value_cost = chunk_mse + float(beta_endpoint) * endpoint_error**2
            pred_diff = np.diff(pred[np.isfinite(pred)])
            med_delta = pred - med
            anchor_delta = pred - anc
            row = {
                "well_id": well_id,
                "chunk_id": int(chunk_id),
                "candidate_id": int(cand_id),
                "candidate_col": col,
                "candidate_global_rank": int(candidate_rank.get(col, cand_id) if candidate_rank else cand_id),
                "candidate_global_rmse": float(candidate_global_rmse.get(col, np.nan) if candidate_global_rmse else np.nan),
                "chunk_mse": chunk_mse,
                "chunk_rmse": float(np.sqrt(chunk_mse)) if np.isfinite(chunk_mse) else np.inf,
                "endpoint_error": endpoint_error,
                "start_error": start_error,
                "value_cost": value_cost,
                "target_log_cost": float(np.log1p(max(value_cost, 0.0))) if np.isfinite(value_cost) else np.inf,
                "feat_pred_start": float(pred[0]) if len(pred) else np.nan,
                "feat_pred_end": float(pred[-1]) if len(pred) else np.nan,
                "feat_pred_mean": float(np.nanmean(pred)) if np.isfinite(pred).any() else np.nan,
                "feat_pred_std": float(np.nanstd(pred)) if np.isfinite(pred).any() else np.nan,
                "feat_pred_span": float(np.nanmax(pred) - np.nanmin(pred)) if np.isfinite(pred).any() else np.nan,
                "feat_pred_slope": float((pred[-1] - pred[0]) / max(chunk_rows - 1, 1)) if len(pred) else np.nan,
                "feat_pred_step_std": float(np.nanstd(pred_diff)) if len(pred_diff) else 0.0,
                "feat_bank_median_abs_mean": float(np.nanmean(np.abs(med_delta))),
                "feat_bank_median_signed_mean": float(np.nanmean(med_delta)),
                "feat_anchor_abs_mean": float(np.nanmean(np.abs(anchor_delta))),
                "feat_anchor_signed_mean": float(np.nanmean(anchor_delta)),
                **chunk_context,
            }
            rows.append(row)
    return pd.DataFrame(rows)


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
    }
    return [c for c in table.columns if c not in forbidden and pd.api.types.is_numeric_dtype(table[c])]


def _train_oof_costs(table: pd.DataFrame, config: NightChunkSelectorConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = _feature_columns(table)
    x = table[features].replace([np.inf, -np.inf], np.nan)
    y = table["target_log_cost"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    groups = table["well_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(config.n_folds, len(unique_groups))
    pred = np.full(len(table), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
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
            random_state=config.seed + fold,
            verbose=-1,
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        pred[valid_idx] = model.predict(x.iloc[valid_idx])
        fold_rows.append(
            {
                "fold": fold,
                "train_wells": int(len(np.unique(groups[train_idx]))),
                "valid_wells": int(len(np.unique(groups[valid_idx]))),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )
    return pred, fold_rows


def _select_predictions(frame: pd.DataFrame, table: pd.DataFrame, candidate_cols: list[str], score_col: str) -> tuple[np.ndarray, np.ndarray]:
    work = frame[["well_id", "row_idx", "true_tvt"]].copy()
    work["_pos"] = np.arange(len(work), dtype=np.int64)
    chunk_lookup = table.sort_values(score_col).groupby(["well_id", "chunk_id"], sort=False).first().reset_index()
    selected: dict[tuple[str, int], int] = {
        (str(row.well_id), int(row.chunk_id)): int(row.candidate_id) for row in chunk_lookup.itertuples(index=False)
    }
    work["chunk_id"] = (work.groupby("well_id", sort=False).cumcount() // int(table.attrs.get("chunk_size", 512))).astype(np.int64)
    x = frame[candidate_cols].to_numpy(dtype=np.float64)
    pred = np.full(len(frame), np.nan, dtype=np.float64)
    chosen = np.full(len(frame), -1, dtype=np.int32)
    for (well_id, chunk_id), idx in work.groupby(["well_id", "chunk_id"], sort=False).indices.items():
        cand = selected.get((str(well_id), int(chunk_id)))
        if cand is None:
            continue
        loc = np.asarray(idx, dtype=np.int64)
        pred[loc] = x[loc, cand]
        chosen[loc] = cand
    return pred, chosen


def _path_metrics(frame: pd.DataFrame, pred: np.ndarray, chosen: np.ndarray) -> dict[str, Any]:
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    switches = 0
    for _, idx in frame.groupby("well_id", sort=False).indices.items():
        loc = np.asarray(idx, dtype=np.int64)
        seq = chosen[loc]
        seq = seq[seq >= 0]
        switches += int(np.sum(seq[1:] != seq[:-1])) if len(seq) > 1 else 0
    return {
        "row_rmse": _rmse(y, pred),
        "switches": int(switches),
        **_well_metrics(frame["well_id"].to_numpy(), y, pred),
    }


def _write_report(output_dir: Path, metrics: dict[str, Any], fold_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Chunk Selector v1",
        "",
        f"Chunk size: `{metrics['chunk_size']}`",
        f"Candidates: `{metrics['candidate_count']}`",
        f"Selector row RMSE: `{metrics['selector_row_rmse']:.4f}`",
        f"Oracle chunk RMSE: `{metrics['chunk_oracle_rmse']:.4f}`",
        "",
        "## Fold Rows",
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
            "- This is the first clean chunk/state selector over the measured path bank.",
            "- If it remains close to best single rather than chunk oracle, feature/scorer design is the bottleneck.",
            "",
        ]
    )
    (output_dir / "chunk_selector_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission(path: Path, metrics: dict[str, Any]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    section = (
        "\n## Post-Night Task C. Chunk Selector v1\n\n"
        "Artifacts:\n\n"
        "- [x] `artifacts/night/chunk_selector_table.parquet`\n"
        "- [x] `artifacts/night/chunk_selector_oof_predictions.parquet`\n"
        "- [x] `artifacts/night/chunk_selector_report.md`\n\n"
        "Verdict:\n\n"
        "```text\n"
        f"Chunk selector v1 RMSE: {metrics['selector_row_rmse']:.4f}; chunk oracle RMSE at same size: {metrics['chunk_oracle_rmse']:.4f}. "
        "This measures how much of the path-bank oracle current features/scorer can capture.\n"
        "```\n"
    )
    if "## Post-Night Task C. Chunk Selector v1" not in text:
        text = text.replace("## Final Decision Tree", section + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_chunk_selector(config: NightChunkSelectorConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(config.candidate_summary_path)
    candidate_cols, global_rmse, global_rank = _choose_columns(summary, config)
    if not candidate_cols:
        raise ValueError("No candidate columns selected")
    base_cols = ["well_id", "row_idx", "true_tvt", "MD", "Z", "GR"]
    available_cols = pq.ParquetFile(config.path_bank_path).schema_arrow.names
    read_cols = [c for c in base_cols if c in available_cols] + candidate_cols
    frame = pd.read_parquet(config.path_bank_path, columns=read_cols)
    table = build_chunk_candidate_table(
        frame,
        candidate_cols,
        chunk_size=config.chunk_size,
        beta_endpoint=config.beta_endpoint,
        candidate_global_rmse=global_rmse,
        candidate_rank=global_rank,
    )
    table.attrs["chunk_size"] = config.chunk_size
    pred_cost, fold_rows = _train_oof_costs(table, config)
    table["pred_log_cost"] = pred_cost
    table.to_parquet(output_dir / "chunk_selector_table.parquet", index=False)
    selector_pred, selector_chosen = _select_predictions(frame, table, candidate_cols, "pred_log_cost")
    oracle_pred, oracle_chosen = _select_predictions(frame, table, candidate_cols, "chunk_mse")
    selector_metrics = _path_metrics(frame, selector_pred, selector_chosen)
    oracle_metrics = _path_metrics(frame, oracle_pred, oracle_chosen)
    pred_frame = frame[["well_id", "row_idx", "true_tvt"]].copy()
    pred_frame["pred_tvt"] = selector_pred
    pred_frame["selected_candidate_id"] = selector_chosen
    pred_frame["candidate"] = "chunk_selector_v1"
    pred_frame.to_parquet(output_dir / "chunk_selector_oof_predictions.parquet", index=False)
    metrics = {
        "task": "night_chunk_selector",
        "candidate_count": int(len(candidate_cols)),
        "chunk_size": int(config.chunk_size),
        "selector_row_rmse": float(selector_metrics["row_rmse"]),
        "selector_mean_well_rmse": float(selector_metrics["mean_well_rmse"]),
        "selector_p90_well_rmse": float(selector_metrics["p90_well_rmse"]),
        "selector_p99_well_rmse": float(selector_metrics["p99_well_rmse"]),
        "selector_worst_well_rmse": float(selector_metrics["worst_well_rmse"]),
        "selector_switches": int(selector_metrics["switches"]),
        "chunk_oracle_rmse": float(oracle_metrics["row_rmse"]),
        "chunk_oracle_switches": int(oracle_metrics["switches"]),
        "folds": fold_rows,
        "candidate_cols": candidate_cols,
    }
    (output_dir / "chunk_selector_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _write_report(output_dir, metrics, fold_rows)
    _update_mission(config.mission_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a chunk/state selector over the night path bank")
    parser.add_argument("--path-bank-path", type=Path, default=NightChunkSelectorConfig.path_bank_path)
    parser.add_argument("--candidate-summary-path", type=Path, default=NightChunkSelectorConfig.candidate_summary_path)
    parser.add_argument("--output-dir", type=Path, default=NightChunkSelectorConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightChunkSelectorConfig.mission_path)
    parser.add_argument("--chunk-size", type=int, default=NightChunkSelectorConfig.chunk_size)
    parser.add_argument("--n-folds", type=int, default=NightChunkSelectorConfig.n_folds)
    parser.add_argument("--max-candidates", type=int, default=NightChunkSelectorConfig.max_candidates)
    parser.add_argument("--iterations", type=int, default=NightChunkSelectorConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=NightChunkSelectorConfig.learning_rate)
    args = parser.parse_args()
    metrics = run_chunk_selector(
        NightChunkSelectorConfig(
            path_bank_path=args.path_bank_path,
            candidate_summary_path=args.candidate_summary_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
            chunk_size=args.chunk_size,
            n_folds=args.n_folds,
            max_candidates=args.max_candidates,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
