"""Post-night selector oracle by granularity."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NightSelectorOracleConfig:
    path_bank_path: Path = Path("artifacts/night/path_bank_oof.parquet")
    candidate_summary_path: Path = Path("artifacts/night/path_bank_candidate_summary.csv")
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")
    chunk_sizes: tuple[int, ...] = (1024, 512, 256, 128)
    max_candidates: int = 24
    min_coverage_frac: float = 0.99
    dp_chunk_size: int = 512
    dp_switch_penalty: float = 500.0


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


def _well_quantiles(well_ids: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
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


def _choose_columns(summary: pd.DataFrame, max_candidates: int, min_coverage_frac: float) -> list[str]:
    work = summary.copy()
    if "coverage_frac" in work:
        work = work[work["coverage_frac"] >= min_coverage_frac]
    work = work.sort_values("pooled_rmse", na_position="last")
    return [str(v) for v in work["path_col"].head(max_candidates).tolist()]


def _candidate_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=np.float64)


def _build_prediction_by_groups(frame: pd.DataFrame, x: np.ndarray, columns: list[str], group_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    pred = np.full(len(frame), np.nan, dtype=np.float64)
    chosen = np.full(len(frame), -1, dtype=np.int32)
    for _, idx in frame.groupby(group_keys, sort=False).indices.items():
        loc = np.asarray(idx, dtype=np.int64)
        subx = x[loc]
        suby = y[loc, None]
        finite = np.isfinite(subx) & np.isfinite(suby)
        sq = np.where(finite, (subx - suby) ** 2, np.inf)
        sse = np.sum(sq, axis=0)
        if not np.isfinite(sse).any():
            continue
        best = int(np.argmin(sse))
        pred[loc] = subx[:, best]
        chosen[loc] = best
    return pred, chosen


def _row_oracle(frame: pd.DataFrame, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    sq = (x - y[:, None]) ** 2
    sq[~np.isfinite(sq)] = np.inf
    best = np.argmin(sq, axis=1)
    pred = x[np.arange(len(x)), best]
    return pred, best.astype(np.int32)


def _add_chunk_ids(frame: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    work = frame[["well_id", "row_idx", "true_tvt"]].copy()
    order = work.groupby("well_id", sort=False).cumcount()
    work["chunk_id"] = (order // int(chunk_size)).astype(np.int64)
    return work


def _dp_select_for_well(cost: np.ndarray, switch_penalty: float) -> np.ndarray:
    t_count, c_count = cost.shape
    dp = np.full((t_count, c_count), np.inf, dtype=np.float64)
    back = np.full((t_count, c_count), -1, dtype=np.int32)
    dp[0] = cost[0]
    penalty = np.full((c_count, c_count), float(switch_penalty), dtype=np.float64)
    np.fill_diagonal(penalty, 0.0)
    for t in range(1, t_count):
        prev = dp[t - 1][:, None] + penalty
        back[t] = np.argmin(prev, axis=0)
        dp[t] = cost[t] + np.min(prev, axis=0)
    path = np.zeros(t_count, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(t_count - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _dp_chunk_oracle(frame: pd.DataFrame, x: np.ndarray, chunk_size: int, switch_penalty: float) -> tuple[np.ndarray, np.ndarray]:
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    pred = np.full(len(frame), np.nan, dtype=np.float64)
    chosen = np.full(len(frame), -1, dtype=np.int32)
    work = _add_chunk_ids(frame, chunk_size)
    n_candidates = x.shape[1]
    for _, well_idx in work.groupby("well_id", sort=False).indices.items():
        well_loc = np.asarray(well_idx, dtype=np.int64)
        well_chunks = work.iloc[well_loc]["chunk_id"].to_numpy()
        chunk_values = np.unique(well_chunks)
        cost = np.full((len(chunk_values), n_candidates), np.inf, dtype=np.float64)
        chunk_locs: list[np.ndarray] = []
        for t, chunk in enumerate(chunk_values):
            loc = well_loc[well_chunks == chunk]
            chunk_locs.append(loc)
            subx = x[loc]
            suby = y[loc, None]
            finite = np.isfinite(subx) & np.isfinite(suby)
            cost[t] = np.sum(np.where(finite, (subx - suby) ** 2, np.inf), axis=0)
        path = _dp_select_for_well(cost, switch_penalty=switch_penalty)
        for loc, cand in zip(chunk_locs, path, strict=True):
            pred[loc] = x[loc, cand]
            chosen[loc] = cand
    return pred, chosen


def _row_metrics(frame: pd.DataFrame, pred: np.ndarray, chosen: np.ndarray, name: str, columns: list[str]) -> dict[str, Any]:
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    q = _well_quantiles(frame["well_id"].to_numpy(), y, pred)
    switches = 0
    for _, idx in frame.groupby("well_id", sort=False).indices.items():
        seq = chosen[np.asarray(idx, dtype=np.int64)]
        seq = seq[seq >= 0]
        switches += int(np.sum(seq[1:] != seq[:-1])) if len(seq) > 1 else 0
    return {
        "granularity": name,
        "row_rmse": _rmse(y, pred),
        "rows": int(np.isfinite(pred).sum()),
        "candidate_count": int(len(columns)),
        "switches": int(switches),
        **q,
    }


def _write_report(output_dir: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Selector Oracle Granularity",
        "",
        "| granularity | row_rmse | mean_well | p90 | p99 | worst | switches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.granularity} | {row.row_rmse:.4f} | {row.mean_well_rmse:.4f} | "
            f"{row.p90_well_rmse:.4f} | {row.p99_well_rmse:.4f} | {row.worst_well_rmse:.4f} | {int(row.switches)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Whole-well oracle answers whether a single candidate per well is enough.",
            "- Chunk oracles answer how local the selector must become.",
            "- DP chunk oracle is a smoother upper bound with a switch penalty.",
            "",
        ]
    )
    (output_dir / "selector_oracle_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission(path: Path, best_chunk: float, whole: float, row: float) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    section = (
        "\n## Post-Night Task B. Selector Oracle Granularity\n\n"
        "Artifacts:\n\n"
        "- [x] `artifacts/night/selector_oracle_granularity.csv`\n"
        "- [x] `artifacts/night/selector_oracle_report.md`\n\n"
        "Verdict:\n\n"
        "```text\n"
        f"Whole-well oracle: {whole:.4f}; best chunk oracle: {best_chunk:.4f}; row oracle: {row:.4f}. "
        "If chunk oracle is far below whole-well, train chunk/state selector rather than well-level selector.\n"
        "```\n"
    )
    if "## Post-Night Task B. Selector Oracle Granularity" not in text:
        text = text.replace("## Final Decision Tree", section + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_selector_oracle(config: NightSelectorOracleConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(config.candidate_summary_path)
    columns = _choose_columns(summary, config.max_candidates, config.min_coverage_frac)
    frame = pd.read_parquet(config.path_bank_path, columns=["well_id", "row_idx", "true_tvt"] + columns)
    x = _candidate_matrix(frame, columns)
    rows: list[dict[str, Any]] = []
    pred, chosen = _row_oracle(frame, x)
    rows.append(_row_metrics(frame, pred, chosen, "row_oracle", columns))
    pred, chosen = _build_prediction_by_groups(frame, x, columns, ["well_id"])
    rows.append(_row_metrics(frame, pred, chosen, "whole_well_oracle", columns))
    chunk_scores: list[float] = []
    for size in config.chunk_sizes:
        chunk_frame = _add_chunk_ids(frame, size)
        pred, chosen = _build_prediction_by_groups(chunk_frame, x, columns, ["well_id", "chunk_id"])
        metrics = _row_metrics(frame, pred, chosen, f"chunk_{size}_oracle", columns)
        rows.append(metrics)
        chunk_scores.append(float(metrics["row_rmse"]))
    pred, chosen = _dp_chunk_oracle(frame, x, config.dp_chunk_size, config.dp_switch_penalty)
    rows.append(_row_metrics(frame, pred, chosen, f"dp_chunk_{config.dp_chunk_size}_switch{config.dp_switch_penalty:g}", columns))
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "selector_oracle_granularity.csv", index=False)
    _write_report(output_dir, table)
    row_rmse = float(table.loc[table["granularity"] == "row_oracle", "row_rmse"].iloc[0])
    whole_rmse = float(table.loc[table["granularity"] == "whole_well_oracle", "row_rmse"].iloc[0])
    best_chunk = float(np.nanmin(chunk_scores)) if chunk_scores else float("nan")
    metrics = {
        "task": "night_selector_oracle",
        "candidate_count": int(len(columns)),
        "row_oracle_rmse": row_rmse,
        "whole_well_oracle_rmse": whole_rmse,
        "best_chunk_oracle_rmse": best_chunk,
    }
    (output_dir / "selector_oracle_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _update_mission(config.mission_path, best_chunk=best_chunk, whole=whole_rmse, row=row_rmse)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute path-bank selector oracle by granularity")
    parser.add_argument("--path-bank-path", type=Path, default=NightSelectorOracleConfig.path_bank_path)
    parser.add_argument("--candidate-summary-path", type=Path, default=NightSelectorOracleConfig.candidate_summary_path)
    parser.add_argument("--output-dir", type=Path, default=NightSelectorOracleConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightSelectorOracleConfig.mission_path)
    parser.add_argument("--chunk-sizes", type=str, default="1024,512,256,128")
    parser.add_argument("--max-candidates", type=int, default=NightSelectorOracleConfig.max_candidates)
    parser.add_argument("--min-coverage-frac", type=float, default=NightSelectorOracleConfig.min_coverage_frac)
    parser.add_argument("--dp-chunk-size", type=int, default=NightSelectorOracleConfig.dp_chunk_size)
    parser.add_argument("--dp-switch-penalty", type=float, default=NightSelectorOracleConfig.dp_switch_penalty)
    args = parser.parse_args()
    chunk_sizes = tuple(int(v) for v in args.chunk_sizes.split(",") if v.strip())
    metrics = run_selector_oracle(
        NightSelectorOracleConfig(
            path_bank_path=args.path_bank_path,
            candidate_summary_path=args.candidate_summary_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
            chunk_sizes=chunk_sizes,
            max_candidates=args.max_candidates,
            min_coverage_frac=args.min_coverage_frac,
            dp_chunk_size=args.dp_chunk_size,
            dp_switch_penalty=args.dp_switch_penalty,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
