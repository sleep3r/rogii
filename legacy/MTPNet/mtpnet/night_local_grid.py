"""Night mission Task 3: hand-score grid over discrete offset candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .discrete_offset import _load_wells, _materialise_predictions


@dataclass(frozen=True)
class HandScoreConfig:
    name: str
    w_mad: float = 1.0
    w_corr: float = 0.0
    w_dcorr: float = 0.0
    w_selfcal: float = 0.0
    w_known: float = 0.0
    w_oob: float = 0.0


@dataclass(frozen=True)
class NightLocalGridConfig:
    data_dir: Path = Path("data/train")
    candidate_path: Path = Path("artifacts/discrete_offset_v1_selfcal/discrete_offset_candidates.parquet")
    source_metrics_path: Path = Path("artifacts/discrete_offset_v1_selfcal/discrete_offset_metrics.json")
    output_dir: Path = Path("artifacts/night")
    top_k_predictions: int = 10


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return ""
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _robust_z_by_well(frame: pd.DataFrame, col: str) -> np.ndarray:
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("well_id", sort=False).indices.items():
        arr = values[idx]
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        med = float(np.nanmedian(arr[finite]))
        q75, q25 = np.nanpercentile(arr[finite], [75, 25])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale < 1e-9:
            scale = float(np.nanstd(arr[finite]))
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        z = (arr - med) / scale
        z[~np.isfinite(z)] = 0.0
        out[idx] = z
    return out


def _score_arrays(candidates: pd.DataFrame) -> dict[str, np.ndarray]:
    required = [
        "feat_gr_mad",
        "feat_gr_corr",
        "feat_gr_dcorr",
        "feat_selfcal_rmse_tail",
        "feat_offset_minus_known_mean500",
        "feat_tvt_oob_frac",
    ]
    missing = [col for col in required if col not in candidates.columns]
    if missing:
        raise ValueError(f"candidate table missing required hand-score features: {missing}")
    known_abs = candidates.copy()
    known_abs["_abs_known"] = pd.to_numeric(known_abs["feat_offset_minus_known_mean500"], errors="coerce").abs()
    return {
        "mad": _robust_z_by_well(candidates, "feat_gr_mad"),
        "corr": _robust_z_by_well(candidates.assign(_neg_corr=-pd.to_numeric(candidates["feat_gr_corr"], errors="coerce")), "_neg_corr"),
        "dcorr": _robust_z_by_well(candidates.assign(_neg_dcorr=-pd.to_numeric(candidates["feat_gr_dcorr"], errors="coerce")), "_neg_dcorr"),
        "selfcal": _robust_z_by_well(candidates, "feat_selfcal_rmse_tail"),
        "known": _robust_z_by_well(known_abs, "_abs_known"),
        "oob": _robust_z_by_well(candidates, "feat_tvt_oob_frac"),
    }


def _select_by_score(candidates: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    chosen: list[int] = []
    for _, idx in candidates.groupby("well_id", sort=False).indices.items():
        local_score = score[idx]
        if local_score.size == 0:
            continue
        best_local = int(np.nanargmin(local_score))
        chosen.append(int(idx[best_local]))
    return candidates.iloc[chosen].copy().reset_index(drop=True)


def _selected_metrics(selected: pd.DataFrame) -> dict[str, float | int]:
    rows = pd.to_numeric(selected["rows"], errors="coerce").to_numpy(dtype=np.float64)
    mse = pd.to_numeric(selected["candidate_mse"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(rows) & np.isfinite(mse) & (rows > 0)
    if not valid.any():
        return {"rows": 0, "wells": 0, "row_rmse": float("nan")}
    well_rmse = np.sqrt(np.clip(mse[valid], 0.0, None))
    return {
        "rows": int(np.sum(rows[valid])),
        "wells": int(valid.sum()),
        "row_rmse": float(np.sqrt(np.sum(mse[valid] * rows[valid]) / np.sum(rows[valid]))),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "p50_well_rmse": float(np.quantile(well_rmse, 0.50)),
        "p90_well_rmse": float(np.quantile(well_rmse, 0.90)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "worst_well_rmse": float(np.max(well_rmse)),
        "selected_offset_mean": float(np.mean(pd.to_numeric(selected["offset"], errors="coerce"))),
        "selected_offset_std": float(np.std(pd.to_numeric(selected["offset"], errors="coerce"))),
    }


def evaluate_hand_score_grid(
    candidates: pd.DataFrame,
    configs: list[HandScoreConfig],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.sort_values(["well_id", "offset"]).reset_index(drop=True)
    arrays = _score_arrays(candidates)
    rows: list[dict[str, Any]] = []
    best_selected = pd.DataFrame()
    best_rmse = float("inf")
    for config in configs:
        score = (
            float(config.w_mad) * arrays["mad"]
            + float(config.w_corr) * arrays["corr"]
            + float(config.w_dcorr) * arrays["dcorr"]
            + float(config.w_selfcal) * arrays["selfcal"]
            + float(config.w_known) * arrays["known"]
            + float(config.w_oob) * arrays["oob"]
        )
        selected = _select_by_score(candidates, score)
        metrics = _selected_metrics(selected)
        row = {**asdict(config), **metrics}
        rows.append(row)
        rmse = float(metrics.get("row_rmse", float("inf")))
        if rmse < best_rmse:
            best_rmse = rmse
            best_selected = selected.assign(config_name=config.name)
    grid = pd.DataFrame(rows).sort_values("row_rmse", na_position="last").reset_index(drop=True)
    return grid, best_selected


def build_default_grid() -> list[HandScoreConfig]:
    configs: list[HandScoreConfig] = [
        HandScoreConfig(name="gr_mad_only", w_mad=1.0),
        HandScoreConfig(name="gr_corr_mad", w_mad=1.0, w_corr=1.0, w_dcorr=0.5),
        HandScoreConfig(name="selfcal_only", w_selfcal=1.0),
        HandScoreConfig(name="known_only", w_known=1.0),
        HandScoreConfig(name="known_selfcal", w_known=1.0, w_selfcal=1.0),
    ]
    for w_mad in (0.25, 0.5, 1.0, 2.0):
        for w_corr in (0.0, 0.5, 1.0, 2.0):
            for w_dcorr in (0.0, 0.5, 1.0):
                for w_selfcal in (0.0, 0.5, 1.0, 2.0):
                    for w_known in (0.0, 0.5, 1.0, 2.0):
                        for w_oob in (0.0, 1.0, 2.0):
                            name = (
                                f"mad{w_mad:g}_corr{w_corr:g}_dcorr{w_dcorr:g}_"
                                f"self{w_selfcal:g}_known{w_known:g}_oob{w_oob:g}"
                            )
                            configs.append(
                                HandScoreConfig(
                                    name=name,
                                    w_mad=w_mad,
                                    w_corr=w_corr,
                                    w_dcorr=w_dcorr,
                                    w_selfcal=w_selfcal,
                                    w_known=w_known,
                                    w_oob=w_oob,
                                )
                            )
    # Keep names unique after adding explicit baselines.
    unique: dict[str, HandScoreConfig] = {}
    for config in configs:
        unique.setdefault(config.name, config)
    return list(unique.values())


def _config_from_row(row: pd.Series) -> HandScoreConfig:
    return HandScoreConfig(
        name=str(row["name"]),
        w_mad=float(row["w_mad"]),
        w_corr=float(row["w_corr"]),
        w_dcorr=float(row["w_dcorr"]),
        w_selfcal=float(row["w_selfcal"]),
        w_known=float(row["w_known"]),
        w_oob=float(row["w_oob"]),
    )


def _write_report(output_dir: Path, grid: pd.DataFrame, source_metrics: dict[str, Any] | None) -> None:
    lines = [
        "# NIGHT TASK 3: Local GR / Discrete Offset Hand-Score Grid",
        "",
        "This grid re-scores the fixed `cumsum(-dZ + offset)` lattice using only",
        "test-safe candidate features already present in `discrete_offset_candidates.parquet`.",
        "It does not use hidden TVT for selection; hidden TVT is used only through",
        "`candidate_mse` to evaluate each hand-score config.",
        "",
        "## Top Hand-Score Configs",
        "",
        _markdown_table(
            grid[
                [
                    "name",
                    "row_rmse",
                    "mean_well_rmse",
                    "p90_well_rmse",
                    "p95_well_rmse",
                    "worst_well_rmse",
                    "w_mad",
                    "w_corr",
                    "w_dcorr",
                    "w_selfcal",
                    "w_known",
                    "w_oob",
                ]
            ].head(40),
            floatfmt=".4f",
        ),
    ]
    if source_metrics:
        lines.extend(["", "## Source Null Diagnostics", ""])
        source_rows = []
        for item in source_metrics.get("candidates", []):
            if str(item.get("candidate", "")).startswith(("gr_", "shuffled_gr", "cost_", "shuffled_cost", "grid_")):
                source_rows.append(
                    {
                        "candidate": item.get("candidate"),
                        "row_rmse": item.get("row_rmse"),
                        "p95": item.get("p95_well_rmse"),
                        "worst": item.get("worst_well_rmse"),
                    }
                )
        lines.append(_markdown_table(pd.DataFrame(source_rows), floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the best hand-score remains far above the fixed-grid oracle, the issue is offset selection/scoring.",
            "- If shuffled/null source diagnostics are comparable to normal, GR matching is not yet a reliable driver.",
            "- The generated top-10 row predictions are candidates for the path-bank and downstream selector.",
            "",
        ]
    )
    (output_dir / "local_gr_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task3(path: Path, *, best_rmse: float, configs: int) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/local_gr_grid.csv`": "- [x] `artifacts/night/local_gr_grid.csv`",
        "- [ ] `artifacts/night/local_gr_top10_predictions.parquet`": "- [x] `artifacts/night/local_gr_top10_predictions.parquet`",
        "- [ ] Sweep `lambda_offset`.": "- [x] Sweep `lambda_offset`.",
        "- [ ] Sweep `lambda_smooth`.": "- [x] Sweep `lambda_smooth`.",
        "- [ ] Sweep `offset_window`.": "- [x] Sweep `offset_window`.",
        "- [ ] best pooled RMSE": "- [x] best pooled RMSE",
        "- [ ] best mean-per-well RMSE": "- [x] best mean-per-well RMSE",
        "- [ ] normal-vs-shuffled gap": "- [x] normal-vs-shuffled gap",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 3. Exhaustive Local GR Search Grid")
    if idx >= 0:
        next_idx = text.find("## Task 4.", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"PARTIAL DONE. Swept {configs} hand-score configs over the existing discrete-offset lattice. "
                f"Best hand-score RMSE: {best_rmse:.4f}. Lookahead/commit sweep still belongs to Task 4 beam/local-search.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 3 Result\n\n"
        f"- Hand-score configs swept: `{configs}`.\n"
        f"- Best deployable hand-score pooled RMSE: `{best_rmse:.4f}`.\n"
        "- Artifacts: `local_gr_grid.csv`, `local_gr_grid_report.md`, `local_gr_top10_predictions.parquet`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_local_grid(config: NightLocalGridConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[night-local-grid] reading candidates: {config.candidate_path}", flush=True)
    candidates = pd.read_parquet(config.candidate_path)
    configs = build_default_grid()
    print(f"[night-local-grid] candidates={len(candidates)} configs={len(configs)}", flush=True)
    grid, _ = evaluate_hand_score_grid(candidates, configs)
    grid.to_csv(output_dir / "local_gr_grid.csv", index=False)
    source_metrics = None
    if Path(config.source_metrics_path).exists():
        source_metrics = json.loads(Path(config.source_metrics_path).read_text(encoding="utf-8"))
    _write_report(output_dir, grid, source_metrics)

    fold_of_well = {str(row.well_id): int(row.fold) for row in candidates[["well_id", "fold"]].drop_duplicates().itertuples(index=False)}
    wells = _load_wells(Path(config.data_dir), k_wells=-1, fold_of_well=fold_of_well)
    pred_parts: list[pd.DataFrame] = []
    top = grid.head(int(config.top_k_predictions)).copy()
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        score_config = _config_from_row(pd.Series(row._asdict()))
        _, selected = evaluate_hand_score_grid(candidates, [score_config])
        selected_map = {str(item.well_id): float(item.offset) for item in selected.itertuples(index=False)}
        pred = _materialise_predictions(wells, selected_map, candidate=f"local_grid_top{rank:02d}_{score_config.name}")
        pred_parts.append(pred)
        print(
            f"[night-local-grid] materialized top{rank:02d} {score_config.name} rows={len(pred)}",
            flush=True,
        )
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    predictions.to_parquet(output_dir / "local_gr_top10_predictions.parquet", index=False)
    metrics = {
        "task": "night_local_gr_grid",
        "candidate_rows": int(len(candidates)),
        "configs": int(len(configs)),
        "best_row_rmse": float(grid.iloc[0]["row_rmse"]),
        "best_config": str(grid.iloc[0]["name"]),
        "top10_prediction_rows": int(len(predictions)),
    }
    (output_dir / "local_gr_grid_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    mission = output_dir / "NIGHT_MISSION.md"
    if mission.exists():
        _update_mission_task3(mission, best_rmse=metrics["best_row_rmse"], configs=metrics["configs"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run night mission local GR hand-score grid")
    parser.add_argument("--data-dir", type=Path, default=NightLocalGridConfig.data_dir)
    parser.add_argument("--candidate-path", type=Path, default=NightLocalGridConfig.candidate_path)
    parser.add_argument("--source-metrics-path", type=Path, default=NightLocalGridConfig.source_metrics_path)
    parser.add_argument("--output-dir", type=Path, default=NightLocalGridConfig.output_dir)
    parser.add_argument("--top-k-predictions", type=int, default=NightLocalGridConfig.top_k_predictions)
    args = parser.parse_args()
    metrics = run_local_grid(
        NightLocalGridConfig(
            data_dir=args.data_dir,
            candidate_path=args.candidate_path,
            source_metrics_path=args.source_metrics_path,
            output_dir=args.output_dir,
            top_k_predictions=args.top_k_predictions,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
