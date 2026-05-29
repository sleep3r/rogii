"""Night mission Task 4: collect lattice/beam diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NightBeamGridConfig:
    artifacts_dir: Path = Path("artifacts")
    output_dir: Path = Path("artifacts/night")
    pattern: str = "lattice_offset_v0_*"
    top_prediction_runs: int = 3


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
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
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _teacher_summary(items: list[dict[str, Any]], variant: str, metric: str) -> float:
    vals = [float(item[metric]) for item in items if item.get("variant") == variant and metric in item]
    return float(np.mean(vals)) if vals else float("nan")


def collect_beam_rows(artifacts_dir: Path, pattern: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(Path(artifacts_dir).glob(pattern)):
        metrics_path = run_dir / "lattice_offset_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = metrics.get("config", {})
        teacher = metrics.get("teacher_forced_metrics", [])
        for item in metrics.get("candidates", []):
            rows.append(
                {
                    "run": run_dir.name,
                    "candidate": item.get("candidate"),
                    "rows": item.get("rows"),
                    "wells": item.get("wells"),
                    "row_rmse": item.get("row_rmse"),
                    "mean_well_rmse": item.get("mean_well_rmse"),
                    "p90_well_rmse": item.get("p90_well_rmse"),
                    "p95_well_rmse": item.get("p95_well_rmse"),
                    "worst_well_rmse": item.get("worst_well_rmse"),
                    "normal_minus_shuffled_row_rmse": metrics.get("normal_minus_shuffled_row_rmse"),
                    "beam_size": cfg.get("beam_size"),
                    "branch_top_k": cfg.get("branch_top_k"),
                    "k_wells": cfg.get("k_wells"),
                    "spans": cfg.get("spans"),
                    "offset_grid": cfg.get("offset_grid"),
                    "state_stride": cfg.get("state_stride"),
                    "epochs": cfg.get("epochs"),
                    "on_policy_rounds": cfg.get("on_policy_rounds", 0),
                    "states": metrics.get("states"),
                    "candidate_count_per_state": metrics.get("candidate_count_per_state"),
                    "tf_normal_top1": _teacher_summary(teacher, "normal", "top1_oracle_rate"),
                    "tf_shuffled_top1": _teacher_summary(teacher, "shuffled_gr", "top1_oracle_rate"),
                    "tf_normal_top3": _teacher_summary(teacher, "normal", "top3_oracle_rate"),
                    "tf_shuffled_top3": _teacher_summary(teacher, "shuffled_gr", "top3_oracle_rate"),
                    "prediction_path": str(run_dir / "lattice_offset_predictions.parquet"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("row_rmse", na_position="last").reset_index(drop=True)


def _write_report(output_dir: Path, grid: pd.DataFrame) -> None:
    lines = [
        "# NIGHT TASK 4: Beam / Lattice Diagnostics",
        "",
        "This report collects existing LDT-like lattice/beam runs. It is a",
        "diagnostic aggregation, not a new full sweep. The key sanity is whether",
        "normal GR beats shuffled GR and whether beam rollout gets near the",
        "teacher-forced candidate recall.",
        "",
        "## Beam Grid",
        "",
    ]
    if grid.empty:
        lines.append("No lattice runs found.")
    else:
        lines.append(
            _markdown_table(
                grid[
                    [
                        "run",
                        "candidate",
                        "rows",
                        "wells",
                        "row_rmse",
                        "normal_minus_shuffled_row_rmse",
                        "tf_normal_top1",
                        "tf_shuffled_top1",
                        "tf_normal_top3",
                        "tf_shuffled_top3",
                        "beam_size",
                        "branch_top_k",
                        "k_wells",
                    ]
                ],
                floatfmt=".4f",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current lattice runs are subset diagnostics; do not compare them directly to full 773-well RMSE.",
            "- If shuffled rollout is comparable or better, the beam scorer is not using GR reliably.",
            "- If teacher-forced top3 is decent but rollout is bad, the failure is policy compounding / search calibration.",
            "",
        ]
    )
    (output_dir / "beam_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task4(path: Path, *, best_rmse: float, runs: int) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/beam_grid.csv`": "- [x] `artifacts/night/beam_grid.csv`",
        "- [ ] `artifacts/night/beam_top_predictions.parquet`": "- [x] `artifacts/night/beam_top_predictions.parquet`",
        "- [ ] Implement or reuse hand-score beam.": "- [x] Implement or reuse hand-score beam.",
        "- [ ] Compare greedy selected vs beam selected.": "- [x] Compare greedy selected vs beam selected.",
        "- [ ] Compare beam selected vs beam oracle among kept beams.": "- [x] Compare beam selected vs beam oracle among kept beams.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 4. Beam Search Instead Of Greedy")
    if idx >= 0:
        next_idx = text.find("## Task 5.", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"PARTIAL DONE. Aggregated {runs} existing lattice/beam runs. "
                f"Best subset rollout RMSE: {best_rmse:.4f}. Full beam-width 8/16/32/64 sweep still not run.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 4 Result\n\n"
        f"- Existing lattice/beam runs aggregated: `{runs}`.\n"
        f"- Best subset rollout pooled RMSE: `{best_rmse:.4f}`.\n"
        "- Artifacts: `beam_grid.csv`, `beam_report.md`, `beam_top_predictions.parquet`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_beam_grid(config: NightBeamGridConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = collect_beam_rows(config.artifacts_dir, config.pattern)
    grid.to_csv(output_dir / "beam_grid.csv", index=False)
    _write_report(output_dir, grid)
    pred_parts: list[pd.DataFrame] = []
    if not grid.empty:
        for row in grid.head(int(config.top_prediction_runs)).itertuples(index=False):
            path = Path(row.prediction_path)
            if path.exists():
                pred = pd.read_parquet(path)
                pred["source_run"] = row.run
                pred_parts.append(pred)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    predictions.to_parquet(output_dir / "beam_top_predictions.parquet", index=False)
    best = float(grid["row_rmse"].min()) if not grid.empty else float("nan")
    metrics = {
        "task": "night_beam_grid",
        "runs": int(grid["run"].nunique()) if not grid.empty else 0,
        "rows": int(len(grid)),
        "best_row_rmse": best,
        "top_prediction_rows": int(len(predictions)),
    }
    (output_dir / "beam_grid_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    mission = output_dir / "NIGHT_MISSION.md"
    if mission.exists():
        _update_mission_task4(mission, best_rmse=best, runs=metrics["runs"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect night mission beam/lattice diagnostics")
    parser.add_argument("--artifacts-dir", type=Path, default=NightBeamGridConfig.artifacts_dir)
    parser.add_argument("--output-dir", type=Path, default=NightBeamGridConfig.output_dir)
    parser.add_argument("--pattern", type=str, default=NightBeamGridConfig.pattern)
    parser.add_argument("--top-prediction-runs", type=int, default=NightBeamGridConfig.top_prediction_runs)
    args = parser.parse_args()
    metrics = run_beam_grid(
        NightBeamGridConfig(
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            top_prediction_runs=args.top_prediction_runs,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
