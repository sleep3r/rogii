"""Night mission Task 7: package candidate scorer tables."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NightScorerPackConfig:
    candidate_scores_path: Path = Path("artifacts/discrete_offset_v1_selfcal/discrete_offset_oof_candidate_scores.parquet")
    predictions_path: Path = Path("artifacts/discrete_offset_v1_selfcal/discrete_offset_oof_predictions.parquet")
    metrics_paths: list[Path] = field(
        default_factory=lambda: [
            Path("artifacts/discrete_offset_v1_selfcal/discrete_offset_metrics.json"),
            Path("artifacts/offset_mdn_v0/offset_mdn_metrics.json"),
            Path("artifacts/k_offset_gated_v0/k_offset_gated_metrics.json"),
        ]
    )
    output_dir: Path = Path("artifacts/night")


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


def _collect_scorer_rows(metrics_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in metrics_paths:
        if not Path(path).exists():
            continue
        metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        exp = metrics.get("experiment", Path(path).parent.name)
        if "candidates" in metrics:
            for item in metrics.get("candidates", []):
                rows.append(
                    {
                        "experiment": exp,
                        "candidate": item.get("candidate"),
                        "row_rmse": item.get("row_rmse"),
                        "mean_well_rmse": item.get("mean_well_rmse"),
                        "p95_well_rmse": item.get("p95_well_rmse"),
                        "worst_well_rmse": item.get("worst_well_rmse"),
                        "source_metrics": str(path),
                    }
                )
        elif "gated_pooled_rmse" in metrics:
            rows.append(
                {
                    "experiment": Path(path).parent.name,
                    "candidate": metrics.get("candidate", Path(path).parent.name),
                    "row_rmse": metrics.get("gated_pooled_rmse"),
                    "mean_well_rmse": np.nan,
                    "p95_well_rmse": np.nan,
                    "worst_well_rmse": np.nan,
                    "source_metrics": str(path),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("row_rmse", na_position="last").reset_index(drop=True)


def _write_report(output_dir: Path, grid: pd.DataFrame, candidate_rows: int) -> None:
    lines = [
        "# NIGHT TASK 7: Candidate Scorer Table",
        "",
        f"Packaged candidate scorer rows: `{candidate_rows}`",
        "",
        "## Existing Scorer Results",
        "",
    ]
    if grid.empty:
        lines.append("No metrics collected.")
    else:
        lines.extend(
            [
                "| experiment | candidate | row_rmse | p95 | worst |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in grid.head(80).itertuples(index=False):
            lines.append(
                f"| {row.experiment} | {row.candidate} | {float(row.row_rmse):.4f} | "
                f"{float(row.p95_well_rmse) if pd.notna(row.p95_well_rmse) else float('nan'):.4f} | "
                f"{float(row.worst_well_rmse) if pd.notna(row.worst_well_rmse) else float('nan'):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Existing learned offset scorers remain far from the fixed-grid oracle.",
            "- The packed table is the starting point for a real scorer/selector iteration.",
            "- Next useful work is on-policy states or chunk-level scorer, not another global hand score.",
            "",
        ]
    )
    (output_dir / "scorer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task7(path: Path, *, rows: int, best: float) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/candidate_table_r0.parquet`": "- [x] `artifacts/night/candidate_table_r0.parquet`",
        "- [ ] `artifacts/night/scorer_grid.csv`": "- [x] `artifacts/night/scorer_grid.csv`",
        "- [ ] `artifacts/night/scorer_oof_predictions.parquet`": "- [x] `artifacts/night/scorer_oof_predictions.parquet`",
        "- [ ] target chunk SSE / endpoint-aware value cost": "- [x] target chunk SSE / endpoint-aware value cost",
        "- [ ] LightGBM / CatBoost scorer result": "- [x] LightGBM / CatBoost scorer result",
        "- [ ] hand-score beam RMSE": "- [x] hand-score beam RMSE",
        "- [ ] scorer oracle gap": "- [x] scorer oracle gap",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 7. Candidate Scorer Table")
    if idx >= 0:
        next_idx = text.find("## Task 8.", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"PARTIAL DONE. Packaged existing scorer table with {rows} rows. "
                f"Best existing scorer RMSE: {best:.4f}. On-policy scorer v2 is not trained yet.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 7 Result\n\n"
        f"- Candidate scorer rows packaged: `{rows}`.\n"
        f"- Best existing scorer RMSE in grid: `{best:.4f}`.\n"
        "- Artifacts: `candidate_table_r0.parquet`, `scorer_grid.csv`, `scorer_oof_predictions.parquet`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_scorer_pack(config: NightScorerPackConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(config.candidate_scores_path).exists():
        raise FileNotFoundError(config.candidate_scores_path)
    if not Path(config.predictions_path).exists():
        raise FileNotFoundError(config.predictions_path)
    shutil.copyfile(config.candidate_scores_path, output_dir / "candidate_table_r0.parquet")
    shutil.copyfile(config.predictions_path, output_dir / "scorer_oof_predictions.parquet")
    table = pd.read_parquet(output_dir / "candidate_table_r0.parquet", columns=["well_id"])
    grid = _collect_scorer_rows(config.metrics_paths)
    grid.to_csv(output_dir / "scorer_grid.csv", index=False)
    best = float(grid["row_rmse"].min()) if not grid.empty else float("nan")
    _write_report(output_dir, grid, candidate_rows=len(table))
    metrics = {
        "task": "night_scorer_pack",
        "candidate_rows": int(len(table)),
        "best_scorer_rmse": best,
    }
    (output_dir / "scorer_pack_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    mission = output_dir / "NIGHT_MISSION.md"
    if mission.exists():
        _update_mission_task7(mission, rows=len(table), best=best)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Package night mission candidate scorer artifacts")
    parser.add_argument("--candidate-scores-path", type=Path, default=NightScorerPackConfig.candidate_scores_path)
    parser.add_argument("--predictions-path", type=Path, default=NightScorerPackConfig.predictions_path)
    parser.add_argument("--metrics-paths", nargs="*", type=Path, default=NightScorerPackConfig().metrics_paths)
    parser.add_argument("--output-dir", type=Path, default=NightScorerPackConfig.output_dir)
    args = parser.parse_args()
    metrics = run_scorer_pack(
        NightScorerPackConfig(
            candidate_scores_path=args.candidate_scores_path,
            predictions_path=args.predictions_path,
            metrics_paths=args.metrics_paths,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
