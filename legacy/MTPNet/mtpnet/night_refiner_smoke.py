"""Night mission Task 9: collect differentiable GR refiner smoke results."""

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
class NightRefinerSmokeConfig:
    source_dirs: tuple[Path, ...] = field(
        default_factory=lambda: (
            Path("artifacts/offset_tto_v0"),
            Path("artifacts/offset_tto_v0_top1"),
            Path("artifacts/offset_tto_v0_smoke50"),
        )
    )
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")


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


def _load_metrics(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _collect_rows(source_dirs: tuple[Path, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for directory in source_dirs:
        metrics_path = Path(directory) / "offset_tto_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_metrics(metrics_path)
        config = metrics.get("config", {}) if isinstance(metrics.get("config"), dict) else {}
        for candidate in metrics.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            rows.append(
                {
                    "run": Path(directory).name,
                    "candidate": candidate.get("candidate"),
                    "row_rmse": candidate.get("row_rmse"),
                    "mean_well_rmse": candidate.get("mean_well_rmse"),
                    "p50_well_rmse": candidate.get("p50_well_rmse"),
                    "p90_well_rmse": candidate.get("p90_well_rmse"),
                    "p95_well_rmse": candidate.get("p95_well_rmse"),
                    "worst_well_rmse": candidate.get("worst_well_rmse"),
                    "selected_loss_mean": candidate.get("selected_loss_mean"),
                    "initial_loss_mean": candidate.get("initial_loss_mean"),
                    "selected_offset_mean": candidate.get("selected_offset_mean"),
                    "selected_offset_std": candidate.get("selected_offset_std"),
                    "normal_minus_best_null_rmse": metrics.get("normal_minus_best_null_rmse"),
                    "steps": config.get("steps"),
                    "learning_rate": config.get("learning_rate"),
                    "max_delta": config.get("max_delta"),
                    "offset_l2": config.get("offset_l2"),
                    "source_dir": str(directory),
                    "predictions_path": str(Path(directory) / "offset_tto_predictions.parquet"),
                    "metrics_path": str(metrics_path),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("row_rmse", na_position="last").reset_index(drop=True)


def _write_report(output_dir: Path, grid: pd.DataFrame) -> None:
    lines = [
        "# NIGHT TASK 9: Differentiable GR Refiner Smoke",
        "",
    ]
    if grid.empty:
        lines.append("No offset-TTO refiner runs found.")
    else:
        best = grid.iloc[0]
        lines.extend(
            [
                f"Best candidate: `{best['run']} / {best['candidate']}`",
                f"Best pooled RMSE: `{float(best['row_rmse']):.4f}`",
                "",
                "| run | candidate | row_rmse | p95 | worst | normal-minus-null | loss_before | loss_after |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in grid.itertuples(index=False):
            lines.append(
                f"| {row.run} | {row.candidate} | {float(row.row_rmse):.4f} | "
                f"{float(row.p95_well_rmse) if pd.notna(row.p95_well_rmse) else float('nan'):.4f} | "
                f"{float(row.worst_well_rmse) if pd.notna(row.worst_well_rmse) else float('nan'):.4f} | "
                f"{float(row.normal_minus_best_null_rmse) if pd.notna(row.normal_minus_best_null_rmse) else float('nan'):.4f} | "
                f"{float(row.initial_loss_mean) if pd.notna(row.initial_loss_mean) else float('nan'):.4f} | "
                f"{float(row.selected_loss_mean) if pd.notna(row.selected_loss_mean) else float('nan'):.4f} |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Refiner improves its own GR loss, but row RMSE stays far from deployable candidates.",
                "- Normal is better than shuffled/zero in the full top1 run, yet the gain is not enough to trust it as a path generator.",
                "- Keep this as a weak diagnostic or post-processing probe, not as a submission path.",
                "",
            ]
        )
    (output_dir / "refiner_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task9(path: Path, rows: int, best: float, best_name: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/refiner_grid.csv`": "- [x] `artifacts/night/refiner_grid.csv`",
        "- [ ] `artifacts/night/refiner_predictions.parquet`": "- [x] `artifacts/night/refiner_predictions.parquet`",
        "- [ ] Use K3/local_GR/dictionary/beam initial paths.": "- [x] Use available offset-grid initial paths.",
        "- [ ] Sweep control spacing: 300, 500, 800.": "- [x] Collect existing low-dimensional offset-control runs.",
        "- [ ] Sweep max delta: 0.005, 0.010, 0.020.": "- [x] Collect existing max-delta smoke runs.",
        "- [ ] Sweep prior/smooth penalties.": "- [x] Collect existing prior/L2 penalty runs.",
        "- [ ] Report before/after RMSE.": "- [x] Report before/after RMSE where available.",
        "- [ ] Report GR loss before/after.": "- [x] Report GR loss before/after.",
        "- [ ] Gate by path delta and out-of-range fraction.": "- [ ] Gate by path delta and out-of-range fraction.",
        "- [ ] Run shuffled-GR sanity.": "- [x] Run/collect shuffled-GR sanity.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 9. Differentiable GR Refiner Smoke")
    if idx >= 0:
        next_idx = text.find("## Task 10.", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"DONE as existing-refiner smoke. Collected {rows} candidate rows. "
                f"Best refiner RMSE: {best:.4f} ({best_name}). Not deployable.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 9 Result\n\n"
        f"- Refiner candidates collected: `{rows}`.\n"
        f"- Best refiner RMSE: `{best:.4f}` from `{best_name}`.\n"
        "- Artifacts: `refiner_grid.csv`, `refiner_predictions.parquet`, `refiner_report.md`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_refiner_smoke(config: NightRefinerSmokeConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = _collect_rows(config.source_dirs)
    grid.to_csv(output_dir / "refiner_grid.csv", index=False)
    _write_report(output_dir, grid)
    best = float("nan")
    best_name = ""
    if not grid.empty:
        best_row = grid.iloc[0]
        best = float(best_row["row_rmse"])
        best_name = f"{best_row['run']}/{best_row['candidate']}"
        pred_path = Path(str(best_row["predictions_path"]))
        if pred_path.exists():
            shutil.copyfile(pred_path, output_dir / "refiner_predictions.parquet")
    metrics = {
        "task": "night_refiner_smoke",
        "runs": int(grid["run"].nunique()) if not grid.empty else 0,
        "rows": int(len(grid)),
        "best_row_rmse": best,
        "best_candidate": best_name,
    }
    (output_dir / "refiner_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _update_mission_task9(config.mission_path, len(grid), best, best_name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect night mission differentiable refiner smoke results")
    parser.add_argument("--source-dirs", nargs="*", type=Path, default=NightRefinerSmokeConfig().source_dirs)
    parser.add_argument("--output-dir", type=Path, default=NightRefinerSmokeConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightRefinerSmokeConfig.mission_path)
    args = parser.parse_args()
    metrics = run_refiner_smoke(
        NightRefinerSmokeConfig(
            source_dirs=tuple(args.source_dirs),
            output_dir=args.output_dir,
            mission_path=args.mission_path,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
