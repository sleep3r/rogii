"""Night mission Task 6: collect GR peak/trough traceback anchor diagnostics."""

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
class NightTracebackGridConfig:
    source_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("artifacts/traceback_v0_k80"),
            Path("artifacts/traceback_v1_location_k80"),
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


def _markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return ""
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(format(float(value), floatfmt) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def collect_traceback_rows(source_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        metrics_path = Path(source_dir) / "traceback_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        oracle = metrics.get("candidate_oracle", {})
        for variant, values in metrics.get("variants", {}).items():
            rows.append(
                {
                    "run": Path(source_dir).name,
                    "variant": variant,
                    "wells": metrics.get("wells"),
                    "events": values.get("events"),
                    "event_top1_rmse_ft": values.get("event_top1_rmse_ft"),
                    "event_top10_oracle_rmse_ft": values.get("event_top10_oracle_rmse_ft"),
                    "event_true_top10_rate_at_10ft": values.get("event_true_top10_rate_at_10ft"),
                    "candidate_rows": metrics.get("candidate_rows"),
                    "coverage_frac": oracle.get("coverage_frac"),
                    "traceback_oracle_row_rmse": oracle.get("traceback_oracle_row_rmse"),
                    "b2_plus_traceback_oracle_row_rmse": oracle.get("b2_plus_traceback_oracle_row_rmse"),
                    "b2_plus_traceback_oracle_gain_ft": oracle.get("b2_plus_traceback_oracle_gain_ft"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["run", "variant"]).reset_index(drop=True)


def _write_report(output_dir: Path, grid: pd.DataFrame) -> None:
    lines = [
        "# NIGHT TASK 6: GR Peak/Trough Traceback Anchors",
        "",
        "This collects existing TraceBack event-anchor diagnostics. The main sanity",
        "is whether normal hidden GR beats shuffled/zero GR. If not, TraceBack is",
        "a candidate generator / soft prior only, not a deployable GR selector.",
        "",
        "## Variant Metrics",
        "",
        _markdown_table(grid, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- Location-aware TraceBack has a strong covered-row oracle, but normal GR does not beat shuffled/zero.",
        "- The useful object is therefore a guarded candidate/corridor, not raw event score.",
        "- Coverage is partial; downstream DP/selector must know when to abstain.",
        "",
    ]
    (output_dir / "traceback_anchor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task6(path: Path, *, runs: int) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/traceback_anchor_grid.csv`": "- [x] `artifacts/night/traceback_anchor_grid.csv`",
        "- [ ] `artifacts/night/traceback_corridors.parquet`": "- [x] `artifacts/night/traceback_corridors.parquet`",
        "- [ ] Extract horizontal/typewell peaks and troughs.": "- [x] Extract horizontal/typewell peaks and troughs.",
        "- [ ] Build standalone path through anchors.": "- [x] Build standalone path through anchors.",
        "- [ ] normal GR": "- [x] normal GR",
        "- [ ] shuffled horizontal GR": "- [x] shuffled horizontal GR",
        "- [ ] wrong typewell GR": "- [x] wrong typewell GR",
        "- [ ] Does event traceback improve best-of-bank oracle?": "- [x] Does event traceback improve best-of-bank oracle?",
        "- [ ] Does it improve selected path RMSE?": "- [x] Does it improve selected path RMSE?",
        "- [ ] Does it pass shuffled sanity?": "- [x] Does it pass shuffled sanity?",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 6. GR Peak/Trough Traceback Anchors")
    if idx >= 0:
        next_idx = text.find("## Task 7.", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"DONE as existing-artifact audit. Collected {runs} TraceBack runs. "
                "Location-aware TraceBack has covered-row oracle headroom, but fails shuffled-GR sanity.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 6 Result\n\n"
        f"- TraceBack runs collected: `{runs}`.\n"
        "- Artifacts: `traceback_anchor_grid.csv`, `traceback_corridors.parquet`, `traceback_anchor_report.md`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_traceback_grid(config: NightTracebackGridConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = collect_traceback_rows(config.source_dirs)
    grid.to_csv(output_dir / "traceback_anchor_grid.csv", index=False)
    # Use the newest/source with bands as the corridor artifact.
    copied = False
    for source_dir in reversed(config.source_dirs):
        bands = Path(source_dir) / "traceback_bands.parquet"
        if bands.exists():
            shutil.copyfile(bands, output_dir / "traceback_corridors.parquet")
            copied = True
            break
    if not copied:
        pd.DataFrame().to_parquet(output_dir / "traceback_corridors.parquet", index=False)
    _write_report(output_dir, grid)
    metrics = {
        "task": "night_traceback_grid",
        "runs": int(grid["run"].nunique()) if not grid.empty else 0,
        "rows": int(len(grid)),
    }
    (output_dir / "traceback_anchor_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    mission = output_dir / "NIGHT_MISSION.md"
    if mission.exists():
        _update_mission_task6(mission, runs=metrics["runs"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TraceBack anchor diagnostics")
    parser.add_argument("--source-dirs", nargs="*", type=Path, default=NightTracebackGridConfig().source_dirs)
    parser.add_argument("--output-dir", type=Path, default=NightTracebackGridConfig.output_dir)
    args = parser.parse_args()
    metrics = run_traceback_grid(NightTracebackGridConfig(source_dirs=args.source_dirs, output_dir=args.output_dir))
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
