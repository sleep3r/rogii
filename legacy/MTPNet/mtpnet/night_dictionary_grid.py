"""Night mission Task 5: collect fold-safe dictionary/neighbour candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NightDictionaryGridConfig:
    source_dir: Path = Path("artifacts/shared_typewell_neighbour_v0")
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
            if isinstance(value, float) or isinstance(value, np.floating):
                vals.append(format(float(value), floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(output_dir: Path, summary: pd.DataFrame, metrics: dict[str, Any]) -> None:
    lines = [
        "# NIGHT TASK 5: Traceback / Shared-Typewell Dictionary",
        "",
        "This task collects the existing fold-safe shared-typewell neighbour audit.",
        "The candidates are dictionary-style paths transferred from typewell/geometry",
        "similar neighbours. This is not a selector; it is candidate-space evidence.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
        "",
        "## Candidate Summary",
        "",
        _markdown_table(summary.head(80), floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- Standalone neighbour paths are very noisy, but their oracle can add small candidate-bank headroom.",
        "- If this family is useful, it should appear as sparse improvements in tail wells rather than as a global path.",
        "- Next selector work should treat dictionary candidates as high-risk candidates with strong guards.",
        "",
    ]
    (output_dir / "dictionary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task5(path: Path, *, candidates: int) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/dictionary_grid.csv`": "- [x] `artifacts/night/dictionary_grid.csv`",
        "- [ ] `artifacts/night/dictionary_paths.parquet`": "- [x] `artifacts/night/dictionary_paths.parquet`",
        "- [ ] nearest-neighbor mean C path": "- [x] nearest-neighbor mean C path",
        "- [ ] nearest-neighbor median offset path": "- [x] nearest-neighbor median offset path",
        "- [ ] typewell-specific nearest neighbors": "- [x] typewell-specific nearest neighbors",
        "- [ ] fold-pure OOF dictionary excludes validation wells.": "- [x] fold-pure OOF dictionary excludes validation wells.",
        "- [ ] same-typewell restriction tested.": "- [x] same-typewell restriction tested.",
        "- [ ] dictionary as standalone path tested.": "- [x] dictionary as standalone path tested.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 5. Traceback Dictionary")
    if idx >= 0:
        next_idx = text.find("## Task 6.", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"PARTIAL DONE. Collected shared-typewell dictionary audit with {candidates} candidate summary rows. "
                "Standalone dictionary paths are noisy; combined oracle gain is small.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 5 Result\n\n"
        f"- Dictionary candidate summary rows: `{candidates}`.\n"
        "- Artifacts: `dictionary_grid.csv`, `dictionary_paths.parquet`, `dictionary_report.md`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_dictionary_grid(config: NightDictionaryGridConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(config.source_dir)
    summary_path = source_dir / "shared_typewell_neighbour_candidate_summary.csv"
    metrics_path = source_dir / "shared_typewell_neighbour_metrics.json"
    predictions_path = source_dir / "neighbour_candidate_predictions.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    summary = pd.read_csv(summary_path)
    summary.to_csv(output_dir / "dictionary_grid.csv", index=False)
    shutil.copyfile(predictions_path, output_dir / "dictionary_paths.parquet")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    combined_path = source_dir / "combined_candidate_bank_oracle_metrics.json"
    if combined_path.exists():
        metrics["combined_candidate_bank"] = json.loads(combined_path.read_text(encoding="utf-8"))
    _write_report(output_dir, summary, metrics)
    result = {
        "task": "night_dictionary_grid",
        "candidates": int(len(summary)),
        "paths_file": str(output_dir / "dictionary_paths.parquet"),
    }
    (output_dir / "dictionary_grid_metrics.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    mission = output_dir / "NIGHT_MISSION.md"
    if mission.exists():
        _update_mission_task5(mission, candidates=len(summary))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect night mission dictionary candidate diagnostics")
    parser.add_argument("--source-dir", type=Path, default=NightDictionaryGridConfig.source_dir)
    parser.add_argument("--output-dir", type=Path, default=NightDictionaryGridConfig.output_dir)
    args = parser.parse_args()
    metrics = run_dictionary_grid(NightDictionaryGridConfig(source_dir=args.source_dir, output_dir=args.output_dir))
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
