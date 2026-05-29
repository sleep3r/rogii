"""Assess which OOF path-bank candidates can be regenerated for test."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NightTestBankFeasibilityConfig:
    candidate_summary_path: Path = Path("artifacts/night/path_bank_candidate_summary.csv")
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


def _classify(experiment: str, candidate: str, path_col: str) -> dict[str, Any]:
    text = f"{experiment} {candidate} {path_col}".lower()
    deployable = True
    reason: list[str] = []
    command = ""
    family = "unknown"
    if re.search(r"oracle|truth|target|best_of", text):
        deployable = False
        reason.append("oracle/target-derived")
    if re.search(r"shuffled|zero_gr|zero", text):
        deployable = False
        reason.append("null diagnostic")
    if "smoke" in text:
        deployable = False
        reason.append("smoke/subset artifact")
    if "candidate_selector" in text:
        family = "candidate_selector"
        command = "retrain/generate candidate selector test predictions"
    elif "chunk_ranker" in text or "policy_solver" in text:
        family = "chunk_policy"
        command = "run chunk policy on test path bank"
    elif "residual_stack" in text:
        family = "residual_stack"
        command = "make residual-stack test/inference"
    elif "k_offset" in text or "discrete_offset" in text:
        family = "offset_grid"
        command = "run discrete/k-offset test candidate generator"
    elif "offset_tto" in text:
        family = "offset_tto"
        command = "optional; refiner is weak and should be gated"
    elif "dtvt_state" in text:
        family = "dtvt_state"
        command = "run dtvt-state model test inference"
    elif "shared_typewell" in text:
        family = "dictionary"
        command = "run fold-free train dictionary on test"
    if not reason:
        reason.append("appears test-generatable by heuristic; needs actual inference command")
    return {
        "family": family,
        "deployable": bool(deployable),
        "reason": "; ".join(reason),
        "test_generation_note": command,
    }


def _write_report(output_dir: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Test Path-Bank Feasibility",
        "",
        f"Candidates audited: `{len(table)}`",
        f"Deployable by heuristic: `{int(table['deployable'].sum())}`",
        "",
        "| deployable | family | candidate | pooled_rmse | reason | test note |",
        "|---:|---|---|---:|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {int(row.deployable)} | {row.family} | {row.candidate} | {float(row.pooled_rmse):.4f} | "
            f"{row.reason} | {row.test_generation_note} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a feasibility audit, not `path_bank_test.parquet` itself.",
            "- A real test bank still needs candidate-family inference commands wired and schema-checked.",
            "- Diagnostic/null/oracle candidates must stay out of submit candidates.",
            "",
        ]
    )
    (output_dir / "path_bank_test_feasibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission(path: Path, metrics: dict[str, Any]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    section = (
        "\n## Post-Night Task E. Test Path-Bank Feasibility\n\n"
        "Artifacts:\n\n"
        "- [x] `artifacts/night/path_bank_test_feasibility.csv`\n"
        "- [x] `artifacts/night/path_bank_test_feasibility.md`\n\n"
        "Verdict:\n\n"
        "```text\n"
        f"Deployable-ish candidates by heuristic: {metrics['deployable_candidates']} / {metrics['candidates']}. "
        "This is not yet path_bank_test.parquet; it is the wiring checklist for test inference.\n"
        "```\n"
    )
    if "## Post-Night Task E. Test Path-Bank Feasibility" not in text:
        text = text.replace("## Final Decision Tree", section + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_test_bank_feasibility(config: NightTestBankFeasibilityConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(config.candidate_summary_path)
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        cls = _classify(str(row.experiment), str(row.candidate), str(row.path_col))
        rows.append({**row._asdict(), **cls})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "path_bank_test_feasibility.csv", index=False)
    _write_report(output_dir, table)
    metrics = {
        "task": "night_test_bank_feasibility",
        "candidates": int(len(table)),
        "deployable_candidates": int(table["deployable"].sum()),
    }
    (output_dir / "path_bank_test_feasibility_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _update_mission(config.mission_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit which path-bank candidates can be generated on test")
    parser.add_argument("--candidate-summary-path", type=Path, default=NightTestBankFeasibilityConfig.candidate_summary_path)
    parser.add_argument("--output-dir", type=Path, default=NightTestBankFeasibilityConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightTestBankFeasibilityConfig.mission_path)
    args = parser.parse_args()
    metrics = run_test_bank_feasibility(
        NightTestBankFeasibilityConfig(
            candidate_summary_path=args.candidate_summary_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
