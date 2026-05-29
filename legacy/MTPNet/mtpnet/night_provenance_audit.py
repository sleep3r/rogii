"""Post-night audit: classify low-RMSE artifacts as deployable or oracle/leaky."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_INFERENCE_COLUMNS = {"TVT", "Geology", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}


@dataclass(frozen=True)
class NightProvenanceAuditConfig:
    scoreboard_path: Path = Path("artifacts/night/oof_scoreboard.csv")
    scorer_grid_path: Path = Path("artifacts/night/scorer_grid.csv")
    path_bank_summary_path: Path = Path("artifacts/night/path_bank_candidate_summary.csv")
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")
    rmse_threshold: float = 9.0
    train_well_count: int = 773
    min_full_well_frac: float = 0.8


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


def _generator_script(experiment: str, candidate: str) -> str:
    text = f"{experiment} {candidate}".lower()
    mapping = [
        ("discrete_offset", "mtpnet/discrete_offset.py"),
        ("offset_mdn", "mtpnet/offset_mdn.py"),
        ("k_offset", "mtpnet/k_offset_gated.py"),
        ("candidate_selector", "mtpnet/candidate_selector.py"),
        ("chunk_ranker", "mtpnet/chunk_policy.py"),
        ("policy_solver", "policy_solver/"),
        ("shared_typewell", "mtpnet/shared_typewell_neighbour.py"),
        ("traceback", "mtpnet/traceback.py"),
        ("template_envelope", "promising/template_envelope/"),
        ("pathformer", "pathformer/"),
        ("racformer", "racformer/"),
        ("mtp", "mtpnet/track.py"),
        ("geoaligner", "geoaligner/"),
        ("residual_stack", "mtpnet/residual_stack.py"),
    ]
    for needle, script in mapping:
        if needle in text:
            return script
    return ""


def _prediction_file(experiment: str, candidate: str, source: str) -> str:
    if source.endswith(".parquet"):
        return source
    safe = f"{experiment}_{candidate}".replace("/", "_").replace(" ", "_")
    return f"artifacts/night/predictions/{safe}*.parquet"


def classify_row(
    *,
    experiment: str,
    candidate: str,
    rows: int | float,
    wells: int | float,
    source: str = "",
    train_well_count: int = 773,
    min_full_well_frac: float = 0.8,
) -> dict[str, Any]:
    text = f"{experiment} {candidate} {source}".lower()
    reason: list[str] = []
    artifact_class = "deployable_candidate"
    used_true_tvt = False
    used_formations = False
    used_oracle_offset = False
    fold_pure = True
    can_run_on_test = True
    deployable = True

    oracle_tokens = ["oracle", "truth", "true_tvt", "target_value", "best_of_bank", "row_oracle"]
    if any(token in text for token in oracle_tokens):
        artifact_class = "oracle"
        used_true_tvt = True
        used_oracle_offset = "offset" in text or "grid_oracle" in text
        deployable = False
        can_run_on_test = False
        reason.append("oracle/target-derived selection")

    if "grid_oracle" in text:
        artifact_class = "oracle"
        used_true_tvt = True
        used_oracle_offset = True
        deployable = False
        can_run_on_test = False
        reason.append("grid_oracle chooses offset using validation TVT")

    if any(col.lower() in text for col in (c.lower() for c in FORBIDDEN_INFERENCE_COLUMNS if c != "TVT")):
        used_formations = True
        deployable = False
        can_run_on_test = False
        reason.append("mentions train-only formation/geology column")

    if "smoke" in text or float(wells or 0) < train_well_count * min_full_well_frac:
        if artifact_class != "oracle":
            artifact_class = "subset"
        deployable = False
        fold_pure = False
        can_run_on_test = False
        reason.append("subset/smoke coverage")

    if "ranker" in text and "mtp_v1_prior_conditioned" in text:
        if artifact_class == "deployable_candidate":
            artifact_class = "leakage_risk"
        deployable = False
        fold_pure = False
        can_run_on_test = False
        reason.append("known ranker leakage/partial-validation risk")

    if "shuffled" in text or "zero_gr" in text:
        if artifact_class == "deployable_candidate":
            artifact_class = "null_diagnostic"
        deployable = False
        can_run_on_test = False
        reason.append("null/shuffled diagnostic")

    return {
        "artifact_class": artifact_class,
        "deployable": bool(deployable),
        "fold_pure": bool(fold_pure),
        "can_run_on_test": bool(can_run_on_test),
        "used_true_tvt": bool(used_true_tvt),
        "used_formations": bool(used_formations),
        "used_oracle_offset": bool(used_oracle_offset),
        "generator_script": _generator_script(experiment, candidate),
        "prediction_file": _prediction_file(experiment, candidate, source),
        "reason": "; ".join(dict.fromkeys(reason)) if reason else "schema-safe by heuristic audit",
    }


def _load_scoreboard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"pooled_rmse": "rmse"})
    df["source_table"] = str(path)
    df["source"] = ""
    return df


def _load_scorer_grid(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"row_rmse": "rmse", "source_metrics": "source"})
    df["rows"] = np.nan
    df["wells"] = np.nan
    df["source_table"] = str(path)
    return df


def _write_report(output_dir: Path, audit: pd.DataFrame, clean: pd.DataFrame, threshold: float) -> None:
    suspicious = audit[(audit["rmse"] < threshold) & (~audit["deployable"])]
    lines = [
        "# Suspicious Low-RMSE Artifact Audit",
        "",
        f"RMSE threshold: `{threshold}`",
        f"Rows audited: `{len(audit)}`",
        f"Clean deployable scoreboard rows: `{len(clean)}`",
        f"Suspicious rows below threshold: `{len(suspicious)}`",
        "",
        "## Low-RMSE Suspicious Rows",
        "",
        "| experiment | candidate | rmse | class | true_tvt | oracle_offset | fold_pure | test | reason |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in suspicious.sort_values("rmse").head(120).itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.candidate} | {float(row.rmse):.4f} | {row.artifact_class} | "
            f"{int(row.used_true_tvt)} | {int(row.used_oracle_offset)} | {int(row.fold_pure)} | "
            f"{int(row.can_run_on_test)} | {row.reason} |"
        )
    lines.extend(
        [
            "",
            "## Key Verdict",
            "",
            "`7.6375` is classified as `grid_oracle`: it chooses the offset using validation TVT and is not deployable.",
            "",
        ]
    )
    (output_dir / "suspicious_artifacts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission(path: Path, *, suspicious: int, clean_best: float) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    section = (
        "\n## Post-Night Task A. Provenance Audit\n\n"
        "Artifacts:\n\n"
        "- [x] `artifacts/night/provenance_audit.csv`\n"
        "- [x] `artifacts/night/clean_scoreboard.csv`\n"
        "- [x] `artifacts/night/suspicious_artifacts.md`\n\n"
        "Verdict:\n\n"
        "```text\n"
        f"`7.6375` is oracle/grid-offset, not deployable. Suspicious low-RMSE rows below threshold: {suspicious}. "
        f"Best clean full-ish deployable scoreboard RMSE: {clean_best:.4f}.\n"
        "```\n"
    )
    if "## Post-Night Task A. Provenance Audit" not in text:
        text = text.replace("## Final Decision Tree", section + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_provenance_audit(config: NightProvenanceAuditConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [_load_scoreboard(config.scoreboard_path), _load_scorer_grid(config.scorer_grid_path)]
    audit = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if audit.empty:
        raise ValueError("No scoreboard/scorer rows to audit")
    audited_rows: list[dict[str, Any]] = []
    for row in audit.itertuples(index=False):
        cls = classify_row(
            experiment=str(row.experiment),
            candidate=str(row.candidate),
            rows=getattr(row, "rows", np.nan),
            wells=getattr(row, "wells", np.nan),
            source=str(getattr(row, "source", "")),
            train_well_count=config.train_well_count,
            min_full_well_frac=config.min_full_well_frac,
        )
        audited_rows.append({**row._asdict(), **cls})
    audit_df = pd.DataFrame(audited_rows)
    audit_df.to_csv(output_dir / "provenance_audit.csv", index=False)
    clean = audit_df[audit_df["deployable"]].copy()
    clean = clean.sort_values("rmse", na_position="last").reset_index(drop=True)
    clean.to_csv(output_dir / "clean_scoreboard.csv", index=False)
    _write_report(output_dir, audit_df, clean, config.rmse_threshold)
    suspicious = audit_df[(audit_df["rmse"] < config.rmse_threshold) & (~audit_df["deployable"])]
    clean_best = float(clean["rmse"].min()) if not clean.empty else float("nan")
    metrics = {
        "task": "night_provenance_audit",
        "audited_rows": int(len(audit_df)),
        "suspicious_below_threshold": int(len(suspicious)),
        "clean_rows": int(len(clean)),
        "clean_best_rmse": clean_best,
    }
    (output_dir / "provenance_audit_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _update_mission(config.mission_path, suspicious=len(suspicious), clean_best=clean_best)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit low-RMSE artifacts for oracle/leakage provenance")
    parser.add_argument("--scoreboard-path", type=Path, default=NightProvenanceAuditConfig.scoreboard_path)
    parser.add_argument("--scorer-grid-path", type=Path, default=NightProvenanceAuditConfig.scorer_grid_path)
    parser.add_argument("--output-dir", type=Path, default=NightProvenanceAuditConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightProvenanceAuditConfig.mission_path)
    parser.add_argument("--rmse-threshold", type=float, default=NightProvenanceAuditConfig.rmse_threshold)
    args = parser.parse_args()
    metrics = run_provenance_audit(
        NightProvenanceAuditConfig(
            scoreboard_path=args.scoreboard_path,
            scorer_grid_path=args.scorer_grid_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
            rmse_threshold=args.rmse_threshold,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
