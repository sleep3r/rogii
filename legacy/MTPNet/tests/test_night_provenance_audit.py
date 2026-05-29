from pathlib import Path

import pandas as pd

from mtpnet.night_provenance_audit import NightProvenanceAuditConfig, classify_row, run_provenance_audit


def test_classify_row_marks_oracle_and_subset_as_not_deployable() -> None:
    oracle = classify_row(
        experiment="discrete_offset_v1_selfcal",
        candidate="grid_oracle",
        rows=1000,
        wells=773,
        source="metrics.json",
    )
    assert oracle["artifact_class"] == "oracle"
    assert oracle["deployable"] is False
    assert oracle["used_true_tvt"] is True
    assert "oracle" in oracle["reason"]

    subset = classify_row(
        experiment="pathformer_smoke",
        candidate="pathformer_direct",
        rows=1000,
        wells=4,
        source="pred.parquet",
    )
    assert subset["artifact_class"] == "subset"
    assert subset["deployable"] is False
    assert subset["fold_pure"] is False


def test_run_provenance_audit_writes_clean_scoreboard_and_suspicious_report(tmp_path: Path) -> None:
    scoreboard = pd.DataFrame(
        {
            "experiment": ["discrete_offset_v1_selfcal", "candidate_selector_v0", "pathformer_smoke"],
            "candidate": ["grid_oracle", "candidate_selector_v0", "pathformer_direct"],
            "rows": [1000, 1000, 100],
            "wells": [10, 10, 1],
            "pooled_rmse": [7.0, 10.0, 8.0],
            "mean_well_rmse": [6.0, 9.0, 8.0],
        }
    )
    scoreboard_path = tmp_path / "scoreboard.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    scorer = pd.DataFrame(
        {
            "experiment": ["offset_mdn_v0"],
            "candidate": ["grid_oracle"],
            "row_rmse": [7.0],
            "source_metrics": ["offset_mdn_metrics.json"],
        }
    )
    scorer_path = tmp_path / "scorer.csv"
    scorer.to_csv(scorer_path, index=False)

    metrics = run_provenance_audit(
        NightProvenanceAuditConfig(
            scoreboard_path=scoreboard_path,
            scorer_grid_path=scorer_path,
            output_dir=tmp_path / "night",
            mission_path=tmp_path / "missing.md",
            train_well_count=10,
            min_full_well_frac=0.8,
        )
    )

    assert metrics["suspicious_below_threshold"] == 3
    clean = pd.read_csv(tmp_path / "night" / "clean_scoreboard.csv")
    assert list(clean["candidate"]) == ["candidate_selector_v0"]
    audit = pd.read_csv(tmp_path / "night" / "provenance_audit.csv")
    assert "used_true_tvt" in audit.columns
    assert (tmp_path / "night" / "suspicious_artifacts.md").exists()
