from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.tail_audit import (
    build_tail_audit,
    classify_tail_row,
    run_tail_audit_from_frames,
)


def _hidden_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a_0", "a_1", "b_0", "b_1", "c_0", "c_1"],
            "well_id": ["a", "a", "b", "b", "c", "c"],
            "row_idx": [0, 1, 0, 1, 0, 1],
            "step": [0, 1, 0, 1, 0, 1],
            "TVT": [10.0, 20.0, 100.0, 110.0, 50.0, 60.0],
            "TVT_input": [np.nan] * 6,
            "GR": [1.0, 2.0, 3.0, np.nan, 10.0, 100.0],
            "base_tvt": [16.0, 26.0, 100.0, 110.0, 70.0, 80.0],
            "b2_tvt": [15.0, 25.0, 101.0, 111.0, 72.0, 82.0],
        }
    )


def _candidate_rows() -> pd.DataFrame:
    rows = []
    for candidate, preds in {
        "mtp_track_weighted": {
            "a": [15.0, 25.0],
            "b": [100.5, 110.5],
            "c": [70.0, 80.0],
        },
        "mtp_row_oracle": {
            "a": [10.0, 20.0],
            "b": [100.0, 110.0],
            "c": [69.0, 79.0],
        },
    }.items():
        for well_id, values in preds.items():
            for index, value in enumerate(values):
                rows.append(
                    {
                        "id": f"{well_id}_{index}",
                        "well_id": well_id,
                        "row_idx": index,
                        "step": index,
                        "pred_tvt": value,
                        "candidate": candidate,
                    }
                )
    return pd.DataFrame(rows)


def test_build_tail_audit_finds_selector_fail_and_all_candidates_bad() -> None:
    audit = build_tail_audit(
        hidden_rows=_hidden_rows(),
        candidate_rows=_candidate_rows(),
        primary_candidate="mtp_track_weighted",
        top_n=3,
    )

    by_well = audit.set_index("well_id")
    assert by_well.loc["a", "rmse_b2"] == pytest.approx(5.0)
    assert by_well.loc["a", "rmse__mtp_track_weighted"] == pytest.approx(5.0)
    assert by_well.loc["a", "best_candidate_oracle_rmse"] == pytest.approx(0.0)
    assert by_well.loc["a", "candidate_exists_selector_fails"] is True
    assert by_well.loc["a", "tail_class"] == "H_candidate_exists_selector_fails"

    assert by_well.loc["c", "all_candidates_bad"] is True
    assert by_well.loc["c", "tail_class"] == "G_all_candidates_fail"
    assert by_well.loc["c", "GR_volatility"] > by_well.loc["a", "GR_volatility"]


def test_classify_tail_row_prefers_explicit_failure_classes() -> None:
    assert (
        classify_tail_row(
            {
                "candidate_exists_selector_fails": True,
                "all_candidates_bad": True,
            }
        )
        == "G_all_candidates_fail"
    )
    assert (
        classify_tail_row(
            {
                "candidate_exists_selector_fails": True,
                "all_candidates_bad": False,
            }
        )
        == "H_candidate_exists_selector_fails"
    )


def test_run_tail_audit_from_frames_writes_artifacts(tmp_path: Path) -> None:
    summary = run_tail_audit_from_frames(
        hidden_rows=_hidden_rows(),
        candidate_rows=_candidate_rows(),
        output_dir=tmp_path,
        primary_candidate="mtp_track_weighted",
        top_n=2,
    )

    assert summary["wells"] == 3
    assert summary["top_worst_wells"][0]["well_id"] == "c"
    assert (tmp_path / "well_tail_audit.csv").exists()
    assert (tmp_path / "tail_candidate_summary.csv").exists()
    assert (tmp_path / "tail_audit_metrics.json").exists()
    report = (tmp_path / "tail_audit_report.md").read_text(encoding="utf-8")
    assert "GEOMTP_TAIL_AUDIT" in report
    assert "H_candidate_exists_selector_fails" in report
