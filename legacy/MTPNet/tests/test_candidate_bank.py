from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _hidden_rows() -> pd.DataFrame:
    rows = []
    for row_idx in range(6):
        is_hidden = row_idx >= 2
        tvt = 100.0 + row_idx * 2.0
        rows.append(
            {
                "id": f"a_{row_idx}",
                "well_id": "a",
                "row_idx": row_idx,
                "step": row_idx,
                "TVT": tvt,
                "TVT_input": np.nan if is_hidden else tvt,
                "GR": 10.0 + row_idx,
                "b2_tvt": tvt - 5.0,
                "base_tvt": tvt - 7.0,
                "a_p50_tvt": tvt + 3.0,
            }
        )
    return pd.DataFrame(rows)


def test_candidate_bank_generates_shift_drift_slope_and_residual_candidates() -> None:
    from mtpnet.candidate_bank import build_candidate_bank_from_frames

    residual = _hidden_rows()[["id", "well_id", "row_idx", "step", "TVT"]].copy()
    residual["pred_tvt"] = residual["TVT"] - 1.0
    residual = residual[residual["row_idx"] >= 2]

    bank = build_candidate_bank_from_frames(_hidden_rows(), residual_predictions=residual)

    candidates = set(bank["candidate"].unique())
    assert "residual_stack_v0" in candidates
    assert "b2_shift+20" in candidates
    assert "b2_drift_end+40" in candidates
    assert "slope+0" in candidates
    assert bank.loc[bank["candidate"] == "residual_stack_v0", "pred_tvt"].notna().all()


def test_candidate_bank_can_add_soft_segment_step_candidates() -> None:
    from mtpnet.candidate_bank import build_candidate_bank_from_frames

    step_predictions = pd.DataFrame(
        {
            "well_id": ["a", "a", "a", "a"],
            "compressed_step": [2, 3, 4, 5],
            "top1_tvt": [104.0, 106.0, 108.0, 110.0],
            "dp_tvt": [103.0, 105.0, 107.0, 109.0],
            "true_tvt": [999.0, 999.0, 999.0, 999.0],
            "target_rank": [1, 1, 1, 1],
        }
    )

    bank = build_candidate_bank_from_frames(
        _hidden_rows(),
        step_predictions=step_predictions,
    )

    candidates = set(bank["candidate"].unique())
    assert "softseg_top1" in candidates
    assert "softseg_dp" in candidates
    assert "true_tvt" not in bank.columns
    assert "target_rank" not in bank.columns
    assert bank.loc[bank["candidate"] == "softseg_top1", "pred_tvt"].notna().all()


def test_candidate_bank_can_add_traceback_candidates_without_target_leakage() -> None:
    from mtpnet.candidate_bank import build_candidate_bank_from_frames

    traceback = pd.DataFrame(
        {
            "id": ["a_2", "a_3"],
            "well_id": ["a", "a"],
            "row_idx": [2, 3],
            "candidate": ["traceback_band_w80", "traceback_band_w80"],
            "pred_tvt": [104.0, 106.0],
            "TVT": [999.0, 999.0],
            "true_TVT": [999.0, 999.0],
        }
    )

    bank = build_candidate_bank_from_frames(
        _hidden_rows().drop(columns=["TVT"]),
        traceback_candidates=traceback,
    )

    tb = bank[bank["candidate"].astype(str) == "traceback_band_w80"]
    assert tb["id"].tolist() == ["a_2", "a_3"]
    assert tb["pred_tvt"].tolist() == [104.0, 106.0]
    assert "TVT" not in bank.columns
    assert "true_TVT" not in bank.columns


def test_candidate_bank_generation_does_not_require_true_tvt_for_paths() -> None:
    from mtpnet.candidate_bank import build_candidate_bank_from_frames

    hidden = _hidden_rows()
    no_target = hidden.drop(columns=["TVT"])

    bank = build_candidate_bank_from_frames(no_target)

    assert "TVT" not in bank.columns
    assert {"candidate", "pred_tvt", "id", "well_id", "row_idx"}.issubset(bank.columns)


def test_candidate_bank_smoke_writes_tail_audit_artifacts(tmp_path: Path) -> None:
    from mtpnet.candidate_bank import run_candidate_bank_from_frames

    residual = _hidden_rows()[["id", "well_id", "row_idx", "step", "TVT"]].copy()
    residual["pred_tvt"] = residual["TVT"] - 1.0
    residual = residual[residual["row_idx"] >= 2]

    metrics = run_candidate_bank_from_frames(
        _hidden_rows(),
        output_dir=tmp_path,
        residual_predictions=residual,
        primary_candidate="residual_stack_v0",
    )

    assert metrics["candidates"] >= 1
    assert (tmp_path / "candidate_bank.parquet").exists()
    assert (tmp_path / "tail_audit_report.md").exists()


def test_candidate_bank_streaming_oracle_writes_wide_artifacts(tmp_path: Path) -> None:
    from mtpnet.candidate_bank import run_candidate_bank_oracle_from_frames

    residual = _hidden_rows()[["id", "well_id", "row_idx", "step", "TVT"]].copy()
    residual["pred_tvt"] = residual["TVT"] - 1.0
    residual = residual[residual["row_idx"] >= 2]

    metrics = run_candidate_bank_oracle_from_frames(
        _hidden_rows(),
        output_dir=tmp_path,
        residual_predictions=residual,
        primary_candidate="residual_stack_v0",
    )

    assert metrics["wells"] == 1
    assert metrics["candidates"] >= 1
    assert metrics["diagnostics"]["all_candidates_bad_wells"] == 0
    assert metrics["oracle"]["mean_well_rmse"] < metrics["b2"]["mean_well_rmse"]
    assert (tmp_path / "candidate_bank_oracle_wells.csv").exists()
    assert (tmp_path / "candidate_bank_oracle_candidate_summary.csv").exists()
    assert (tmp_path / "candidate_bank_oracle_metrics.json").exists()
    assert (tmp_path / "candidate_bank_oracle_report.md").exists()


def test_candidate_bank_streaming_oracle_does_not_write_long_bank(tmp_path: Path) -> None:
    from mtpnet.candidate_bank import run_candidate_bank_oracle_from_frames

    run_candidate_bank_oracle_from_frames(
        _hidden_rows(),
        output_dir=tmp_path,
        residual_predictions=None,
    )

    assert not (tmp_path / "candidate_bank.parquet").exists()
