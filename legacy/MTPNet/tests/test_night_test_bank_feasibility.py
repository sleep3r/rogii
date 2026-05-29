from pathlib import Path

import pandas as pd

from mtpnet.night_test_bank_feasibility import NightTestBankFeasibilityConfig, run_test_bank_feasibility


def test_run_test_bank_feasibility_marks_null_and_deployable_candidates(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "path_col": ["p0__candidate_selector_v0", "p1__grid_oracle", "p2__tto_shuffled_gr"],
            "experiment": ["candidate_selector_v0", "discrete_offset", "offset_tto"],
            "candidate": ["candidate_selector_v0", "grid_oracle", "tto_shuffled_gr"],
            "coverage_frac": [1.0, 1.0, 1.0],
            "pooled_rmse": [10.0, 7.0, 16.0],
        }
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)
    metrics = run_test_bank_feasibility(
        NightTestBankFeasibilityConfig(
            candidate_summary_path=summary_path,
            output_dir=tmp_path / "night",
            mission_path=tmp_path / "missing.md",
        )
    )

    assert metrics["deployable_candidates"] == 1
    out = pd.read_csv(tmp_path / "night" / "path_bank_test_feasibility.csv")
    assert out.loc[out["candidate"] == "candidate_selector_v0", "deployable"].iloc[0]
    assert not out.loc[out["candidate"] == "grid_oracle", "deployable"].iloc[0]
    assert not out.loc[out["candidate"] == "tto_shuffled_gr", "deployable"].iloc[0]
