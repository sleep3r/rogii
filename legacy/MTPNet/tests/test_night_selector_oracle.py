from pathlib import Path

import numpy as np
import pandas as pd

from mtpnet.night_selector_oracle import NightSelectorOracleConfig, run_selector_oracle


def test_run_selector_oracle_computes_whole_chunk_and_row_oracles(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["a"] * 4 + ["b"] * 4,
            "row_idx": list(range(4)) + list(range(4)),
            "true_tvt": [0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0],
            "p0": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
            "p1": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    bank_path = tmp_path / "bank.parquet"
    frame.to_parquet(bank_path, index=False)
    summary = pd.DataFrame({"path_col": ["p0", "p1"], "coverage_frac": [1.0, 1.0], "pooled_rmse": [5.0, 5.0]})
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    metrics = run_selector_oracle(
        NightSelectorOracleConfig(
            path_bank_path=bank_path,
            candidate_summary_path=summary_path,
            output_dir=tmp_path / "night",
            mission_path=tmp_path / "missing.md",
            chunk_sizes=(2,),
            max_candidates=2,
        )
    )

    assert metrics["candidate_count"] == 2
    table = pd.read_csv(tmp_path / "night" / "selector_oracle_granularity.csv")
    assert set(table["granularity"]) >= {"row_oracle", "whole_well_oracle", "chunk_2_oracle"}
    row_rmse = float(table.loc[table["granularity"] == "row_oracle", "row_rmse"].iloc[0])
    chunk_rmse = float(table.loc[table["granularity"] == "chunk_2_oracle", "row_rmse"].iloc[0])
    whole_rmse = float(table.loc[table["granularity"] == "whole_well_oracle", "row_rmse"].iloc[0])
    assert np.isclose(row_rmse, 0.0)
    assert np.isclose(chunk_rmse, 0.0)
    assert whole_rmse > 0.0
    assert (tmp_path / "night" / "selector_oracle_report.md").exists()
