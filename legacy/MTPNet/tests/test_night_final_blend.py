from pathlib import Path

import numpy as np
import pandas as pd

from mtpnet.night_final_blend import NightFinalBlendConfig, run_final_blend


def test_run_final_blend_writes_grid_and_oof_predictions(tmp_path: Path) -> None:
    bank = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "well_id": ["a", "a", "b", "b"],
            "row_idx": [0, 1, 0, 1],
            "true_tvt": [1.0, 2.0, 3.0, 4.0],
            "hidden_len": [2, 2, 2, 2],
            "hidden_len_bucket": ["short"] * 4,
            "p0": [1.0, 2.0, 6.0, 8.0],
            "p1": [0.0, 0.0, 3.0, 4.0],
        }
    )
    bank_path = tmp_path / "path_bank_oof.parquet"
    bank.to_parquet(bank_path, index=False)
    summary = pd.DataFrame(
        {
            "path_col": ["p0", "p1"],
            "pooled_rmse": [2.5, 1.1],
            "coverage_frac": [1.0, 1.0],
        }
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    metrics = run_final_blend(
        NightFinalBlendConfig(
            path_bank_path=bank_path,
            candidate_summary_path=summary_path,
            output_dir=tmp_path / "night",
            mission_path=tmp_path / "missing.md",
            max_candidates=2,
        )
    )

    assert metrics["candidate_count"] == 2
    assert np.isfinite(metrics["best_blend_rmse"])
    assert (tmp_path / "night" / "final_blend_grid.csv").exists()
    assert (tmp_path / "night" / "final_oof_predictions.parquet").exists()
    grid = pd.read_csv(tmp_path / "night" / "final_blend_grid.csv")
    assert "optimized_convex_blend" in set(grid["candidate"])
