from pathlib import Path

import numpy as np
import pandas as pd

from mtpnet.night_chunk_selector import NightChunkSelectorConfig, build_chunk_candidate_table, run_chunk_selector


def test_build_chunk_candidate_table_has_labels_and_schema_safe_features(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["a"] * 4,
            "row_idx": [0, 1, 2, 3],
            "true_tvt": [0.0, 0.0, 10.0, 10.0],
            "MD": [0.0, 1.0, 2.0, 3.0],
            "Z": [0.0, -1.0, -2.0, -3.0],
            "GR": [100.0, 101.0, 120.0, 121.0],
            "p0": [0.0, 0.0, 0.0, 0.0],
            "p1": [10.0, 10.0, 10.0, 10.0],
        }
    )
    table = build_chunk_candidate_table(frame, ["p0", "p1"], chunk_size=2, beta_endpoint=0.3)

    assert len(table) == 4
    assert {"chunk_mse", "target_log_cost", "feat_pred_start", "feat_bank_median_abs_mean"}.issubset(table.columns)
    first_chunk = table[table["chunk_id"] == 0].sort_values("candidate_id")
    assert first_chunk.iloc[0]["chunk_mse"] == 0.0
    assert first_chunk.iloc[1]["chunk_mse"] == 100.0
    forbidden = {"TVT", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "Geology"}
    assert not forbidden.intersection(table.columns)


def test_run_chunk_selector_smoke_writes_predictions(tmp_path: Path) -> None:
    rows = []
    for well in range(4):
        for i in range(6):
            true = float(well + i)
            rows.append(
                {
                    "well_id": f"w{well}",
                    "row_idx": i,
                    "true_tvt": true,
                    "MD": float(i),
                    "Z": -float(i),
                    "GR": 100.0 + i,
                    "p0": true if well % 2 == 0 else true + 5.0,
                    "p1": true + 5.0 if well % 2 == 0 else true,
                }
            )
    bank = pd.DataFrame(rows)
    bank_path = tmp_path / "bank.parquet"
    bank.to_parquet(bank_path, index=False)
    summary = pd.DataFrame({"path_col": ["p0", "p1"], "coverage_frac": [1.0, 1.0], "pooled_rmse": [3.0, 3.0]})
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    metrics = run_chunk_selector(
        NightChunkSelectorConfig(
            path_bank_path=bank_path,
            candidate_summary_path=summary_path,
            output_dir=tmp_path / "night",
            mission_path=tmp_path / "missing.md",
            chunk_size=3,
            n_folds=2,
            max_candidates=2,
            iterations=5,
            learning_rate=0.1,
        )
    )

    assert metrics["candidate_count"] == 2
    assert np.isfinite(metrics["selector_row_rmse"])
    assert (tmp_path / "night" / "chunk_selector_table.parquet").exists()
    assert (tmp_path / "night" / "chunk_selector_oof_predictions.parquet").exists()
    assert (tmp_path / "night" / "chunk_selector_metrics.json").exists()
