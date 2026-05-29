from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_collect_beam_grid_reads_metrics_and_predictions(tmp_path: Path) -> None:
    from mtpnet.night_beam_grid import NightBeamGridConfig, run_beam_grid

    run_dir = tmp_path / "artifacts" / "lattice_offset_v0_beam"
    run_dir.mkdir(parents=True)
    metrics = {
        "experiment": "lattice_offset_v0",
        "config": {"beam_size": 8, "branch_top_k": 3, "k_wells": 2, "spans": "64,128"},
        "states": 4,
        "candidate_count_per_state": 6,
        "teacher_forced_metrics": [
            {"variant": "normal", "top1_oracle_rate": 0.5, "top3_oracle_rate": 0.75},
            {"variant": "shuffled_gr", "top1_oracle_rate": 0.25, "top3_oracle_rate": 0.5},
        ],
        "candidates": [
            {"candidate": "normal", "row_rmse": 10.0, "rows": 3, "wells": 1},
            {"candidate": "shuffled", "row_rmse": 12.0, "rows": 3, "wells": 1},
        ],
        "normal_minus_shuffled_row_rmse": -2.0,
    }
    (run_dir / "lattice_offset_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame(
        {
            "id": ["w0_1", "w0_2"],
            "well_id": ["w0", "w0"],
            "row_idx": [1, 2],
            "pred_tvt": [1.0, 2.0],
            "candidate": ["normal", "normal"],
        }
    ).to_parquet(run_dir / "lattice_offset_predictions.parquet", index=False)

    result = run_beam_grid(
        NightBeamGridConfig(
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "night",
        )
    )

    assert result["runs"] == 1
    grid = pd.read_csv(tmp_path / "night" / "beam_grid.csv")
    assert grid.iloc[0]["row_rmse"] == 10.0
    assert (tmp_path / "night" / "beam_top_predictions.parquet").exists()
