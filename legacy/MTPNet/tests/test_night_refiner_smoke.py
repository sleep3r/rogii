import json
from pathlib import Path

import pandas as pd

from mtpnet.night_refiner_smoke import NightRefinerSmokeConfig, run_refiner_smoke


def test_run_refiner_smoke_collects_tto_runs_and_copies_best_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "offset_tto_v0"
    run_dir.mkdir()
    (run_dir / "offset_tto_metrics.json").write_text(
        json.dumps(
            {
                "experiment": "offset_tto_v0",
                "wells": 2,
                "candidates": [
                    {"candidate": "tto_normal", "row_rmse": 9.0, "mean_well_rmse": 8.0},
                    {"candidate": "tto_shuffled_gr", "row_rmse": 10.0, "mean_well_rmse": 9.0},
                ],
                "normal_minus_best_null_rmse": -1.0,
            }
        )
    )
    pred = pd.DataFrame({"well_id": ["a"], "row_id": [1], "pred_tvt": [10.0]})
    pred.to_parquet(run_dir / "offset_tto_predictions.parquet", index=False)

    output_dir = tmp_path / "night"
    metrics = run_refiner_smoke(
        NightRefinerSmokeConfig(
            source_dirs=(run_dir,),
            output_dir=output_dir,
            mission_path=tmp_path / "missing.md",
        )
    )

    assert metrics["runs"] == 1
    assert metrics["best_row_rmse"] == 9.0
    assert (output_dir / "refiner_grid.csv").exists()
    assert (output_dir / "refiner_predictions.parquet").exists()
    grid = pd.read_csv(output_dir / "refiner_grid.csv")
    assert set(grid["candidate"]) == {"tto_normal", "tto_shuffled_gr"}
