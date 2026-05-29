from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_scorer_pack_copies_candidate_table_and_predictions(tmp_path: Path) -> None:
    from mtpnet.night_scorer_pack import NightScorerPackConfig, run_scorer_pack

    src = tmp_path / "src"
    src.mkdir()
    pd.DataFrame({"well_id": ["w0"], "candidate_mse": [1.0], "pred_log_mse": [0.5]}).to_parquet(
        src / "scores.parquet",
        index=False,
    )
    pd.DataFrame({"id": ["w0_1"], "pred_tvt": [100.0]}).to_parquet(src / "pred.parquet", index=False)
    (src / "metrics.json").write_text(
        json.dumps({"candidates": [{"candidate": "selector", "row_rmse": 10.0}]}),
        encoding="utf-8",
    )

    result = run_scorer_pack(
        NightScorerPackConfig(
            candidate_scores_path=src / "scores.parquet",
            predictions_path=src / "pred.parquet",
            metrics_paths=[src / "metrics.json"],
            output_dir=tmp_path / "night",
        )
    )

    assert result["candidate_rows"] == 1
    assert (tmp_path / "night" / "candidate_table_r0.parquet").exists()
    assert (tmp_path / "night" / "scorer_oof_predictions.parquet").exists()
    assert (tmp_path / "night" / "scorer_grid.csv").exists()
