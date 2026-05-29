from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_dictionary_grid_copies_summary_and_paths(tmp_path: Path) -> None:
    from mtpnet.night_dictionary_grid import NightDictionaryGridConfig, run_dictionary_grid

    src = tmp_path / "shared_typewell_neighbour_v0"
    src.mkdir()
    pd.DataFrame(
        {
            "candidate": ["neighbour_top1"],
            "mean_well_rmse": [12.0],
            "p95_well_rmse": [20.0],
            "worst_well_rmse": [30.0],
            "wells": [3],
        }
    ).to_csv(src / "shared_typewell_neighbour_candidate_summary.csv", index=False)
    (src / "shared_typewell_neighbour_metrics.json").write_text(
        json.dumps({"wells": 3, "oracle": {"mean": 10.0}, "anchor": {"mean": 40.0}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "id": ["w0_1"],
            "well_id": ["w0"],
            "row_idx": [1],
            "candidate": ["neighbour_top1"],
            "pred_tvt": [100.0],
        }
    ).to_parquet(src / "neighbour_candidate_predictions.parquet", index=False)

    result = run_dictionary_grid(
        NightDictionaryGridConfig(source_dir=src, output_dir=tmp_path / "night")
    )

    assert result["candidates"] == 1
    assert (tmp_path / "night" / "dictionary_grid.csv").exists()
    assert (tmp_path / "night" / "dictionary_paths.parquet").exists()
