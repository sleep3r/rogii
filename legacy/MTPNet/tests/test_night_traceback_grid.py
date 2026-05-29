from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_traceback_grid_collects_variant_metrics_and_corridors(tmp_path: Path) -> None:
    from mtpnet.night_traceback_grid import NightTracebackGridConfig, run_traceback_grid

    src = tmp_path / "traceback_v1"
    src.mkdir()
    (src / "traceback_metrics.json").write_text(
        json.dumps(
            {
                "wells": 2,
                "variants": {"normal_GR": {"event_top1_rmse_ft": 10.0}},
                "candidate_oracle": {"traceback_oracle_row_rmse": 5.0},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"well_id": ["w0"], "step": [0], "band_center_tvt": [100.0]}).to_parquet(
        src / "traceback_bands.parquet",
        index=False,
    )

    result = run_traceback_grid(
        NightTracebackGridConfig(source_dirs=[src], output_dir=tmp_path / "night")
    )

    assert result["runs"] == 1
    assert (tmp_path / "night" / "traceback_anchor_grid.csv").exists()
    assert (tmp_path / "night" / "traceback_corridors.parquet").exists()
