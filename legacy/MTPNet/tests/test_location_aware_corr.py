from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _tiny_well(n: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    tvt = np.arange(n, dtype=np.float32) * 10.0 + 1000.0
    z = -0.35 * tvt + np.sin(np.arange(n, dtype=np.float32) / 3.0) * 12.0
    gr = np.sin(tvt / 20.0) * 18.0 + np.cos(tvt / 7.0) * 5.0 + 80.0
    tvt_input = tvt.copy()
    tvt_input[7:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(n, dtype=np.float32) * 30.0,
            "X": np.arange(n, dtype=np.float32),
            "Y": np.zeros(n, dtype=np.float32),
            "Z": z,
            "GR": gr,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr})
    return horizontal, typewell


def test_location_aware_score_breaks_repeated_gr_tie_by_nearest_location() -> None:
    from mtpnet.location_aware_corr import location_aware_score

    typewell_gr = np.array([10.0, 40.0, 10.0], dtype=np.float32)
    tvt_grid = np.array([0.0, 100.0, 200.0], dtype=np.float32)

    score = location_aware_score(
        typewell_gr=typewell_gr,
        horizontal_gr=10.0,
        tvt_grid=tvt_grid,
        query_tvt=200.0,
        location_sigma_ft=50.0,
        gr_weight=1.0,
        location_weight=1.0,
    )

    assert int(score.argmax()) == 2
    assert score[2] > score[0]


def test_location_aware_corr_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.location_aware_corr import LocationAwareCorrConfig, run_location_aware_corr

    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for well_id, shift in [("a", 0.0), ("b", 5.0), ("c", -5.0)]:
        horizontal, typewell = _tiny_well()
        horizontal["TVT"] += shift
        horizontal["TVT_input"] += shift
        typewell["TVT"] += shift
        horizontal.to_csv(train / f"{well_id}__horizontal_well.csv", index=False)
        typewell.to_csv(train / f"{well_id}__typewell.csv", index=False)

    output = tmp_path / "loc_corr"
    run_location_aware_corr(
        LocationAwareCorrConfig(
            data_dir=tmp_path / "data",
            output_dir=output,
            rows_per_step=1,
            vertical_step_ft=10.0,
            k_wells=-1,
            topk=(1, 3),
            seed=7,
        )
    )

    metrics = json.loads((output / "location_aware_corr_metrics.json").read_text())
    assert metrics["num_wells"] == 3
    assert metrics["value_location_raw"]["top3_rate"] == pytest.approx(1.0)
    assert (output / "location_aware_corr_steps.parquet").exists()
    assert (output / "location_aware_corr_report.md").exists()
    assert (output / "figures" / "example_location_aware_heatmap.png").exists()
