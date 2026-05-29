from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.conditional_corr_panel import (
    ConditionalCorrConfig,
    build_well_conditional_corr_panel,
    location_prior_scores,
    run_conditional_corr_panel,
)


def test_location_prior_scores_prefer_bins_near_anchor() -> None:
    tvt_grid = np.arange(0.0, 101.0, 10.0, dtype=np.float32)

    scores = location_prior_scores(tvt_grid, anchor_tvt=42.0, sigma_ft=20.0)

    assert int(np.nanargmax(scores)) == 4
    assert scores[4] > scores[0]
    assert scores[4] > scores[9]


def test_gr_plus_location_records_increment_over_location_only() -> None:
    tvt = np.arange(16, dtype=np.float32) * 10.0
    gr = np.array(
        [4, 7, 9, 11, 13, 25, 45, 66, 48, 26, 14, 9, 5, 3, 2, 1],
        dtype=np.float32,
    )
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr})
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr})

    result = build_well_conditional_corr_panel(
        "well_a",
        horizontal,
        typewell,
        ConditionalCorrConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1, 2),
            min_patch_points=2,
            location_sigma_ft=18.0,
        ),
        seed=7,
    )

    assert {"gr_only", "location_only", "gr_location", "shuffled_gr_location"}.issubset(
        result.metrics
    )
    assert result.metrics["gr_location"]["corr_target_top1_rate"] == pytest.approx(1.0)
    assert (
        result.metrics["comparison"]["gr_location_vs_location_only_top1_gap_ft"]
        >= 0.0
    )
    assert set(result.steps["variant"].unique()).issuperset(
        {"gr_location", "shuffled_gr_location", "location_only"}
    )


def test_run_conditional_corr_panel_writes_artifacts(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    tvt = np.arange(12, dtype=np.float32) * 5.0
    gr = np.array([1, 4, 8, 14, 23, 41, 63, 43, 22, 13, 7, 2], dtype=np.float32)
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr}).to_csv(
        train / "well_a__horizontal_well.csv", index=False
    )
    pd.DataFrame({"TVT": tvt, "GR": gr}).to_csv(
        train / "well_a__typewell.csv", index=False
    )

    output = tmp_path / "conditional"
    metrics = run_conditional_corr_panel(
        data_dir=tmp_path / "data",
        output_dir=output,
        rows_per_step=1,
        vertical_step_ft=5.0,
        patch_radii=(1, 2),
        min_patch_points=2,
        k_wells=-1,
    )

    saved = json.loads((output / "conditional_corr_metrics.json").read_text())
    assert metrics["gr_location"]["num_wells"] == 1
    assert saved["gr_location"]["num_wells"] == 1
    assert "gr_location_vs_shuffled_top10_rate_gap" in saved["comparison"]
    assert (output / "conditional_corr_by_well.csv").exists()
    assert (output / "conditional_corr_steps.parquet").exists()
    assert (output / "conditional_corr_report.md").exists()
