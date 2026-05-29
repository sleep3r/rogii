from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.correlation_panel import (
    CorrelationPanelConfig,
    known_tail_linear_anchor,
    build_well_correlation_panel,
    run_correlation_panel,
    score_gr_patch_against_typewell,
    score_gr_patch_multiscale,
)


def test_patch_score_ranks_matching_typewell_patch_highest() -> None:
    horizontal_gr = np.array([0.0, 10.0, 11.0, 12.0, 4.0], dtype=np.float32)
    typewell_gr = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 50.0], dtype=np.float32)

    scores = score_gr_patch_against_typewell(
        horizontal_gr,
        typewell_gr,
        step_index=2,
        patch_radius=1,
    )

    assert int(np.nanargmax(scores)) == 4
    assert scores[4] > scores[1]


def test_well_correlation_panel_reports_true_path_in_top1_for_clean_match() -> None:
    tvt = np.arange(8, dtype=np.float32) * 10.0
    gr = np.array([2.0, 4.0, 8.0, 20.0, 35.0, 21.0, 9.0, 5.0], dtype=np.float32)
    tvt_input = tvt.copy()
    tvt_input[3:] = np.nan
    horizontal = pd.DataFrame(
        {
            "id": [f"well_a_{i}" for i in range(len(tvt))],
            "TVT": tvt,
            "TVT_input": tvt_input,
            "GR": gr,
        }
    )
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr})

    result = build_well_correlation_panel(
        "well_a",
        horizontal,
        typewell,
        CorrelationPanelConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radius=1,
            min_patch_points=2,
        ),
    )

    assert result.metrics["num_steps"] == 5
    assert result.metrics["corr_top1_rmse_ft"] == pytest.approx(0.0)
    assert result.metrics["corr_target_top1_rate"] == pytest.approx(1.0)
    assert result.steps["true_rank"].max() == 1


def test_known_tail_linear_anchor_extrapolates_last_known_slope() -> None:
    tvt_input = np.array([100.0, 105.0, 111.0, np.nan, np.nan], dtype=np.float32)

    anchor = known_tail_linear_anchor(tvt_input)

    assert anchor.tolist() == pytest.approx([100.0, 105.0, 111.0, 117.0, 123.0])


def test_localized_search_band_excludes_far_better_false_match() -> None:
    tvt = np.arange(10, dtype=np.float32) * 10.0
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    horizontal_gr = np.array([0, 1, 2, 10, 20, 30, 31, 32, 6, 7], dtype=np.float32)
    typewell_gr = np.array([30, 31, 32, 1, 2, 3, 30, 31, 32, 99], dtype=np.float32)
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": horizontal_gr})
    typewell = pd.DataFrame({"TVT": tvt, "GR": typewell_gr})

    global_result = build_well_correlation_panel(
        "well_a",
        horizontal,
        typewell,
        CorrelationPanelConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radius=1,
            min_patch_points=2,
        ),
    )
    localized_result = build_well_correlation_panel(
        "well_a",
        horizontal,
        typewell,
        CorrelationPanelConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radius=1,
            min_patch_points=2,
            anchor_source="known_tail_linear",
            search_radius_ft=5.0,
        ),
    )

    global_step = global_result.steps.loc[global_result.steps["step"] == 6].iloc[0]
    localized_step = localized_result.steps.loc[localized_result.steps["step"] == 6].iloc[0]
    assert abs(global_step["top1_tvt"] - global_step["true_tvt"]) > 5.0
    assert localized_step["top1_tvt"] == pytest.approx(localized_step["true_tvt"])


def test_multiscale_panel_records_best_patch_radius() -> None:
    tvt = np.arange(12, dtype=np.float32) * 10.0
    gr = np.array([1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 4, 3], dtype=np.float32)
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr})
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr})

    result = build_well_correlation_panel(
        "well_a",
        horizontal,
        typewell,
        CorrelationPanelConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1, 2),
            min_patch_points=2,
        ),
    )

    assert set(result.steps["best_patch_radius"].dropna().unique()).issubset({1, 2})


def test_stretch_factor_recovers_stretched_typewell_match() -> None:
    horizontal_gr = np.array([0.0, 10.0, 20.0, 30.0, 0.0], dtype=np.float32)
    typewell_gr = np.zeros(11, dtype=np.float32)
    typewell_gr[[3, 5, 7]] = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    typewell_gr[[4, 5, 6]] = np.array([10.0, 20.0, 5.0], dtype=np.float32)

    scores_no_stretch, _ = score_gr_patch_multiscale(
        horizontal_gr,
        typewell_gr,
        step_index=2,
        patch_radii=(1,),
        stretch_factors=(1.0,),
        min_patch_points=3,
    )
    scores_with_stretch, best = score_gr_patch_multiscale(
        horizontal_gr,
        typewell_gr,
        step_index=2,
        patch_radii=(1,),
        stretch_factors=(1.0, 2.0),
        min_patch_points=3,
    )

    assert int(np.nanargmax(scores_no_stretch)) != 5
    assert int(np.nanargmax(scores_with_stretch)) == 5
    assert best == (1, 2.0)


def test_run_correlation_panel_writes_metrics_and_step_artifacts(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    tvt = np.arange(10, dtype=np.float32) * 5.0
    gr = np.sin(np.arange(10, dtype=np.float32)) * 20.0 + 40.0
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    pd.DataFrame(
        {
            "id": [f"well_a_{i}" for i in range(len(tvt))],
            "TVT": tvt,
            "TVT_input": tvt_input,
            "GR": gr,
        }
    ).to_csv(train / "well_a__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": tvt, "GR": gr}).to_csv(
        train / "well_a__typewell.csv", index=False
    )

    output = tmp_path / "corr"
    run_correlation_panel(
        data_dir=tmp_path / "data",
        output_dir=output,
        rows_per_step=1,
        vertical_step_ft=5.0,
        patch_radius=1,
        k_wells=-1,
        include_shuffled=False,
    )

    metrics = json.loads((output / "correlation_panel_metrics.json").read_text())
    assert metrics["normal"]["num_wells"] == 1
    assert metrics["normal"]["corr_top1_rmse_ft"] == pytest.approx(0.0)
    assert (output / "correlation_panel_by_well.csv").exists()
    assert (output / "correlation_panel_steps.parquet").exists()
    assert (output / "correlation_panel_report.md").exists()
