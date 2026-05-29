from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.formation_correlation import (
    FormationCorrelationConfig,
    build_well_formation_correlation,
    infer_allowed_geology_from_surfaces,
    run_formation_correlation_audit,
)


def test_oracle_geology_band_removes_repeated_pattern_false_match() -> None:
    tvt = np.arange(12, dtype=np.float32) * 10.0
    horizontal_gr = np.array([0, 1, 2, 8, 9, 10, 50, 51, 52, 8, 9, 10], dtype=np.float32)
    typewell_gr = np.array([0, 1, 2, 50, 51, 52, 50, 51, 52, 8, 9, 10], dtype=np.float32)
    geology = ["A"] * 6 + ["B"] * 6
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": horizontal_gr})
    typewell = pd.DataFrame({"TVT": tvt, "GR": typewell_gr, "Geology": geology})

    result = build_well_formation_correlation(
        "well_a",
        horizontal,
        typewell,
        FormationCorrelationConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1,),
            min_patch_points=2,
        ),
        shuffle_gr_seed=123,
    )

    global_step = result.steps[
        (result.steps["variant"] == "global") & (result.steps["step"] == 7)
    ].iloc[0]
    oracle_step = result.steps[
        (result.steps["variant"] == "oracle_geology") & (result.steps["step"] == 7)
    ].iloc[0]
    assert global_step["candidate_bins"] > oracle_step["candidate_bins"]
    assert oracle_step["top1_geology"] == "B"
    assert oracle_step["candidate_bins"] > 0
    assert result.metrics["oracle_geology"]["candidate_bins_mean"] < result.metrics["global"]["candidate_bins_mean"]


def test_anchor_geology_reports_zone_match_rate() -> None:
    tvt = np.arange(10, dtype=np.float32) * 10.0
    gr = np.array([1, 2, 3, 4, 8, 9, 10, 11, 12, 13], dtype=np.float32)
    geology = ["A"] * 5 + ["B"] * 5
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr})
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": geology})

    result = build_well_formation_correlation(
        "well_a",
        horizontal,
        typewell,
        FormationCorrelationConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1,),
            min_patch_points=2,
        ),
    )

    assert "anchor_geology" in result.metrics
    assert result.metrics["anchor_geology"]["anchor_geology_match_rate"] == pytest.approx(1.0)


def test_surface_geology_allows_deployable_coarse_subzones() -> None:
    horizontal = pd.DataFrame(
        {
            "Z": [-101.0, -151.0, -251.0, -351.0, -451.0, -575.0, -710.0],
            "ANCC": [-100.0] * 7,
            "ASTNU": [-200.0] * 7,
            "ASTNL": [-300.0] * 7,
            "EGFDU": [-400.0] * 7,
            "EGFDL": [-500.0] * 7,
            "BUDA": [-700.0] * 7,
        }
    )

    allowed = infer_allowed_geology_from_surfaces(horizontal)

    assert allowed[0] == ("ANCC",)
    assert allowed[1] == ("ANCC",)
    assert allowed[2] == ("ASTNU",)
    assert allowed[3] == ("ASTNL",)
    assert allowed[4] == ("EGFDU",)
    assert set(allowed[5]) == {"EGFDL", "LTHL", "LTGT", "LBHL", "MNSS"}
    assert allowed[6] == ("BUDA",)


def test_surface_geology_variant_uses_horizontal_surfaces() -> None:
    tvt = np.arange(12, dtype=np.float32) * 10.0
    gr = np.array([0, 1, 2, 4, 8, 9, 50, 51, 52, 20, 19, 18], dtype=np.float32)
    geology = ["ANCC"] * 4 + ["ASTNU"] * 2 + ["EGFDL"] * 3 + ["BUDA"] * 3
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan
    horizontal = pd.DataFrame(
        {
            "TVT": tvt,
            "TVT_input": tvt_input,
            "GR": gr,
            "Z": [-110, -120, -130, -140, -250, -260, -560, -570, -580, -720, -730, -740],
            "ANCC": [-100] * 12,
            "ASTNU": [-200] * 12,
            "ASTNL": [-300] * 12,
            "EGFDU": [-400] * 12,
            "EGFDL": [-500] * 12,
            "BUDA": [-700] * 12,
        }
    )
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": geology})

    result = build_well_formation_correlation(
        "well_a",
        horizontal,
        typewell,
        FormationCorrelationConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1,),
            min_patch_points=2,
        ),
    )

    assert "surface_geology" in result.metrics
    assert result.metrics["surface_geology"]["surface_geology_match_rate"] == pytest.approx(1.0)
    assert result.metrics["surface_geology"]["candidate_bins_mean"] < result.metrics["global"]["candidate_bins_mean"]


def test_run_formation_correlation_audit_writes_artifacts(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    tvt = np.arange(10, dtype=np.float32) * 5.0
    gr = np.sin(np.arange(10, dtype=np.float32)) * 20.0 + 40.0
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    geology = ["A"] * 5 + ["B"] * 5
    pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr}).to_csv(
        train / "well_a__horizontal_well.csv", index=False
    )
    pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": geology}).to_csv(
        train / "well_a__typewell.csv", index=False
    )

    output = tmp_path / "formation"
    run_formation_correlation_audit(
        data_dir=tmp_path / "data",
        output_dir=output,
        rows_per_step=1,
        vertical_step_ft=5.0,
        patch_radii=(1, 2),
        k_wells=-1,
    )

    metrics = json.loads((output / "formation_correlation_metrics.json").read_text())
    assert metrics["global"]["num_wells"] == 1
    assert metrics["oracle_geology"]["num_wells"] == 1
    assert (output / "formation_correlation_by_well.csv").exists()
    assert (output / "formation_correlation_steps.parquet").exists()
    assert (output / "formation_correlation_report.md").exists()
