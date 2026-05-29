from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtpnet.true_path_gr_audit import run_true_path_gr_audit
from mtpnet.typewell_mismatch import TypewellMismatchConfig, build_well_typewell_mismatch


def _write_well(root: Path, well_id: str, *, shift: float = 0.0) -> None:
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    tvt = np.arange(18, dtype=np.float32) * 5.0 + shift
    gr = np.sin(np.arange(18, dtype=np.float32) / 2.0) * 20.0 + 50.0
    tvt_input = tvt.copy()
    tvt_input[8:] = np.nan
    geology = ["ANCC"] * 5 + ["ASTNU"] * 5 + ["EGFDL"] * 5 + ["BUDA"] * 3
    horizontal = pd.DataFrame(
        {
            "id": [f"{well_id}_{idx}" for idx in range(len(tvt))],
            "MD": np.arange(len(tvt), dtype=np.float32),
            "X": np.linspace(0.0, 1.0, len(tvt), dtype=np.float32),
            "Y": np.linspace(1.0, 2.0, len(tvt), dtype=np.float32),
            "Z": -tvt,
            "GR": gr,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": geology})
    horizontal.to_csv(train / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(train / f"{well_id}__typewell.csv", index=False)


def test_true_path_steps_include_zone_and_true_best_patch_params() -> None:
    tvt = np.arange(12, dtype=np.float32) * 10.0
    gr = np.array([0, 1, 3, 8, 20, 35, 50, 34, 18, 7, 3, 1], dtype=np.float32)
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr})
    typewell = pd.DataFrame(
        {
            "TVT": tvt,
            "GR": gr,
            "Geology": ["ANCC"] * 4 + ["ASTNU"] * 4 + ["EGFDL"] * 4,
        }
    )

    result = build_well_typewell_mismatch(
        "well_a",
        horizontal,
        typewell,
        TypewellMismatchConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1, 2),
            stretch_factors=(1.0, 1.25),
            min_patch_points=2,
        ),
        shuffle_gr_seed=123,
    )

    assert {"true_zone", "true_best_patch_radius", "true_best_stretch_factor"}.issubset(
        result.steps.columns
    )
    assert result.steps["true_zone"].isin({"ASTNU", "EGFDL"}).all()
    assert np.isfinite(result.steps["true_best_patch_radius"]).all()
    assert np.isfinite(result.steps["true_best_stretch_factor"]).all()


def test_true_path_gr_audit_writes_report_metrics_and_figures(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_well(data, "well_a")
    _write_well(data, "well_b", shift=2.5)
    output = tmp_path / "audit"

    run_true_path_gr_audit(
        data_dir=data,
        output_dir=output,
        rows_per_step=1,
        vertical_step_ft=5.0,
        patch_radii=(1, 2),
        stretch_factors=(1.0,),
        k_wells=-1,
        seed=7,
    )

    metrics = json.loads((output / "true_path_gr_metrics.json").read_text())
    steps = pd.read_parquet(output / "true_path_gr_steps.parquet")
    by_well = pd.read_csv(output / "true_path_gr_by_well.csv")

    assert metrics["aggregate"]["num_wells"] == 2
    assert "by_zone" in metrics
    assert not steps.empty
    assert not by_well.empty
    assert (output / "true_path_gr_report.md").exists()
    assert (output / "figures" / "true_path_gap_distribution.png").exists()
    assert (output / "figures" / "true_rank_histogram.png").exists()
    assert (output / "figures" / "true_path_percentile_by_zone.png").exists()
