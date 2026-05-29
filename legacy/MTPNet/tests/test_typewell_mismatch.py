from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.typewell_mismatch import (
    TypewellMismatchConfig,
    build_well_typewell_mismatch,
    run_typewell_mismatch_audit,
)


def test_true_path_match_is_stronger_than_shuffled_for_clean_typewell() -> None:
    tvt = np.arange(12, dtype=np.float32) * 10.0
    gr = np.array([0, 1, 3, 8, 20, 35, 50, 34, 18, 7, 3, 1], dtype=np.float32)
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr})
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr})

    result = build_well_typewell_mismatch(
        "well_a",
        horizontal,
        typewell,
        TypewellMismatchConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1, 2),
            min_patch_points=2,
        ),
        shuffle_gr_seed=123,
    )

    assert result.metrics["num_steps"] == 7
    assert result.metrics["true_rank_mean"] == pytest.approx(1.0)
    assert result.metrics["true_top10_rate"] == pytest.approx(1.0)
    assert result.metrics["true_score_better_than_shuffled_rate"] > 0.7
    assert result.metrics["mismatch_flag"] == 0
    assert {"true_score", "shuffled_true_score", "score_gap_vs_shuffled"}.issubset(
        result.steps.columns
    )


def test_typewell_mismatch_flags_bad_true_path_match() -> None:
    tvt = np.arange(20, dtype=np.float32) * 10.0
    horizontal_gr = np.array(
        [1, 2, 4, 8, 12, 18, 24, 30, 45, 60, 44, 22, 10, 4, 2, 1, 3, 5, 7, 9],
        dtype=np.float32,
    )
    typewell_gr = np.roll(horizontal_gr, -7).astype(np.float32)
    tvt_input = tvt.copy()
    tvt_input[8:] = np.nan
    horizontal = pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": horizontal_gr})
    typewell = pd.DataFrame({"TVT": tvt, "GR": typewell_gr})

    result = build_well_typewell_mismatch(
        "well_bad",
        horizontal,
        typewell,
        TypewellMismatchConfig(
            rows_per_step=1,
            vertical_step_ft=10.0,
            patch_radii=(1, 2),
            min_patch_points=2,
            mismatch_topk=3,
            mismatch_rate_threshold=0.6,
        ),
        shuffle_gr_seed=123,
    )

    assert result.metrics["true_top3_rate"] < 0.6
    assert result.metrics["mismatch_flag"] == 1
    assert result.metrics["score_margin_mean"] > 0.0


def test_run_typewell_mismatch_audit_writes_artifacts(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    tvt = np.arange(10, dtype=np.float32) * 5.0
    gr = np.sin(np.arange(10, dtype=np.float32)) * 20.0 + 40.0
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr}).to_csv(
        train / "well_a__horizontal_well.csv", index=False
    )
    pd.DataFrame({"TVT": tvt, "GR": gr}).to_csv(
        train / "well_a__typewell.csv", index=False
    )

    output = tmp_path / "mismatch"
    run_typewell_mismatch_audit(
        data_dir=tmp_path / "data",
        output_dir=output,
        rows_per_step=1,
        vertical_step_ft=5.0,
        patch_radii=(1, 2),
        k_wells=-1,
    )

    metrics = json.loads((output / "typewell_mismatch_metrics.json").read_text())
    assert metrics["aggregate"]["num_wells"] == 1
    assert metrics["aggregate"]["true_top10_rate"] == pytest.approx(1.0)
    assert (output / "typewell_mismatch_by_well.csv").exists()
    assert (output / "typewell_mismatch_steps.parquet").exists()
    assert (output / "typewell_mismatch_report.md").exists()
