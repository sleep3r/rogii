from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.direct_solver import build_submission_frames, sample_rows, solve_well


def test_sample_rows_sorts_indices_within_well() -> None:
    sample = pd.DataFrame({"id": ["well1_5", "well1_2", "well1_10", "well2_3"]})
    assert sample_rows(sample) == {"well1": [2, 5, 10], "well2": [3]}


EXPECTED_VARIANTS = {
    "linear_tailfit",
    "geo_tailfit",
    "stage1_raw",
    "stage12_raw",
    "cem_raw",
    "cem_top_median",
}


def _synthetic_well(seed: int = 11) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.default_rng(seed)
    n = 96
    idx = np.arange(n, dtype=float)
    tvt = 1000.0 + 0.08 * idx
    tvt_input = tvt.copy()
    tvt_input[60:] = np.nan
    df = pd.DataFrame(
        {
            "MD": idx,
            "X": idx * 0.1,
            "Y": idx * 0.05,
            "Z": 2000.0 - 0.04 * idx,
            "GR": 80.0 + np.sin(idx / 6.0) + rng.normal(scale=0.2, size=n),
            "TVT_input": tvt_input,
            "TVT": tvt,
            "ANCC": 2000.0 - 0.04 * idx - (tvt - 990.0),
            "ASTNU": 2000.0 - 0.04 * idx - (tvt - 985.0),
            "ASTNL": 2000.0 - 0.04 * idx - (tvt - 980.0),
            "EGFDU": 2000.0 - 0.04 * idx - (tvt - 975.0),
            "EGFDL": 2000.0 - 0.04 * idx - (tvt - 970.0),
            "BUDA": 2000.0 - 0.04 * idx - (tvt - 965.0),
        }
    )
    rows = list(range(60, n))
    return df, rows


def test_solve_well_produces_simplified_variants() -> None:
    df, rows = _synthetic_well()
    outputs, diagnostics = solve_well(
        "abc12345",
        df,
        rows,
        anchor_by_id={},
        train_eval=True,
        tail_rows=32,
    )
    n = len(df)
    assert EXPECTED_VARIANTS <= set(outputs)
    # The simplification removed the gated safe/bold families and the consensus
    # blends; only raw families and optional anchor blends are produced now.
    forbidden = {"gr_safe", "geo_safe", "gr_bold", "geo_bold",
                 "consensus_safe", "consensus_bold", "consensus_bold_family"}
    assert forbidden.isdisjoint(outputs)
    assert len(diagnostics) >= len(EXPECTED_VARIANTS)
    for name, values in outputs.items():
        assert len(values) == n, name
        assert np.isfinite(values[rows]).all(), name


def test_solve_well_emits_anchor_blends_only_when_anchor_present() -> None:
    df, rows = _synthetic_well(seed=17)
    tvt = df["TVT"].to_numpy(float)
    anchor = {f"abc12345_{i}": float(tvt[i]) for i in rows}
    outputs, _ = solve_well(
        "abc12345", df, rows, anchor_by_id=anchor, train_eval=True, tail_rows=32,
    )
    assert "cem_top_median_anchor_blend40" in outputs
    assert "stage12_raw_anchor_blend60" in outputs

    outputs_no_anchor, _ = solve_well(
        "abc12345", df, rows, anchor_by_id={}, train_eval=True, tail_rows=32,
    )
    assert "cem_top_median_anchor_blend40" not in outputs_no_anchor
    assert "stage12_raw_anchor_blend60" not in outputs_no_anchor


def test_build_submission_frames_train_eval(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    n = 90
    idx = np.arange(n, dtype=float)
    tvt = 1100.0 + 0.05 * idx
    tvt_input = tvt.copy()
    tvt_input[70:] = np.nan
    well = "deadbeef"
    pd.DataFrame(
        {
            "MD": idx,
            "X": idx,
            "Y": idx * 0.2,
            "Z": 2500.0 - 0.02 * idx,
            "GR": 75.0 + np.cos(idx / 5.0),
            "TVT_input": tvt_input,
            "TVT": tvt,
            "ANCC": 2500.0 - 0.02 * idx - (tvt - 1000.0),
        }
    ).to_csv(train_dir / f"{well}__horizontal_well.csv", index=False)
    submissions, diag, eval_frame = build_submission_frames(
        data_dir=data_dir,
        anchor_submission=None,
        output_dir=tmp_path / "out",
        train_eval=True,
        tail_rows=32,
    )
    assert "stage12_raw" in submissions
    assert "cem_raw" in submissions
    assert "linear_tailfit" in submissions
    assert "geo_tailfit" in submissions
    assert not diag.empty
    assert eval_frame is not None and not eval_frame.empty
    assert (tmp_path / "out" / "submission_direct_stage12_raw.csv").exists()
