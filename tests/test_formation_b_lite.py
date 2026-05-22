from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.formation_b_lite import (
    add_b_scores,
    b0_sanity,
    compute_raw_scores,
    run,
)


def _write_b_lite_fixture(tmp_path: Path) -> pd.DataFrame:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    well = "well_a"
    n = 70
    hidden_start = 45
    idx = np.arange(n, dtype=float)
    tvt = 1000.0 + 1.5 * idx
    gr = 80.0 + 12.0 * np.sin(tvt / 12.0)
    tvt_input = tvt.copy()
    tvt_input[hidden_start:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": idx,
            "X": idx,
            "Y": idx * 0.0,
            "Z": -3000.0 - idx,
            "TVT": tvt,
            "TVT_input": tvt_input,
            "GR": gr,
            "ANCC": 2000.0,
            "ASTNU": 1975.0,
            "ASTNL": 1950.0,
            "EGFDU": 1925.0,
            "EGFDL": 1900.0,
            "BUDA": 1875.0,
        }
    )
    horizontal.to_csv(train_dir / f"{well}__horizontal_well.csv", index=False)
    tw_tvt = np.linspace(float(tvt.min() - 80.0), float(tvt.max() + 80.0), 400)
    typewell = pd.DataFrame(
        {
            "TVT": tw_tvt,
            "GR": 80.0 + 12.0 * np.sin(tw_tvt / 12.0),
            "Geology": np.where(tw_tvt < np.median(tw_tvt), "ANCC", "ASTNU"),
        }
    )
    typewell.to_csv(train_dir / f"{well}__typewell.csv", index=False)
    hidden_idx = np.arange(hidden_start, n)
    hidden_tvt = tvt[hidden_idx]
    return pd.DataFrame(
        {
            "id": [f"{well}_{i}" for i in hidden_idx],
            "well_id": well,
            "fold": 1,
            "row_idx": hidden_idx,
            "TVT": hidden_tvt,
            "GR": gr[hidden_idx],
            "hidden_frac": np.linspace(0.0, 1.0, len(hidden_idx)),
            "hidden_rows": float(len(hidden_idx)),
            "tvtF_ANCC_full": hidden_tvt.copy(),
            "tvtF_ANCC_late": hidden_tvt + 35.0,
            "S_hat_ANCC_std": 1.0,
            "S_hat_ANCC_plane_residual": 1.0,
        }
    )


def test_b_lite_scores_rank_true_like_candidate(tmp_path: Path) -> None:
    frame = _write_b_lite_fixture(tmp_path)

    raw, diag, info = compute_raw_scores(frame, data_dir=tmp_path, train_dir=tmp_path / "train")
    scores = add_b_scores(raw)
    sanity = b0_sanity(scores, diag)

    good = scores[scores["candidate_name"] == "tvtF_ANCC_full"].iloc[0]
    bad = scores[scores["candidate_name"] == "tvtF_ANCC_late"].iloc[0]
    assert info["a_candidates_count"] >= 2
    assert good["b_path_corr"] > bad["b_path_corr"]
    assert good["b_combined_score"] < bad["b_combined_score"]
    assert int(sanity["true_path_rank"].iloc[0]) <= 3
    assert bool(sanity["true_better_than_plus40"].iloc[0])


def test_b_lite_run_writes_report(tmp_path: Path) -> None:
    frame = _write_b_lite_fixture(tmp_path)
    input_path = tmp_path / "a_candidates.parquet"
    output_dir = tmp_path / "b_lite"
    frame.to_parquet(input_path, index=False)

    metrics = run(
        Namespace(
            input=input_path,
            data_dir=tmp_path,
            train_dir=tmp_path / "train",
            output_dir=output_dir,
            schema10_oof=None,
            schema10_column=None,
            progress_interval=0,
        )
    )

    assert metrics["rows"] == len(frame)
    assert (output_dir / "B_LITE_REPORT.md").is_file()
    assert (output_dir / "b_candidate_scores.parquet").is_file()
    assert metrics["best_selector"]["selector_name"]
