from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

from rogii.formation_b2_constrained import (
    build_b2_metadata,
    filter_report,
    intersection_selectors,
    run,
)


def _fixture_frame() -> pd.DataFrame:
    rows = []
    for well_idx, well in enumerate(("well_a", "well_b")):
        base = 1000.0 + 100.0 * well_idx
        for row_idx in range(5):
            tvt = base + float(row_idx)
            rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well_id": well,
                    "row_idx": row_idx,
                    "TVT": tvt,
                    "GR": 80.0 + row_idx,
                    "schema10_oof_raw": tvt + 1.0,
                    "tvtF_ANCC_full": tvt,
                    "tvtF_ANCC_late": tvt + 25.0,
                    "anchor_fit_rmse__tvtF_ANCC_full": 1.0,
                    "anchor_fit_rmse__tvtF_ANCC_late": 8.0,
                    "late_anchor_fit_rmse__tvtF_ANCC_full": 1.5,
                    "late_anchor_fit_rmse__tvtF_ANCC_late": 9.0,
                    "roughness__tvtF_ANCC_full": 0.1,
                    "roughness__tvtF_ANCC_late": 0.2,
                    "finite_frac__tvtF_ANCC_full": 1.0,
                    "finite_frac__tvtF_ANCC_late": 1.0,
                    "b_ANCC_std": 0.5,
                }
            )
    return pd.DataFrame(rows)


def _fixture_b_scores() -> pd.DataFrame:
    rows = []
    for well in ("well_a", "well_b"):
        rows.extend(
            [
                {
                    "well_id": well,
                    "candidate_name": "tvtF_ANCC_full",
                    "hidden_rmse": 0.0,
                    "b_combined_score": -2.0,
                    "b_combined_without_surface_terms": -2.0,
                    "b_combined_gr_only": -2.0,
                    "b_path_corr": 0.9,
                    "b_dgr_corr": 0.8,
                    "b_ncc15_mean": 0.7,
                    "b_ncc_multiscale_mean": 0.7,
                    "surface_std": 1.0,
                },
                {
                    "well_id": well,
                    "candidate_name": "tvtF_ANCC_late",
                    "hidden_rmse": 25.0,
                    "b_combined_score": 3.0,
                    "b_combined_without_surface_terms": 3.0,
                    "b_combined_gr_only": 3.0,
                    "b_path_corr": 0.1,
                    "b_dgr_corr": 0.1,
                    "b_ncc15_mean": 0.1,
                    "b_ncc_multiscale_mean": 0.1,
                    "surface_std": 1.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_b2_metadata_filters_and_intersections_choose_safe_candidate() -> None:
    frame = _fixture_frame()
    b_scores = _fixture_b_scores()

    meta = build_b2_metadata(frame, b_scores, baseline_available=True, progress_interval=0)
    filters = filter_report(frame, meta, baseline_available=True)
    choices, intersections = intersection_selectors(frame, meta)

    assert set(meta["candidate_name"]) == {"tvtF_ANCC_full", "tvtF_ANCC_late"}
    assert float(filters["kept_oracle_rmse"].min()) == 0.0
    assert intersections.iloc[0]["rmse"] == 0.0
    assert choices.groupby("selector_name")["well_id"].nunique().min() == 2


def test_b2_run_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "a_candidates.parquet"
    b_scores_path = tmp_path / "b_scores.parquet"
    output_dir = tmp_path / "b2"
    _fixture_frame().to_parquet(input_path, index=False)
    _fixture_b_scores().to_parquet(b_scores_path, index=False)

    metrics = run(
        Namespace(
            input=input_path,
            b_scores=b_scores_path,
            output_dir=output_dir,
            schema10_oof=None,
            schema10_column=None,
            progress_interval=0,
        )
    )

    assert metrics["schema10_available"] is True
    assert (output_dir / "B2_CONSTRAINED_REPORT.md").is_file()
    assert (output_dir / "b2_candidate_metadata.parquet").is_file()
    assert metrics["best_intersection"][0]["rmse"] == 0.0
    assert metrics["best_safe_blend"]
