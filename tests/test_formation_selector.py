from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rogii.formation_plane_knn import _anchor_train_pseudo_split
from rogii.formation_selector import (
    _alpha_for_band,
    build_candidate_metadata,
    build_selector_outputs,
    score_safe_blends,
)


def _selector_frame() -> pd.DataFrame:
    rows = []
    for well_idx, well in enumerate(["w1", "w2"]):
        for row_idx in range(4):
            tvt = 100.0 + 10.0 * well_idx + row_idx
            rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well_id": well,
                    "fold": 1,
                    "row_idx": row_idx,
                    "TVT": tvt,
                    "GR": 80.0,
                    "hidden_frac": row_idx / 3.0,
                    "hidden_rows": 4.0,
                    "schema10_oof_raw": tvt + 3.0,
                    "S_hat_ANCC_std": 2.0,
                    "S_hat_ANCC_plane_residual": 1.0,
                    "neighbor_dist_mean": 10.0,
                    "neighbor_dist_min": 3.0,
                    "b_ANCC_full": 1.0,
                    "b_ANCC_late": 1.5,
                    "b_ANCC_wls": 1.25,
                    "b_ANCC_std": 0.5,
                    "b_ANCC_late_minus_full": 0.5,
                    "tvtF_ANCC_full": tvt + (0.2 if well == "w1" else 4.0),
                    "tvtF_ANCC_late": tvt + (5.0 if well == "w1" else 0.3),
                    "anchor_fit_rmse__tvtF_ANCC_full": 1.0 if well == "w1" else 5.0,
                    "late_anchor_fit_rmse__tvtF_ANCC_full": 1.2 if well == "w1" else 5.0,
                    "anchor_bias__tvtF_ANCC_full": 0.2,
                    "anchor_slope_error__tvtF_ANCC_full": 0.1,
                    "pseudo_hidden_rmse__tvtF_ANCC_full": 0.8 if well == "w1" else 5.0,
                    "pseudo_hidden_bias__tvtF_ANCC_full": 0.1,
                    "pseudo_hidden_slope_error__tvtF_ANCC_full": 0.1,
                    "roughness__tvtF_ANCC_full": 0.1,
                    "finite_frac__tvtF_ANCC_full": 1.0,
                    "anchor_fit_rmse__tvtF_ANCC_late": 5.0 if well == "w1" else 1.0,
                    "late_anchor_fit_rmse__tvtF_ANCC_late": 5.0 if well == "w1" else 1.1,
                    "anchor_bias__tvtF_ANCC_late": 0.3,
                    "anchor_slope_error__tvtF_ANCC_late": 0.1,
                    "pseudo_hidden_rmse__tvtF_ANCC_late": 5.0 if well == "w1" else 0.7,
                    "pseudo_hidden_bias__tvtF_ANCC_late": 0.1,
                    "pseudo_hidden_slope_error__tvtF_ANCC_late": 0.1,
                    "roughness__tvtF_ANCC_late": 0.2,
                    "finite_frac__tvtF_ANCC_late": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_anchor_train_pseudo_split_is_ordered_and_disjoint() -> None:
    train, pseudo = _anchor_train_pseudo_split(np.arange(10))

    assert len(train) == 7
    assert len(pseudo) == 3
    assert set(train).isdisjoint(set(pseudo))
    assert int(train[-1]) < int(pseudo[0])


def test_candidate_metadata_contains_required_selector_fields() -> None:
    metadata = build_candidate_metadata(_selector_frame())

    row = metadata[
        (metadata["well_id"] == "w1")
        & (metadata["candidate_name"] == "tvtF_ANCC_full")
    ].iloc[0]
    assert row["formation"] == "ANCC"
    assert row["knn_type"] == "plane"
    assert row["bias_mode"] == "full"
    assert np.isfinite(row["robust_score"])
    assert row["pseudo_hidden_source"] == "pseudo_hidden"


def test_selectors_and_soft_blend_are_stable() -> None:
    frame = _selector_frame()
    metadata = build_candidate_metadata(frame)

    predictions, choices, selectors = build_selector_outputs(frame, metadata)

    assert "pseudo_hidden_min" in selectors
    w1_choice = choices[
        (choices["selector_name"] == "pseudo_hidden_min")
        & (choices["well_id"] == "w1")
    ].iloc[0]
    assert w1_choice["selected_candidate"] == "tvtF_ANCC_full"
    soft = choices[choices["selector_name"] == "top3_soft_t2"].iloc[0]
    weights = json.loads(soft["blend_weights"])
    assert np.isclose(sum(weights), 1.0)
    assert len(soft["blend_candidates"].split(",")) <= 3
    assert np.isfinite(predictions["top3_soft_t2"]).all()


def test_safe_blend_uses_confidence_band_and_clip() -> None:
    frame = _selector_frame()
    metadata = build_candidate_metadata(frame)
    predictions, choices, selectors = build_selector_outputs(frame, metadata)

    bands = np.array(["high", "mid", "low", "off"], dtype=object)
    alpha = _alpha_for_band(bands, high=0.6, mid=0.3, low=0.1)
    assert np.allclose(alpha, [0.6, 0.3, 0.1, 0.0])

    scores = score_safe_blends(predictions, choices, selectors)
    assert not scores.empty
    assert {"clip", "alpha_high", "alpha_mid", "alpha_low"}.issubset(scores.columns)
    assert np.isfinite(scores["rmse"]).any()
