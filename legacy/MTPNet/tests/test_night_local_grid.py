from __future__ import annotations

import pandas as pd


def test_evaluate_hand_score_grid_selects_lowest_hand_score() -> None:
    from mtpnet.night_local_grid import HandScoreConfig, evaluate_hand_score_grid

    candidates = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w1", "w1"],
            "offset": [0.0, 1.0, 0.0, 1.0],
            "rows": [10, 10, 20, 20],
            "candidate_mse": [1.0, 100.0, 25.0, 4.0],
            "feat_gr_mad": [0.1, 2.0, 2.0, 0.1],
            "feat_gr_corr": [0.5, -0.5, -0.5, 0.5],
            "feat_gr_dcorr": [0.0, 0.0, 0.0, 0.0],
            "feat_selfcal_rmse_tail": [1.0, 1.0, 1.0, 1.0],
            "feat_offset_minus_known_mean500": [0.0, 1.0, 0.0, 1.0],
            "feat_tvt_oob_frac": [0.0, 0.0, 0.0, 0.0],
        }
    )

    grid, selected = evaluate_hand_score_grid(
        candidates,
        [HandScoreConfig(name="gr_only", w_mad=1.0, w_corr=1.0)],
    )

    assert grid.iloc[0]["row_rmse"] == 3.0**0.5
    assert selected["offset"].tolist() == [0.0, 1.0]
