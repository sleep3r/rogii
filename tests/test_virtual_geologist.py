from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.virtual_geologist import (
    _topk_columns,
    oracle_scores,
    solve_virtual_geologist_well,
)


def _frame(offset: float = 8.0, n: int = 80) -> pd.DataFrame:
    idx = np.arange(n, dtype=float)
    true = 1100.0 + 0.2 * idx
    base = true + offset
    gr = 70.0 + np.sin(true / 15.0) * 10.0
    return pd.DataFrame(
        {
            "id": [f"well0_{int(i)}" for i in idx],
            "well": "well0",
            "row_index": idx.astype(int),
            "fold": 1,
            "tvt_true": true,
            "schema10_oof_raw": base,
            "last_known_tvt": np.full(n, true[0] - 1.0),
            "gr": gr,
            "kg_form_mean_tvt": true + 0.5,
            "kg_form_ancc_tvt": true - 0.5,
            "kg_dense_ancc_tvt": true + 1.0,
        }
    )


def _typewell() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tvt = np.linspace(1080.0, 1140.0, 400)
    gr = 70.0 + np.sin(tvt / 15.0) * 10.0
    dgr = np.gradient(gr, tvt)
    return tvt, gr, dgr


def test_virtual_geologist_solves_bounded_shift() -> None:
    frame = _frame(offset=8.0)
    solved = solve_virtual_geologist_well(
        frame,
        typewell=_typewell(),
        cfg={
            "offset_grid": [-16.0, 16.0, 4.0],
            "dip_grid": [0.0, 0.0, 1.0],
            "stretch_grid": [0.0, 0.0, 1.0],
            "top_k": 3,
            "w_gr": 0.2,
            "w_surface": 2.0,
            "w_tail": 0.0,
            "w_known": 0.0,
            "w_smooth": 0.0,
        },
    )

    assert len(solved) == len(frame)
    assert np.isfinite(solved["vg_best_tvt"]).all()
    assert abs(float(solved["vg_shift"].median()) + 8.0) <= 4.0
    assert "vg_top3_tvt" in solved.columns


def test_virtual_geologist_oracle_scores_are_finite() -> None:
    solved = solve_virtual_geologist_well(
        _frame(offset=6.0),
        typewell=_typewell(),
        cfg={
            "offset_grid": [-12.0, 12.0, 6.0],
            "dip_grid": [-6.0, 6.0, 6.0],
            "stretch_grid": [0.0, 0.0, 1.0],
            "top_k": 4,
            "w_tail": 0.0,
        },
    )
    topk = _topk_columns(solved)
    scores = oracle_scores(solved, topk)

    assert len(topk) == 4
    assert np.isfinite(scores["row_topk_oracle_rmse"])
    assert np.isfinite(scores["thirds_topk_oracle_rmse"])
