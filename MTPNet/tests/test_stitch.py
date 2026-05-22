import numpy as np
import pandas as pd
import pytest

from mtpnet.stitch import (
    _serializable_mode_windows,
    aggregate_mode_windows,
    dp_decode_mode_windows,
    evaluate_row_predictions,
    triangular_weights,
)


def test_triangular_weights_keep_edges_positive() -> None:
    weights = triangular_weights(5)

    assert weights.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])


def test_aggregate_mode_windows_supports_top1_weighted_and_top3() -> None:
    windows = pd.DataFrame(
        [
            {
                "well_id": "a",
                "start_step": 0,
                "logits": np.array([3.0, 1.0, -2.0], dtype=np.float32),
                "path_tvt": np.array(
                    [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
                    dtype=np.float32,
                ),
            },
            {
                "well_id": "a",
                "start_step": 1,
                "logits": np.array([0.0, 4.0, 3.0], dtype=np.float32),
                "path_tvt": np.array(
                    [[99.0, 99.0], [12.0, 13.0], [22.0, 23.0]],
                    dtype=np.float32,
                ),
            },
        ]
    )

    top1 = aggregate_mode_windows(
        windows, history_steps=1, future_steps=2, strategy="top1"
    )
    weighted = aggregate_mode_windows(
        windows, history_steps=1, future_steps=2, strategy="weighted"
    )
    top3 = aggregate_mode_windows(
        windows, history_steps=1, future_steps=2, strategy="top3"
    )

    assert top1.loc[(top1["well_id"] == "a") & (top1["step"] == 1), "pred_tvt"].item() == pytest.approx(10.0)
    assert weighted["pred_tvt"].between(10.0, 99.0).all()
    assert len(top3) == 3


def test_dp_decode_prefers_smooth_high_mass_path() -> None:
    windows = pd.DataFrame(
        [
            {
                "well_id": "a",
                "start_step": 0,
                "logits": np.array([3.0, 2.0], dtype=np.float32),
                "path_tvt": np.array([[10.0, 11.0, 12.0], [50.0, 51.0, 52.0]], dtype=np.float32),
            }
        ]
    )

    decoded = dp_decode_mode_windows(
        windows,
        history_steps=1,
        future_steps=3,
        top_n=2,
        lambda_step=0.2,
        bin_size_ft=1.0,
    )

    assert decoded["pred_tvt"].tolist() == pytest.approx([10.0, 11.0, 12.0])


def test_evaluate_row_predictions_reports_rmse_and_guard_shifts() -> None:
    rows = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "row_idx": [0, 1, 0, 1],
            "step": [0, 0, 0, 0],
            "TVT": [10.0, 12.0, 20.0, 22.0],
            "base_tvt": [9.0, 11.0, 19.0, 21.0],
            "b2_tvt": [10.5, 12.5, 18.0, 20.0],
            "GR": [1.0, np.nan, 2.0, 3.0],
        }
    )
    pred = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "step": [0, 0],
            "pred_tvt": [11.0, 21.0],
        }
    )

    metrics = evaluate_row_predictions(rows, pred, "candidate")

    assert metrics["candidate"] == "candidate"
    assert metrics["rows"] == 4
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mean_well_rmse"] == pytest.approx(1.0)
    assert metrics["p95_abs_shift_vs_b2"] > 0.0


def test_serializable_mode_windows_converts_nested_arrays_to_lists() -> None:
    frame = pd.DataFrame(
        [
            {
                "well_id": "a",
                "start_step": 0,
                "logits": np.array([1.0, 2.0], dtype=np.float32),
                "probs": np.array([0.25, 0.75], dtype=np.float32),
                "path_tvt": np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32),
            }
        ]
    )

    serializable = _serializable_mode_windows(frame)

    assert serializable.loc[0, "logits"] == [1.0, 2.0]
    assert serializable.loc[0, "path_tvt"] == [[10.0, 11.0], [12.0, 13.0]]
