from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_top_state_frame() -> pd.DataFrame:
    rows = []
    patterns = {
        "a": [0, 1, 2, 3, 4, 3, 2, 1],
        "b": [5, 4, 3, 2, 1, 2, 3, 4],
        "c": [2, 2, 2, 3, 4, 4, 3, 2],
        "d": [1, 2, 3, 2, 1, 1, 2, 3],
    }
    for well_id, top_values in patterns.items():
        for row_idx, top in enumerate(top_values):
            hidden = row_idx >= 3
            tvt = 100.0 + float(top)
            rows.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "MD": 1000.0 + row_idx * 10.0,
                    "X": float(row_idx),
                    "Y": 0.0,
                    "Z": -float(top),
                    "GR": 80.0 + float(top),
                    "TVT": tvt,
                    "TVT_input": np.nan if hidden else tvt,
                    "ANCC": 1000.0 + float(top),
                }
            )
    return pd.DataFrame(rows)


def test_top_state_feature_builder_is_schema_safe() -> None:
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS
    from mtpnet.top_state_teacher import build_top_state_dataset

    dataset = build_top_state_dataset(_tiny_top_state_frame(), rows_per_step=1)

    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(dataset.feature_columns)
    assert {"target_state", "target_delta"}.issubset(dataset.rows.columns)
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()
    assert set(dataset.rows["target_state"].unique()).issubset({0, 1, 2})


def test_top_state_smoke_writes_oof_artifacts(tmp_path: Path) -> None:
    from mtpnet.top_state_teacher import TopStateTeacherConfig, run_top_state_teacher_from_frame

    metrics = run_top_state_teacher_from_frame(
        _tiny_top_state_frame(),
        config=TopStateTeacherConfig(
            output_dir=tmp_path,
            rows_per_step=1,
            n_folds=2,
            iterations=20,
            depth=2,
            learning_rate=0.2,
            seed=3,
            n_panel_wells=1,
        ),
    )

    assert metrics["candidate"] == "top_state_teacher_v0"
    assert metrics["hidden"]["accuracy"] >= 0.0
    preds = pd.read_parquet(tmp_path / "top_state_oof_predictions.parquet")
    assert {"prob_down", "prob_flat", "prob_up", "pred_expected_sign"}.issubset(preds.columns)
    assert (tmp_path / "top_state_metrics.json").exists()
    assert (tmp_path / "top_state_report.md").exists()
    assert list((tmp_path / "figures").glob("*.png"))


def test_chunk_policy_can_use_top_state_predictions() -> None:
    from mtpnet.chunk_policy import build_chunk_policy_dataset

    frame = _tiny_top_state_frame()
    top_state = pd.DataFrame(
        {
            "well_id": ["a"] * 5,
            "step": [3, 4, 5, 6, 7],
            "prob_down": [0.05, 0.05, 0.8, 0.8, 0.8],
            "prob_flat": [0.05, 0.05, 0.1, 0.1, 0.1],
            "prob_up": [0.9, 0.9, 0.1, 0.1, 0.1],
            "pred_expected_sign": [1.0, 1.0, -1.0, -1.0, -1.0],
        }
    )

    dataset = build_chunk_policy_dataset(frame, chunk_size=3, top_state_predictions=top_state)

    assert "feat_top_state_available_frac" in dataset.feature_columns
    assert "feat_top_state_sign_agree" in dataset.feature_columns
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()
