from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_rows() -> pd.DataFrame:
    rows = []
    for well_id, offset, b2_bias in [
        ("a", 0.0, -5.0),
        ("b", 20.0, -5.0),
        ("c", -15.0, 4.0),
        ("d", 30.0, 4.0),
    ]:
        for row_idx in range(8):
            is_hidden = row_idx >= 3
            tvt = 100.0 + offset + row_idx * 2.0
            rows.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "step": row_idx,
                    "MD": 1000.0 + row_idx,
                    "X": 10.0 + row_idx,
                    "Y": 20.0,
                    "Z": -100.0 - row_idx,
                    "GR": 80.0 + row_idx,
                    "TVT": tvt,
                    "TVT_input": np.nan if is_hidden else tvt,
                    "b2_tvt": tvt + b2_bias,
                    "base_tvt": tvt + b2_bias - 1.0,
                    "a_p50_tvt": tvt - b2_bias,
                }
            )
    return pd.DataFrame(rows)


def test_candidate_selector_feature_builder_is_schema_safe() -> None:
    from mtpnet.candidate_selector import build_selector_dataset
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS

    dataset = build_selector_dataset(_tiny_rows())

    assert dataset.feature_columns
    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(dataset.feature_columns)
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()
    assert {"candidate", "target_rmse", "well_id"}.issubset(dataset.rows.columns)


def test_traceback_candidate_action_is_marked_as_risky_non_anchor() -> None:
    from mtpnet.candidate_selector import _candidate_action_features

    features = _candidate_action_features("traceback_band_w80_off+1p00")

    assert features["feat_action_is_traceback"] == 1.0
    assert features["feat_action_is_anchor"] == 0.0
    assert features["feat_action_is_dangerous"] == 1.0
    assert features["feat_action_global_risk"] >= 0.75


def test_template_envelope_action_is_marked_as_risky_non_anchor() -> None:
    from mtpnet.candidate_selector import _candidate_action_features

    features = _candidate_action_features("template_envelope_mid")

    assert features["feat_action_is_template_envelope"] == 1.0
    assert features["feat_action_is_anchor"] == 0.0
    assert features["feat_action_is_dangerous"] == 1.0
    assert features["feat_action_global_risk"] >= 0.70


def test_candidate_selector_can_use_traceback_candidates() -> None:
    from mtpnet.candidate_selector import build_selector_dataset

    traceback = pd.DataFrame(
        {
            "id": ["a_3", "a_4"],
            "well_id": ["a", "a"],
            "row_idx": [3, 4],
            "candidate": ["traceback_band_w80"] * 2,
            "pred_tvt": [106.0, 108.0],
            "TVT": [999.0, 999.0],
        }
    )

    dataset = build_selector_dataset(_tiny_rows(), traceback_candidates=traceback)

    assert "traceback_band_w80" in set(dataset.rows["candidate"])
    assert "feat_action_is_traceback" in dataset.feature_columns
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()


def test_candidate_selector_smoke_writes_oof_artifacts(tmp_path: Path) -> None:
    from mtpnet.candidate_selector import CandidateSelectorConfig, run_candidate_selector_from_frames

    metrics = run_candidate_selector_from_frames(
        _tiny_rows(),
        config=CandidateSelectorConfig(
            output_dir=tmp_path,
            n_folds=2,
            iterations=5,
            learning_rate=0.1,
            depth=2,
            seed=7,
        ),
    )

    assert metrics["wells"] == 4
    assert metrics["folds"] == 2
    assert metrics["selected"]["row_rmse"] >= 0.0
    assert metrics["oracle"]["row_rmse"] <= metrics["b2"]["row_rmse"]
    assert (tmp_path / "selector_row_predictions.parquet").exists()
    assert (tmp_path / "selector_well_predictions.csv").exists()
    assert (tmp_path / "selector_metrics.json").exists()
    assert (tmp_path / "selector_report.md").exists()


def test_candidate_selector_fold_wells_are_disjoint(tmp_path: Path) -> None:
    from mtpnet.candidate_selector import CandidateSelectorConfig, run_candidate_selector_from_frames

    metrics = run_candidate_selector_from_frames(
        _tiny_rows(),
        config=CandidateSelectorConfig(
            output_dir=tmp_path,
            n_folds=2,
            iterations=3,
            learning_rate=0.1,
            depth=2,
            seed=13,
        ),
    )

    for fold in metrics["fold_metrics"]:
        assert set(fold["train_wells"]).isdisjoint(fold["valid_wells"])


def test_candidate_selector_progress_logs(tmp_path: Path, capsys) -> None:
    from mtpnet.candidate_selector import CandidateSelectorConfig, run_candidate_selector_from_frames

    run_candidate_selector_from_frames(
        _tiny_rows(),
        config=CandidateSelectorConfig(
            output_dir=tmp_path,
            n_folds=2,
            iterations=2,
            learning_rate=0.1,
            depth=2,
            seed=17,
            progress_every=1,
        ),
    )

    captured = capsys.readouterr()
    assert "[selector] built candidates for" in captured.err
    assert "[selector] fold 1/2" in captured.err


def test_ranker_guard_dataset_has_mse_regret_and_boundary_features() -> None:
    from mtpnet.candidate_selector import build_selector_dataset
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS

    dataset = build_selector_dataset(_tiny_rows())

    assert {"target_mse", "target_regret_mse", "target_gain_vs_b2_mse"}.issubset(
        dataset.rows.columns
    )
    assert "feat_boundary_left_jump" in dataset.feature_columns
    assert "feat_bridge_mean_abs" in dataset.feature_columns
    assert "feat_candidate_bank_median_mean_abs" in dataset.feature_columns
    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(dataset.feature_columns)


def test_ranker_guard_falls_back_to_b2_when_predicted_gain_is_weak() -> None:
    from mtpnet.candidate_selector import _select_ranker_guard_wells

    local = pd.DataFrame(
        [
            {
                "well_id": "w1",
                "candidate": "b2",
                "ranker_score": 0.0,
                "pred_mse": 25.0,
                "target_rmse": 5.0,
                "target_mse": 25.0,
                "feat_action_is_dangerous": 0.0,
                "fold": 0,
            },
            {
                "well_id": "w1",
                "candidate": "b2_shift+20",
                "ranker_score": 10.0,
                "pred_mse": 24.8,
                "target_rmse": 4.5,
                "target_mse": 20.25,
                "feat_action_is_dangerous": 0.0,
                "fold": 0,
            },
        ]
    )

    selected = _select_ranker_guard_wells(
        local,
        guard_gain_threshold=1.0,
        dangerous_guard_gain_threshold=5.0,
    )

    assert selected.loc[0, "selected_candidate"] == "b2"
    assert selected.loc[0, "guard_reason"] == "weak_predicted_gain"


def test_candidate_selector_ranker_guard_smoke_writes_metrics(tmp_path: Path) -> None:
    from mtpnet.candidate_selector import CandidateSelectorConfig, run_candidate_selector_from_frames

    metrics = run_candidate_selector_from_frames(
        _tiny_rows(),
        config=CandidateSelectorConfig(
            output_dir=tmp_path,
            mode="ranker_guard",
            n_folds=2,
            iterations=5,
            learning_rate=0.1,
            depth=2,
            seed=23,
            guard_gain_threshold=0.0,
            dangerous_guard_gain_threshold=1.0,
        ),
    )

    assert metrics["candidate"] == "candidate_selector_v1_ranker_guard"
    assert "mse_gain_capture" in metrics
    assert "guard_counts" in metrics
    assert (tmp_path / "selector_metrics.json").exists()
