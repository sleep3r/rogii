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
        for row_idx in range(16):
            is_hidden = row_idx >= 4
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


def test_chunk_policy_dataset_is_schema_safe_and_grouped() -> None:
    from mtpnet.chunk_policy import build_chunk_policy_dataset
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS

    dataset = build_chunk_policy_dataset(_tiny_rows(), chunk_size=4)

    assert dataset.rows["group_id"].nunique() > 1
    assert {"target_mse", "target_regret_mse", "chunk_id", "candidate"}.issubset(
        dataset.rows.columns
    )
    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(dataset.feature_columns)
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()
    assert "feat_dz_state_corr" in dataset.feature_columns
    assert "feat_dz_state_sign_agree" in dataset.feature_columns


def test_dz_state_features_reward_candidate_matching_negative_dz() -> None:
    from mtpnet.chunk_policy import _dz_state_features

    z = np.asarray([0.0, -1.0, -3.0, -6.0])
    aligned = _dz_state_features(np.asarray([100.0, 101.0, 103.0, 106.0]), z)
    opposite = _dz_state_features(np.asarray([100.0, 99.0, 97.0, 94.0]), z)

    assert aligned["feat_dz_state_corr"] > 0.99
    assert aligned["feat_dz_state_sign_agree"] == 1.0
    assert aligned["feat_dz_state_scaled_rmse"] < 1e-6
    assert opposite["feat_dz_state_corr"] < -0.99
    assert opposite["feat_dz_state_sign_agree"] == 0.0


def test_chunk_policy_adds_softseg_and_loccorr_features_without_target_leakage() -> None:
    from mtpnet.chunk_policy import build_chunk_policy_dataset

    softseg = pd.DataFrame(
        {
            "well_id": ["a"] * 12,
            "compressed_step": list(range(4, 16)),
            "top1_tvt": [110.0 + idx for idx in range(12)],
            "dp_tvt": [111.0 + idx for idx in range(12)],
            "true_tvt": [999.0] * 12,
            "target_rank": [1] * 12,
        }
    )
    loccorr = pd.DataFrame(
        {
            "well_id": ["a"] * 12,
            "step": list(range(4, 16)),
            "variant": ["value_location_raw"] * 12,
            "top1_tvt": [109.0 + idx for idx in range(12)],
            "true_rank": [1] * 12,
            "top10_sqerr": [0.0] * 12,
        }
    )

    dataset = build_chunk_policy_dataset(
        _tiny_rows(),
        chunk_size=4,
        step_predictions=softseg,
        location_aware_steps=loccorr,
    )

    assert "softseg_top1" in set(dataset.rows["candidate"])
    assert "feat_softseg_top1_mean_abs" in dataset.feature_columns
    assert "feat_loccorr_value_location_raw_mean_abs" in dataset.feature_columns
    assert not any(
        token in column
        for column in dataset.feature_columns
        for token in ("true_tvt", "target", "top10_sqerr", "true_rank")
    )
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()


def test_chunk_policy_adds_traceback_candidates_to_dataset() -> None:
    from mtpnet.chunk_policy import build_chunk_policy_dataset

    traceback = pd.DataFrame(
        {
            "id": ["a_4", "a_5", "a_6", "a_7"],
            "well_id": ["a"] * 4,
            "row_idx": [4, 5, 6, 7],
            "candidate": ["traceback_band_w80"] * 4,
            "pred_tvt": [108.0, 110.0, 112.0, 114.0],
            "TVT": [999.0] * 4,
        }
    )

    dataset = build_chunk_policy_dataset(
        _tiny_rows(),
        chunk_size=4,
        traceback_candidates=traceback,
    )

    assert "traceback_band_w80" in set(dataset.rows["candidate"])
    assert not any(
        token in column
        for column in dataset.feature_columns
        for token in ("TVT", "true_TVT", "target")
    )
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()


def test_dp_decode_prefers_smooth_path_over_jumpy_greedy() -> None:
    from mtpnet.chunk_policy import viterbi_select_candidates

    chunks = [
        {
            "chunk_id": 0,
            "rows": 10,
            "candidates": {
                "a": {"cost": 1.0, "first": 100.0, "last": 101.0},
                "b": {"cost": 2.0, "first": 140.0, "last": 141.0},
            },
        },
        {
            "chunk_id": 1,
            "rows": 10,
            "candidates": {
                "a": {"cost": 2.0, "first": 102.0, "last": 103.0},
                "b": {"cost": 1.0, "first": 142.0, "last": 143.0},
            },
        },
    ]

    selected = viterbi_select_candidates(chunks, switch_penalty=5.0, jump_penalty=2.0)

    assert selected == ["a", "a"]


def test_chunk_policy_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.chunk_policy import ChunkPolicyConfig, run_chunk_policy_from_frames

    metrics = run_chunk_policy_from_frames(
        _tiny_rows(),
        config=ChunkPolicyConfig(
            output_dir=tmp_path,
            n_folds=2,
            chunk_size=4,
            iterations=5,
            learning_rate=0.1,
            depth=2,
            seed=11,
        ),
    )

    assert metrics["candidate"] == "chunk_ranker_dp_v1"
    assert metrics["dp"]["row_rmse"] >= 0.0
    assert metrics["oracle"]["chunk_oracle_row_rmse"] <= metrics["b2"]["row_rmse"]
    assert "selected_row_weighted_rmse" in metrics["fold_metrics"][0]
    assert metrics["fold_metrics"][0]["selected_row_weighted_rmse"] >= 0.0
    assert "selected_p90_chunk_rmse" in metrics["fold_metrics"][0]
    assert (tmp_path / "chunk_policy_metrics.json").exists()
    assert (tmp_path / "chunk_policy_row_predictions.parquet").exists()
    assert (tmp_path / "chunk_policy_report.md").exists()


def test_selfcal_probe_features_use_known_tvt_input_only() -> None:
    from mtpnet.chunk_policy import build_selfcal_probe_features

    rows = _tiny_rows()
    features = build_selfcal_probe_features(rows[rows["well_id"] == "a"], chunk_size=2, max_probes=2)

    assert not features.empty
    assert {"candidate", "feat_probe_rmse_mean", "feat_probe_win_frac"}.issubset(
        features.columns
    )
    assert np.isfinite(features.filter(like="feat_probe_").to_numpy(dtype=np.float64)).all()


def test_spatial_priors_are_fold_safe() -> None:
    from mtpnet.chunk_policy import add_fold_safe_spatial_prior_features, build_chunk_policy_dataset

    dataset = build_chunk_policy_dataset(_tiny_rows(), chunk_size=4)
    rows, features, feature_columns = add_fold_safe_spatial_prior_features(
        dataset.rows,
        dataset.features,
        dataset.feature_columns,
        train_wells=["a", "b"],
        valid_wells=["c"],
    )

    valid = rows["well_id"].astype(str) == "c"
    assert "feat_spatial_candidate_prior_mse" in feature_columns
    assert np.isfinite(features.loc[valid, "feat_spatial_candidate_prior_mse"]).all()
    train_target = rows.loc[rows["well_id"].isin(["a", "b"])].groupby("candidate")[
        "target_mse"
    ].mean()
    candidate = rows.loc[valid, "candidate"].iloc[0]
    expected = np.log1p(float(train_target.loc[candidate]))
    actual = float(features.loc[valid, "feat_spatial_candidate_prior_mse"].iloc[0])
    assert np.isclose(actual, expected)


def test_shared_typewell_priors_are_schema_safe_and_candidate_aware() -> None:
    from mtpnet.chunk_policy import (
        add_fold_safe_shared_typewell_features,
        build_chunk_policy_dataset,
    )
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS

    dataset = build_chunk_policy_dataset(_tiny_rows(), chunk_size=4)
    neighbours = pd.DataFrame(
        [
            {
                "query_well_id": "c",
                "neighbour_well_id": "a",
                "typewell_corr": 0.8,
                "typewell_shift_bins": 4.0,
                "xyz_distance": 100.0,
                "rank": 1,
                "fold": 0,
            },
            {
                "query_well_id": "c",
                "neighbour_well_id": "b",
                "typewell_corr": 0.6,
                "typewell_shift_bins": -2.0,
                "xyz_distance": 200.0,
                "rank": 2,
                "fold": 0,
            },
        ]
    )

    rows, features, feature_columns = add_fold_safe_shared_typewell_features(
        dataset.rows,
        dataset.features,
        dataset.feature_columns,
        neighbours=neighbours,
        train_wells=["a", "b"],
        valid_wells=["c"],
    )

    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(feature_columns)
    assert "feat_stw_top1_corr" in feature_columns
    assert "feat_stw_corr_x_action_shift" in feature_columns
    valid = rows["well_id"].astype(str) == "c"
    assert np.isfinite(features.loc[valid, feature_columns].to_numpy(dtype=np.float64)).all()
    assert np.isclose(float(features.loc[valid, "feat_stw_top1_corr"].iloc[0]), 0.8)
    shift_rows = valid & (rows["candidate"].astype(str).str.contains("_shift"))
    anchor_rows = valid & (rows["candidate"].astype(str) == "b2")
    assert features.loc[shift_rows, "feat_stw_corr_x_action_shift"].max() > 0.0
    assert features.loc[anchor_rows, "feat_stw_corr_x_action_shift"].max() == 0.0


def test_chunk_policy_smoke_can_use_shared_typewell_priors(tmp_path: Path) -> None:
    from mtpnet.chunk_policy import ChunkPolicyConfig, run_chunk_policy_from_frames

    neighbours = pd.DataFrame(
        [
            {
                "query_well_id": well_id,
                "neighbour_well_id": "a",
                "typewell_corr": 0.7,
                "typewell_shift_bins": 3.0,
                "xyz_distance": 123.0,
                "rank": 1,
                "fold": 0,
            }
            for well_id in ["a", "b", "c", "d"]
        ]
    )
    neighbour_path = tmp_path / "typewell_neighbours.parquet"
    neighbours.to_parquet(neighbour_path, index=False)

    metrics = run_chunk_policy_from_frames(
        _tiny_rows(),
        config=ChunkPolicyConfig(
            output_dir=tmp_path,
            n_folds=2,
            chunk_size=4,
            iterations=5,
            learning_rate=0.1,
            depth=2,
            seed=11,
            use_shared_typewell_priors=True,
            shared_typewell_neighbours_path=neighbour_path,
        ),
    )

    assert metrics["use_shared_typewell_priors"] is True
    assert "feat_stw_top1_corr" in metrics["feature_columns"]
    assert (tmp_path / "chunk_policy_metrics.json").exists()
