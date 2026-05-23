from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.config import MTPConfig, RunConfig, WindowConfig
from mtpnet.ranker import (
    FEATURE_COLUMNS,
    apply_ranker_logits,
    build_mode_feature_frame,
    normalize_mode_windows,
    run_ranker_from_frames,
    split_ranker_wells,
    train_catboost_ranker,
)
from mtpnet.stitch import aggregate_mode_windows


def _windows_for_wells(well_ids: list[str]) -> pd.DataFrame:
    rows = []
    for well_index, well_id in enumerate(well_ids):
        target = np.array([100.0 + well_index, 101.0 + well_index], dtype=np.float32)
        rows.append(
            {
                "well_id": well_id,
                "start_step": 0,
                "sample_type": "base_center_hidden",
                "logits": np.array([0.0, 1.0], dtype=np.float32),
                "probs": np.array([0.26894143, 0.7310586], dtype=np.float32),
                "path_tvt": np.array(
                    [
                        target,
                        target + np.array([20.0, 20.0], dtype=np.float32),
                    ],
                    dtype=np.float32,
                ),
                "target_tvt": target,
            }
        )
    return pd.DataFrame(rows)


def _hidden_rows_for_wells(well_ids: list[str]) -> pd.DataFrame:
    rows = []
    for well_index, well_id in enumerate(well_ids):
        for step, offset in [(1, 0.0), (2, 1.0)]:
            tvt = 100.0 + well_index + offset
            rows.append(
                {
                    "id": f"{well_id}_{step}",
                    "well_id": well_id,
                    "row_idx": step,
                    "step": step,
                    "TVT": tvt,
                    "GR": float(step),
                    "base_tvt": tvt + 0.5,
                    "b2_tvt": tvt + 0.25,
                    "a_p50_tvt": tvt + 0.1,
                    "a_p10_tvt": tvt - 2.0,
                    "a_p90_tvt": tvt + 2.0,
                    "b2_danger": 0.2,
                }
            )
    return pd.DataFrame(rows)


def _gr_context(well_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    return {
        well_id: {
            "horizontal_gr": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            "typewell_tvt": np.array([90.0, 100.0, 101.0, 120.0], dtype=np.float32),
            "typewell_gr": np.array([0.0, 1.0, 2.0, 20.0], dtype=np.float32),
        }
        for well_id in well_ids
    }


def test_split_ranker_wells_is_group_safe_and_deterministic() -> None:
    train, valid = split_ranker_wells(["a", "b", "c", "d"], valid_fraction=0.5, seed=7)
    train_again, valid_again = split_ranker_wells(
        ["d", "c", "b", "a"], valid_fraction=0.5, seed=7
    )

    assert train.isdisjoint(valid)
    assert train | valid == {"a", "b", "c", "d"}
    assert train == train_again
    assert valid == valid_again
    assert len(valid) == 2


def test_feature_columns_exclude_target_and_oracle_leakage() -> None:
    forbidden = {
        "target_tvt",
        "mode_rmse_ft",
        "mode_error_ft",
        "is_best_mode",
        "is_top3_mode",
        "TVT",
        "predicted_error_ft",
    }

    assert forbidden.isdisjoint(FEATURE_COLUMNS)
    assert "logit" in FEATURE_COLUMNS
    assert "sample_type" in FEATURE_COLUMNS


def test_build_mode_feature_frame_emits_finite_features() -> None:
    wells = ["a", "b"]
    features = build_mode_feature_frame(
        _windows_for_wells(wells),
        _hidden_rows_for_wells(wells),
        _gr_context(wells),
        history_steps=1,
        future_steps=2,
    )

    assert len(features) == 4
    assert {"well_id", "window_id", "mode_id", "mode_rmse_ft"}.issubset(features.columns)
    numeric = features[FEATURE_COLUMNS].select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    assert set(features["sample_type"]) == {"base_center_hidden"}
    assert features.groupby("window_id")["is_best_mode"].sum().tolist() == [1, 1]


def test_build_mode_feature_frame_accepts_parquet_style_nested_arrays() -> None:
    windows = _windows_for_wells(["a"])
    windows.at[0, "logits"] = np.array([0.0, 1.0], dtype=object)
    path_tvt = np.empty(2, dtype=object)
    path_tvt[0] = np.array([100.0, 101.0], dtype=np.float64)
    path_tvt[1] = np.array([120.0, 121.0], dtype=np.float64)
    windows.at[0, "path_tvt"] = path_tvt

    features = build_mode_feature_frame(
        windows,
        _hidden_rows_for_wells(["a"]),
        _gr_context(["a"]),
        history_steps=1,
        future_steps=2,
    )

    assert features["mode_rmse_ft"].tolist() == pytest.approx([0.0, 20.0])


def test_normalize_mode_windows_makes_nested_arrays_dense() -> None:
    windows = _windows_for_wells(["a"])
    path_tvt = np.empty(2, dtype=object)
    path_tvt[0] = np.array([100.0, 101.0], dtype=np.float64)
    path_tvt[1] = np.array([120.0, 121.0], dtype=np.float64)
    windows.at[0, "path_tvt"] = path_tvt

    normalized = normalize_mode_windows(windows)

    assert normalized.loc[0, "path_tvt"].dtype == np.float32
    assert normalized.loc[0, "path_tvt"].shape == (2, 2)


def test_catboost_ranker_learns_lower_error_mode() -> None:
    wells = ["a", "b", "c", "d"]
    features = build_mode_feature_frame(
        _windows_for_wells(wells),
        _hidden_rows_for_wells(wells),
        _gr_context(wells),
        history_steps=1,
        future_steps=2,
    )
    train_wells, valid_wells = split_ranker_wells(wells, valid_fraction=0.5, seed=3)

    model, _ = train_catboost_ranker(
        features,
        train_wells=train_wells,
        valid_wells=valid_wells,
        seed=3,
        params={"iterations": 80, "depth": 2, "od_wait": 20},
    )
    valid = features[features["well_id"].isin(valid_wells)].copy()
    valid["predicted_error_ft"] = np.expm1(model.predict(valid[list(FEATURE_COLUMNS)]))

    ordered = valid.sort_values(["window_id", "predicted_error_ft"]).groupby(
        "window_id", as_index=False
    )
    assert ordered.head(1)["is_best_mode"].tolist() == [1, 1]


def test_ranker_logits_plug_into_overlap_aggregation() -> None:
    wells = ["a"]
    windows = _windows_for_wells(wells)
    features = build_mode_feature_frame(
        windows,
        _hidden_rows_for_wells(wells),
        _gr_context(wells),
        history_steps=1,
        future_steps=2,
    )
    features["predicted_error_ft"] = features["mode_rmse_ft"]
    reranked = apply_ranker_logits(windows, features, tau_ft=5.0)

    stitched = aggregate_mode_windows(
        reranked,
        history_steps=1,
        future_steps=2,
        strategy="top1",
    )

    assert stitched["pred_tvt"].tolist() == pytest.approx([100.0, 101.0])


def test_ranker_logits_keep_window_ids_after_filtering() -> None:
    wells = ["a", "b"]
    windows = _windows_for_wells(wells)
    windows["window_id"] = ["win_a", "win_b"]
    features = build_mode_feature_frame(
        windows,
        _hidden_rows_for_wells(wells),
        _gr_context(wells),
        history_steps=1,
        future_steps=2,
    )
    features["predicted_error_ft"] = np.where(features["well_id"].eq("b"), 100.0, 0.0)
    features.loc[(features["well_id"] == "b") & (features["mode_id"] == 1), "predicted_error_ft"] = 0.0
    filtered = windows[windows["well_id"] == "b"].copy()

    reranked = apply_ranker_logits(filtered, features, tau_ft=5.0)

    assert int(np.argmax(reranked.iloc[0]["logits"])) == 1


def test_run_ranker_from_frames_writes_artifacts(tmp_path: Path) -> None:
    wells = ["a", "b", "c", "d"]
    cfg = MTPConfig(
        window=WindowConfig(history_steps=1, future_steps=2, rows_per_step=1),
        run=RunConfig(name="tiny_ranker", output_dir=tmp_path),
    )

    summary = run_ranker_from_frames(
        run_dir=tmp_path,
        cfg=cfg,
        mode_windows=_windows_for_wells(wells),
        hidden_rows_all=_hidden_rows_for_wells(wells),
        gr_context=_gr_context(wells),
        seed=5,
        valid_fraction=0.5,
        ranker_params={"iterations": 40, "depth": 2, "od_wait": 10},
        tau_grid=(5.0,),
    )

    assert (tmp_path / "ranker_mode_features.parquet").exists()
    assert (tmp_path / "ranker_predictions.parquet").exists()
    assert (tmp_path / "ranker_candidates.csv").exists()
    assert (tmp_path / "ranker_metrics.json").exists()
    assert (tmp_path / "ranker_report.md").exists()
    assert (tmp_path / "checkpoints" / "mtp_ranker_catboost.cbm").exists()
    assert summary["ranker_split"]["train_wells"]
    assert summary["ranker_split"]["valid_wells"]
