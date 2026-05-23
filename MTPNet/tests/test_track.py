from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.config import MTPConfig, RunConfig, WindowConfig
from mtpnet.track import (
    TrackConfig,
    TrackParticle,
    apply_tracker_logit_source,
    merge_and_prune_particles,
    particles_to_step_predictions,
    run_track_split_audit_from_frames,
    run_tracker_from_frames,
    track_mode_windows,
)


def _mode_window(
    *,
    well_id: str = "a",
    start_step: int,
    logits: list[float],
    paths: list[list[float]],
) -> dict[str, object]:
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    return {
        "well_id": well_id,
        "start_step": start_step,
        "sample_type": "base_center_hidden",
        "logits": np.asarray(logits, dtype=np.float32),
        "probs": probs.astype(np.float32),
        "path_tvt": np.asarray(paths, dtype=np.float32),
        "target_tvt": np.asarray(paths[0], dtype=np.float32),
    }


def _hidden_rows(steps: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"a_{step}" for step in steps],
            "well_id": ["a"] * len(steps),
            "row_idx": steps,
            "step": steps,
            "TVT": [100.0 + step for step in steps],
            "GR": [1.0] * len(steps),
            "base_tvt": [100.0 + step for step in steps],
            "b2_tvt": [100.5 + step for step in steps],
        }
    )


def test_track_mode_windows_carries_forward_particles_across_overlapping_windows() -> None:
    windows = pd.DataFrame(
        [
            _mode_window(
                start_step=0,
                logits=[2.0, 0.0],
                paths=[[10.0, 11.0, 12.0], [50.0, 51.0, 52.0]],
            ),
            _mode_window(
                start_step=1,
                logits=[0.0, 2.0],
                paths=[[11.0, 12.0, 13.0], [70.0, 71.0, 72.0]],
            ),
        ]
    )

    particles = track_mode_windows(
        windows,
        history_steps=1,
        future_steps=3,
        cfg=TrackConfig(keep_top=4, n_realizations=4, merge_tolerance_ft=0.1),
    )

    top = particles["a"][0]
    assert top.steps == (1, 2, 3, 4)
    assert top.tvt == pytest.approx((10.0, 11.0, 12.0, 13.0))
    assert len(particles["a"]) <= 4


def test_merge_and_prune_particles_merges_close_endpoints_and_keeps_top() -> None:
    particles = [
        TrackParticle("a", (1, 2), (10.0, 11.0), -0.1, ("p0",), 1),
        TrackParticle("a", (1, 2), (10.2, 11.2), -0.2, ("p1",), 1),
        TrackParticle("a", (1, 2), (30.0, 31.0), -0.3, ("p2",), 1),
    ]

    merged = merge_and_prune_particles(
        particles, merge_tolerance_ft=0.5, keep_top=2, n_realizations=2
    )

    assert len(merged) == 2
    assert merged[0].merged_count == 2
    assert merged[0].tvt[-1] == pytest.approx(11.0)
    assert merged[1].tvt[-1] == pytest.approx(31.0)


def test_particles_to_step_predictions_supports_top1_and_weighted() -> None:
    particles = {
        "a": [
            TrackParticle("a", (1, 2), (10.0, 12.0), 0.0, ("p0",), 1),
            TrackParticle("a", (1, 2), (20.0, 22.0), -10.0, ("p1",), 1),
        ]
    }

    top1 = particles_to_step_predictions(particles, strategy="top1")
    weighted = particles_to_step_predictions(particles, strategy="weighted")

    assert top1["pred_tvt"].tolist() == pytest.approx([10.0, 12.0])
    assert weighted["pred_tvt"].tolist() == pytest.approx([10.0, 12.0], abs=1e-3)


def test_particles_to_frame_caps_merged_count_for_parquet() -> None:
    from mtpnet.track import particles_to_frame

    huge = 10**100
    frame = particles_to_frame(
        {
            "a": [
                TrackParticle("a", (1,), (10.0,), 0.0, ("p0",), huge),
            ]
        }
    )

    assert frame.loc[0, "merged_count"] < 2**63


def test_run_tracker_from_frames_writes_artifacts(tmp_path: Path) -> None:
    windows = pd.DataFrame(
        [
            _mode_window(
                start_step=0,
                logits=[2.0, 0.0],
                paths=[[101.0, 102.0], [130.0, 131.0]],
            ),
            _mode_window(
                start_step=1,
                logits=[2.0, 0.0],
                paths=[[102.0, 103.0], [130.0, 131.0]],
            ),
        ]
    )
    cfg = MTPConfig(
        window=WindowConfig(history_steps=1, future_steps=2, rows_per_step=1),
        run=RunConfig(name="tiny_track", output_dir=tmp_path),
    )

    summary = run_tracker_from_frames(
        run_dir=tmp_path,
        cfg=cfg,
        mode_windows=windows,
        hidden_rows_all=_hidden_rows([1, 2, 3]),
        track_config=TrackConfig(keep_top=4, n_realizations=4, merge_tolerance_ft=0.5),
    )

    assert (tmp_path / "track_particles.parquet").exists()
    assert (tmp_path / "track_row_predictions.parquet").exists()
    assert (tmp_path / "track_candidates.csv").exists()
    assert (tmp_path / "track_metrics.json").exists()
    assert (tmp_path / "track_report.md").exists()
    assert summary["tracker"]["n_realizations"] == 4
    assert summary["candidates"][0]["rows"] == 3


def test_run_track_split_audit_compares_train_valid_and_nn_ranker(tmp_path: Path) -> None:
    windows = pd.DataFrame(
        [
            _mode_window(
                well_id="a",
                start_step=0,
                logits=[2.0, 0.0],
                paths=[[101.0, 102.0], [130.0, 131.0]],
            ),
            _mode_window(
                well_id="b",
                start_step=0,
                logits=[0.0, 2.0],
                paths=[[101.0, 102.0], [130.0, 131.0]],
            ),
        ]
    )
    windows["window_id"] = ["wa", "wb"]
    ranker_features = pd.DataFrame(
        {
            "window_id": ["wa", "wa", "wb", "wb"],
            "mode_id": [0, 1, 0, 1],
            "predicted_error_ft": [0.0, 30.0, 0.0, 30.0],
        }
    )
    hidden = pd.concat(
        [
            _hidden_rows([1, 2]).assign(well_id="a", id=["a_1", "a_2"]),
            _hidden_rows([1, 2]).assign(well_id="b", id=["b_1", "b_2"]),
        ],
        ignore_index=True,
    )
    cfg = MTPConfig(
        window=WindowConfig(history_steps=1, future_steps=2, rows_per_step=1),
        run=RunConfig(name="tiny_audit", output_dir=tmp_path),
    )

    summary = run_track_split_audit_from_frames(
        run_dir=tmp_path,
        cfg=cfg,
        mode_windows=windows,
        hidden_rows_all=hidden,
        ranker_predictions=ranker_features,
        ranker_train_wells={"a"},
        ranker_valid_wells={"b"},
        track_config=TrackConfig(keep_top=4, n_realizations=4, merge_tolerance_ft=0.5),
    )

    assert set(summary["subsets"]) == {"all_valid", "ranker_train", "ranker_valid"}
    assert summary["subsets"]["ranker_train"]["wells"] == 1
    assert summary["subsets"]["ranker_valid"]["wells"] == 1
    assert "nn" in summary["subsets"]["ranker_valid"]
    assert "ranker" in summary["subsets"]["ranker_valid"]
    assert (tmp_path / "track_split_audit.json").exists()
    assert (tmp_path / "track_split_audit.md").exists()


def test_run_tracker_requires_oof_logits_for_ranker_oof(tmp_path: Path) -> None:
    windows = pd.DataFrame(
        [
            _mode_window(
                start_step=0,
                logits=[2.0, 0.0],
                paths=[[101.0, 102.0], [130.0, 131.0]],
            )
        ]
    )
    with pytest.raises(FileNotFoundError, match="ranker_oof"):
        apply_tracker_logit_source(
            windows,
            logit_source="ranker_oof",
            run_dir=tmp_path,
            ranker_logits=None,
            tau_ft=5.0,
            ranker_beta=0.5,
        )


def test_corr_logit_source_selects_mode_with_higher_corr_score(tmp_path: Path) -> None:
    windows = pd.DataFrame(
        [
            _mode_window(
                start_step=0,
                logits=[0.0, 0.0],
                paths=[[101.0, 102.0], [130.0, 131.0]],
            )
        ]
    )
    windows["corr_scores"] = [np.asarray([-3.0, 3.0], dtype=np.float32)]

    corr_windows, source = apply_tracker_logit_source(
        windows,
        logit_source="corr",
        run_dir=tmp_path,
        ranker_logits=None,
        tau_ft=5.0,
        ranker_beta=0.5,
        corr_beta=1.0,
    )
    nn_windows, _ = apply_tracker_logit_source(
        windows,
        logit_source="corr",
        run_dir=tmp_path,
        ranker_logits=None,
        tau_ft=5.0,
        ranker_beta=0.5,
        corr_beta=0.0,
    )

    assert source == "corr"
    assert int(np.asarray(corr_windows.iloc[0]["logits"]).argmax()) == 1
    np.testing.assert_allclose(corr_windows.iloc[0]["logits"], [-1.0, 1.0])
    np.testing.assert_allclose(nn_windows.iloc[0]["logits"], [0.0, 0.0])


def test_corr_logit_source_requires_corr_scores(tmp_path: Path) -> None:
    windows = pd.DataFrame(
        [
            _mode_window(
                start_step=0,
                logits=[0.0, 0.0],
                paths=[[101.0, 102.0], [130.0, 131.0]],
            )
        ]
    )

    with pytest.raises(ValueError, match="corr_scores"):
        apply_tracker_logit_source(
            windows,
            logit_source="corr",
            run_dir=tmp_path,
            ranker_logits=None,
            tau_ft=5.0,
            ranker_beta=0.5,
            corr_beta=1.0,
        )
