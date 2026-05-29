from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_traceback_config_defaults_are_schema_safe() -> None:
    from mtpnet.traceback_events import TracebackConfig

    cfg = TracebackConfig()

    assert cfg.rows_per_step == 32
    assert cfg.patch_radii == (3, 5, 9, 15)
    assert cfg.min_prominence_z > 0.0


def test_traceback_schema_guard_rejects_forbidden_columns() -> None:
    from mtpnet.traceback_events import assert_traceback_feature_schema_safe

    with pytest.raises(ValueError, match="forbidden"):
        assert_traceback_feature_schema_safe(["MD", "GR", "TVT", "Geology"])

    assert_traceback_feature_schema_safe(["MD", "X", "Y", "Z", "GR", "TVT_input"])


def test_compress_well_rows_keeps_partial_tail_and_known_mask() -> None:
    from mtpnet.traceback_events import compress_well_rows

    frame = pd.DataFrame(
        {
            "id": [f"w_{i}" for i in range(5)],
            "well_id": ["w"] * 5,
            "row_idx": np.arange(5),
            "MD": np.arange(5, dtype=float),
            "X": np.zeros(5),
            "Y": np.zeros(5),
            "Z": -np.arange(5, dtype=float),
            "GR": [1.0, 3.0, np.nan, 7.0, 9.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    comp = compress_well_rows(frame, rows_per_step=2)

    assert comp["step"].tolist() == [0, 1, 2]
    assert comp["row_start"].tolist() == [0, 2, 4]
    assert comp["row_end"].tolist() == [1, 3, 4]
    assert comp["known_mask"].tolist() == [True, False, False]
    assert np.isfinite(comp["GR_filled"]).all()


def test_extract_traceback_events_finds_peak_and_trough() -> None:
    from mtpnet.traceback_events import TracebackConfig, extract_traceback_events

    comp = pd.DataFrame(
        {
            "well_id": ["w"] * 9,
            "step": np.arange(9),
            "row_start": np.arange(9),
            "row_end": np.arange(9),
            "MD": np.arange(9, dtype=float),
            "X": np.zeros(9),
            "Y": np.zeros(9),
            "Z": -np.arange(9, dtype=float),
            "GR_filled": [0.0, 1.0, 5.0, 1.0, 0.0, -1.0, -4.0, -1.0, 0.0],
            "GR_finite_frac": np.ones(9),
            "TVT_input": [
                100,
                101,
                102,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "known_mask": [True, True, True, False, False, False, False, False, False],
        }
    )

    events = extract_traceback_events(
        comp, TracebackConfig(patch_radii=(1,), min_prominence_z=0.5)
    )

    assert {"peak", "trough"}.issubset(set(events["event_type"]))
    assert set(events["well_id"]) == {"w"}
    assert {"patch_gr", "patch_dgr", "patch_radius"}.issubset(events.columns)


def test_same_well_dictionary_uses_only_known_tvt_input_events() -> None:
    from mtpnet.traceback_dictionary import build_same_well_dictionary

    events = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "step": [1, 5],
            "known_mask": [True, False],
            "TVT_input": [101.0, np.nan],
            "true_TVT": [101.0, 105.0],
            "event_type": ["peak", "peak"],
            "prominence": [1.0, 1.0],
            "finite_frac": [1.0, 1.0],
            "patch_radius": [1, 1],
            "patch_gr": [np.array([0, 1, 0], dtype=np.float32)] * 2,
            "patch_dgr": [np.array([1, 0, -1], dtype=np.float32)] * 2,
        }
    )

    dictionary = build_same_well_dictionary(events)

    assert len(dictionary) == 1
    assert dictionary["source"].iloc[0] == "same_well_known"
    assert dictionary["source_TVT"].iloc[0] == 101.0


def test_fold_safe_dictionary_excludes_validation_wells() -> None:
    from mtpnet.traceback_dictionary import build_fold_safe_train_dictionary

    events = pd.DataFrame(
        {
            "well_id": ["train", "valid"],
            "step": [1, 1],
            "true_TVT": [101.0, 201.0],
            "known_mask": [False, False],
            "event_type": ["peak", "peak"],
            "prominence": [1.0, 1.0],
            "finite_frac": [1.0, 1.0],
            "patch_radius": [1, 1],
            "patch_gr": [np.array([0, 1, 0], dtype=np.float32)] * 2,
            "patch_dgr": [np.array([1, 0, -1], dtype=np.float32)] * 2,
        }
    )

    dictionary = build_fold_safe_train_dictionary(events, train_wells=["train"])

    assert dictionary["source_well_id"].tolist() == ["train"]
    assert dictionary["source_TVT"].tolist() == [101.0]


def test_typewell_dictionary_samples_regular_tvt_grid() -> None:
    from mtpnet.traceback_dictionary import build_typewell_dictionary

    typewell = pd.DataFrame(
        {
            "TVT": [100.0, 105.0, 110.0, 115.0, 120.0],
            "GR": [0.0, 1.0, 4.0, 1.0, 0.0],
        }
    )

    dictionary = build_typewell_dictionary(
        typewell, vertical_step_ft=5.0, patch_radii=(1,), min_prominence_z=0.5
    )

    assert not dictionary.empty
    assert set(dictionary["source"]) == {"typewell"}
    assert dictionary["source_TVT"].between(100.0, 120.0).all()


def test_patch_score_ranks_exact_match_above_reverse_patch() -> None:
    from mtpnet.traceback_match import score_event_against_dictionary

    event = pd.Series(
        {
            "event_type": "peak",
            "prominence": 1.0,
            "finite_frac": 1.0,
            "patch_radius": 1,
            "patch_gr": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "patch_dgr": np.array([1.0, 0.0, -1.0], dtype=np.float32),
        }
    )
    dictionary = pd.DataFrame(
        {
            "source": ["exact", "bad"],
            "source_well_id": ["a", "b"],
            "source_step": [0, 0],
            "source_TVT": [100.0, 200.0],
            "event_type": ["peak", "trough"],
            "prominence": [1.0, 1.0],
            "finite_frac": [1.0, 1.0],
            "patch_radius": [1, 1],
            "patch_gr": [
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
            ],
            "patch_dgr": [
                np.array([1.0, 0.0, -1.0], dtype=np.float32),
                np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            ],
        }
    )

    scored = score_event_against_dictionary(event, dictionary, location_weight=0.0)

    assert scored.iloc[0]["source_TVT"] == 100.0
    assert scored.iloc[0]["score"] > scored.iloc[1]["score"]


def test_known_tvt_bridge_interpolates_only_from_input_tvt() -> None:
    from mtpnet.traceback import add_known_tvt_bridge

    comp = pd.DataFrame(
        {
            "well_id": ["w"] * 5,
            "step": np.arange(5),
            "TVT_input": [100.0, np.nan, np.nan, np.nan, 140.0],
            "true_TVT": [100.0, 110.0, 9999.0, 130.0, 140.0],
        }
    )

    bridged = add_known_tvt_bridge(comp)

    assert bridged["bridge_TVT"].tolist() == [100.0, 110.0, 120.0, 130.0, 140.0]


def test_location_prior_prefers_nearby_equal_shape_match() -> None:
    from mtpnet.traceback_match import score_event_against_dictionary

    event = pd.Series(
        {
            "event_type": "peak",
            "prominence": 1.0,
            "finite_frac": 1.0,
            "patch_radius": 1,
            "patch_gr": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "patch_dgr": np.array([1.0, 0.0, -1.0], dtype=np.float32),
        }
    )
    dictionary = pd.DataFrame(
        {
            "source": ["near", "far"],
            "source_well_id": ["a", "b"],
            "source_step": [0, 0],
            "source_TVT": [112.0, 500.0],
            "event_type": ["peak", "peak"],
            "prominence": [1.0, 1.0],
            "finite_frac": [1.0, 1.0],
            "patch_radius": [1, 1],
            "patch_gr": [
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ],
            "patch_dgr": [
                np.array([1.0, 0.0, -1.0], dtype=np.float32),
                np.array([1.0, 0.0, -1.0], dtype=np.float32),
            ],
        }
    )

    scored = score_event_against_dictionary(
        event,
        dictionary,
        location_weight=2.0,
        bridge_tvt=110.0,
        location_sigma_ft=50.0,
    )

    assert scored.iloc[0]["source"] == "near"


def test_anchor_guard_rejects_ambiguous_and_far_matches() -> None:
    from mtpnet.traceback_candidates import matches_to_anchors

    matches = pd.DataFrame(
        {
            "event_well_id": ["w", "w", "w", "w", "w", "w"],
            "event_step": [1, 1, 2, 2, 3, 3],
            "source_TVT": [101.0, 300.0, 500.0, 120.0, 130.0, 300.0],
            "score": [1.0, 0.99, 1.0, 0.2, 1.0, 0.4],
            "event_bridge_tvt": [100.0, 100.0, 100.0, 100.0, 128.0, 128.0],
        }
    )

    anchors = matches_to_anchors(
        matches,
        min_score_quantile=0.0,
        min_top1_gap=0.05,
        max_bridge_delta_ft=60.0,
    )

    assert anchors[["well_id", "step", "anchor_tvt"]].to_dict("records") == [
        {"well_id": "w", "step": 3, "anchor_tvt": 130.0}
    ]


def test_make_sanity_events_supports_shuffled_and_zero_gr() -> None:
    from mtpnet.traceback_match import make_sanity_events

    events = pd.DataFrame(
        {
            "well_id": ["w"] * 2,
            "step": [1, 2],
            "patch_gr": [
                np.array([1, 2, 3], dtype=np.float32),
                np.array([4, 5, 6], dtype=np.float32),
            ],
            "patch_dgr": [
                np.array([1, 1, 1], dtype=np.float32),
                np.array([2, 2, 2], dtype=np.float32),
            ],
        }
    )

    variants = make_sanity_events(events, seed=7)

    assert set(variants) == {"normal_GR", "shuffled_hidden_GR", "zero_hidden_GR"}
    assert np.allclose(variants["zero_hidden_GR"].iloc[0]["patch_gr"], 0.0)
    assert len(variants["shuffled_hidden_GR"]) == len(events)


def test_traceback_band_generation_covers_anchor_steps() -> None:
    from mtpnet.traceback_candidates import build_traceback_bands

    comp = pd.DataFrame(
        {
            "well_id": ["w"] * 5,
            "step": np.arange(5),
            "true_TVT": [100, 105, 110, 115, 120],
        }
    )
    anchors = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "step": [1, 3],
            "anchor_tvt": [105.0, 115.0],
            "confidence": [1.0, 1.0],
        }
    )

    bands = build_traceback_bands(comp, anchors, widths_ft=(40.0,))

    assert len(bands) == 5
    assert bands["band_center_tvt"].between(100.0, 120.0).all()
    assert set(bands["band_width_ft"]) == {40.0}


def test_traceback_candidates_do_not_require_hidden_tvt() -> None:
    from mtpnet.traceback_candidates import build_traceback_candidates

    hidden = pd.DataFrame(
        {
            "id": [f"w_{i}" for i in range(4)],
            "well_id": ["w"] * 4,
            "row_idx": np.arange(4),
            "step": np.arange(4),
            "TVT_input": [np.nan] * 4,
        }
    )
    bands = pd.DataFrame(
        {
            "well_id": ["w"] * 4,
            "step": np.arange(4),
            "band_center_tvt": [100.0, 103.0, 106.0, 109.0],
            "band_width_ft": [80.0] * 4,
            "candidate": ["traceback_band_w80"] * 4,
        }
    )

    candidates = build_traceback_candidates(hidden, bands)

    assert {"id", "well_id", "row_idx", "candidate", "pred_tvt"}.issubset(
        candidates.columns
    )
    assert set(candidates["candidate"]) == {"traceback_band_w80"}
    assert "TVT" not in candidates.columns


def test_traceback_candidates_skip_rows_without_band_center() -> None:
    from mtpnet.traceback_candidates import build_traceback_candidates

    hidden = pd.DataFrame(
        {
            "id": [f"w_{i}" for i in range(2)],
            "well_id": ["w", "w"],
            "row_idx": [0, 1],
            "step": [0, 1],
            "TVT_input": [np.nan, np.nan],
        }
    )
    bands = pd.DataFrame(
        {
            "well_id": ["w"],
            "step": [0],
            "band_center_tvt": [100.0],
            "band_width_ft": [80.0],
            "candidate": ["traceback_band_w80"],
        }
    )

    candidates = build_traceback_candidates(hidden, bands)

    assert candidates["id"].tolist() == ["w_0"]
    assert np.isfinite(candidates["pred_tvt"]).all()


def test_traceback_smoke_writes_metrics_report_and_artifacts(tmp_path) -> None:
    from mtpnet.traceback import run_traceback_from_frames

    rows = []
    gr_values = [0, 1, 5, 1, 0, -1, -4, -1, 0, 1, 4, 1, 0, -1, -3, -1]
    for well_id, offset in [("a", 0.0), ("b", 20.0)]:
        for i in range(16):
            tvt = 100.0 + offset + i
            rows.append(
                {
                    "id": f"{well_id}_{i}",
                    "well_id": well_id,
                    "row_idx": i,
                    "MD": 1000.0 + i,
                    "X": float(i),
                    "Y": 0.0,
                    "Z": -float(i),
                    "GR": gr_values[i],
                    "TVT_input": tvt if i < 4 else np.nan,
                    "TVT": tvt,
                }
            )
    frame = pd.DataFrame(rows)
    typewell = pd.DataFrame(
        {"TVT": np.linspace(90, 140, 20), "GR": np.sin(np.linspace(0, 8, 20))}
    )

    metrics = run_traceback_from_frames(
        frame, typewell=typewell, output_dir=tmp_path, k_wells=-1, seed=3
    )

    assert metrics["wells"] == 2
    assert "normal_GR" in metrics["variants"]
    assert (tmp_path / "traceback_metrics.json").exists()
    assert (tmp_path / "traceback_report.md").exists()
    assert (tmp_path / "traceback_events.parquet").exists()
    assert (tmp_path / "traceback_event_matches.parquet").exists()


def test_traceback_candidate_oracle_improves_when_traceback_candidate_is_exact(
    tmp_path,
) -> None:
    from mtpnet.traceback_candidates import evaluate_traceback_candidate_oracle

    hidden = pd.DataFrame(
        {
            "id": ["a", "b"],
            "well_id": ["w", "w"],
            "row_idx": [0, 1],
            "TVT": [100.0, 110.0],
            "b2_tvt": [120.0, 130.0],
        }
    )
    traceback_candidates = pd.DataFrame(
        {
            "id": ["a", "b"],
            "well_id": ["w", "w"],
            "row_idx": [0, 1],
            "candidate": ["traceback_exact", "traceback_exact"],
            "pred_tvt": [100.0, 110.0],
        }
    )

    metrics = evaluate_traceback_candidate_oracle(hidden, traceback_candidates)

    assert metrics["b2_row_rmse"] > 0.0
    assert metrics["traceback_oracle_row_rmse"] == 0.0
    assert metrics["oracle_gain_ft"] > 0.0
