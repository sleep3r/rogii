import numpy as np
import pandas as pd
import pytest

from mtpnet.config import WindowConfig
from mtpnet.heatmap import build_channels
from mtpnet.priors import PriorTables
from mtpnet.windows import build_windows_for_well, split_wells


def synthetic_horizontal(n: int = 160) -> pd.DataFrame:
    tvt = np.linspace(1000.0, 1040.0, n)
    gr = 80.0 + np.sin(np.linspace(0.0, 8.0, n)) * 20.0
    tvt_input = tvt.copy()
    tvt_input[n // 2 :] = np.nan
    return pd.DataFrame(
        {"id": [f"row_{i}" for i in range(n)], "TVT": tvt, "TVT_input": tvt_input, "GR": gr}
    )


def synthetic_typewell(n: int = 256) -> pd.DataFrame:
    tvt = np.linspace(960.0, 1080.0, n)
    gr = 80.0 + np.sin(np.linspace(0.0, 12.0, n)) * 20.0
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def test_build_windows_shape_and_metadata() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=32,
        vertical_radius_ft=60.0,
        stride_steps=2,
        max_windows_per_well=3,
    )
    windows = build_windows_for_well("well_a", synthetic_horizontal(), synthetic_typewell(), cfg)
    assert len(windows) > 0
    sample = windows[0]
    assert sample.x.shape == (5, 32, 10)
    assert sample.target_bins.shape == (6,)
    assert sample.target_tvt.shape == (6,)
    assert sample.crop_tvt.shape == (32,)
    assert sample.well_id == "well_a"
    assert np.isfinite(sample.x).all()


def test_typewell_crop_uses_regular_vertical_radius_grid() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=5,
        vertical_radius_ft=20.0,
        stride_steps=2,
        max_windows_per_well=1,
    )
    sample = build_windows_for_well(
        "well_a", synthetic_horizontal(), synthetic_typewell(), cfg
    )[0]

    np.testing.assert_allclose(sample.crop_tvt[0], sample.center_tvt - 20.0)
    np.testing.assert_allclose(sample.crop_tvt[-1], sample.center_tvt + 20.0)
    np.testing.assert_allclose(np.diff(sample.crop_tvt), np.full(4, 10.0), atol=1e-4)


def test_valid_window_mode_uses_only_known_tail_history() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=32,
        vertical_radius_ft=60.0,
        stride_steps=2,
        max_windows_per_well=3,
    )
    horizontal = synthetic_horizontal()
    first_hidden_step = 20
    windows = build_windows_for_well(
        "well_a",
        horizontal,
        synthetic_typewell(),
        cfg,
        history_mode="known_tail_only",
        center_source="tvt_input_tail",
    )

    assert [sample.start_step for sample in windows] == [first_hidden_step - cfg.history_steps]
    assert windows[0].start_step + cfg.history_steps == first_hidden_step
    assert windows[0].center_tvt == windows[0].history_tvt[-1]


def test_base_path_windows_cover_hidden_chunks_after_first() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=32,
        vertical_radius_ft=60.0,
        stride_steps=4,
        max_windows_per_well=10,
    )
    first_hidden_step = 20
    windows = build_windows_for_well(
        "well_a",
        synthetic_horizontal(),
        synthetic_typewell(),
        cfg,
        history_mode="base_path",
        center_source="base_path",
    )

    assert len(windows) > 1
    assert {sample.sample_type for sample in windows} == {"base_center_hidden"}
    assert any(sample.start_step > first_hidden_step - cfg.history_steps for sample in windows)
    assert all(np.isfinite(sample.history_tvt).all() for sample in windows)


def test_prior_conditioned_windows_use_loaded_base_and_prior_channels() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=32,
        vertical_radius_ft=60.0,
        stride_steps=4,
        max_windows_per_well=2,
        channels=("base_sdf", "b2_sdf", "a_density", "base_offset_value"),
    )
    horizontal = synthetic_horizontal()
    tvt = horizontal["TVT"].to_numpy(dtype=np.float32)
    ids = horizontal["id"].astype(str).to_numpy()
    prior_frame = pd.DataFrame(
        {
            "base_tvt": tvt + 5.0,
            "b2_tvt": tvt + 3.0,
            "a_p50_tvt": tvt + 1.0,
            "a_p10_tvt": tvt - 4.0,
            "a_p90_tvt": tvt + 6.0,
        },
        index=ids,
    )
    windows = build_windows_for_well(
        "well_a",
        horizontal,
        synthetic_typewell(),
        cfg,
        history_mode="base_path",
        center_source="base_path",
        prior_tables=PriorTables(frame=prior_frame),
    )

    assert len(windows) > 0
    assert windows[0].x.shape == (4, 32, 10)
    expected_center = np.mean(tvt[76:80] + 5.0)
    assert windows[0].center_tvt == pytest.approx(float(expected_center))
    assert np.isfinite(windows[0].x).all()


def test_prior_conditioned_windows_reconstruct_ids_from_well_and_row_index() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=32,
        vertical_radius_ft=60.0,
        stride_steps=4,
        max_windows_per_well=1,
        channels=("base_sdf",),
    )
    horizontal = synthetic_horizontal().drop(columns=["id"])
    tvt = horizontal["TVT"].to_numpy(dtype=np.float32)
    prior_ids = [f"well_a_{i}" for i in range(len(horizontal))]
    prior_frame = pd.DataFrame({"base_tvt": tvt + 7.0}, index=prior_ids)

    windows = build_windows_for_well(
        "well_a",
        horizontal,
        synthetic_typewell(),
        cfg,
        history_mode="base_path",
        center_source="base_path",
        prior_tables=PriorTables(frame=prior_frame),
    )

    expected_center = np.mean(tvt[76:80] + 7.0)
    assert windows[0].center_tvt == pytest.approx(float(expected_center))


def test_build_channels_supports_robust_z_and_derivative_diff() -> None:
    channels = build_channels(
        horizontal_gr=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        typewell_gr=np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32),
        history_bins=np.array([1.0, np.nan, np.nan], dtype=np.float32),
        finite_steps=np.ones(3, dtype=np.float32),
        channels=("gr_z_diff", "dgr_diff"),
    )

    assert channels.shape == (2, 4, 3)
    assert np.isfinite(channels).all()
    assert abs(float(channels[0].mean())) < 1.0


def test_build_channels_supports_prior_sdf_density_and_value_channels() -> None:
    channels = build_channels(
        horizontal_gr=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        typewell_gr=np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32),
        history_bins=np.array([1.0, np.nan, np.nan], dtype=np.float32),
        finite_steps=np.ones(3, dtype=np.float32),
        channels=(
            "base_sdf",
            "b2_sdf",
            "a_p50_sdf",
            "a_density",
            "a_p10_p90_band",
            "base_offset_value",
            "b2_delta_value",
        ),
        prior_bins={
            "base": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "b2": np.array([2.0, 2.0, 2.0], dtype=np.float32),
            "a_p50": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "a_p10": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "a_p90": np.array([2.0, 3.0, 3.0], dtype=np.float32),
        },
        prior_values={
            "base_offset": np.array([0.0, 0.5, 1.0], dtype=np.float32),
            "b2_delta": np.array([0.0, -0.5, -1.0], dtype=np.float32),
        },
    )

    assert channels.shape == (7, 4, 3)
    assert np.isfinite(channels).all()
    assert channels[0, 1, 0] == pytest.approx(0.0)
    assert channels[1, 2, 1] == pytest.approx(0.0)
    assert channels[3, 1, 0] > channels[3, 3, 0]
    assert channels[4, 1, 0] == pytest.approx(1.0)
    assert channels[4, 3, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(channels[5, 0], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(channels[6, 0], np.array([0.0, -0.5, -1.0]))


def test_split_wells_keeps_well_boundaries() -> None:
    train, valid = split_wells(["a", "b", "c", "d", "e"], valid_fraction=0.4, seed=7)
    assert set(train).isdisjoint(valid)
    assert sorted(train + valid) == ["a", "b", "c", "d", "e"]
    assert len(valid) == 2
