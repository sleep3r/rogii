import numpy as np
import pytest

from mtpnet.config import (
    MTPConfig,
    SyntheticConfig,
    WindowConfig,
)
from mtpnet.synthetic import (
    SyntheticTemplate,
    flip_synthetic_sample,
    generate_synthetic_sample,
    sample_typewell_gr,
    stretch_gr_sequence,
)


def _cfg(**synthetic_overrides: object) -> MTPConfig:
    synthetic_args = {
        "enabled": True,
        "noise_std_range": (0.0, 0.0),
        "amplitude_scale_range": (1.0, 1.0),
        "baseline_shift_range": (0.0, 0.0),
        "dropout_max_frac": 0.0,
        "stretch_range": (1.0, 1.0),
        "bad_prior_prob": 0.0,
        "no_prior_prob": 0.0,
        "seed": 7,
    }
    synthetic_args.update(synthetic_overrides)
    synthetic = SyntheticConfig(**synthetic_args)
    return MTPConfig(
        synthetic=synthetic,
        window=WindowConfig(
            history_steps=2,
            future_steps=3,
            vertical_bins=8,
            vertical_radius_ft=80.0,
            channels=(
                "gr_diff",
                "gr_z_diff",
                "dgr_diff",
                "abs_gr_diff",
                "history_mask",
                "history_sdf",
                "finite_mask",
                "anchor_sdf",
                "b2_sdf",
                "a_p50_sdf",
                "a_density",
                "a_p10_p90_band",
                "anchor_offset_value",
                "b2_delta_value",
            ),
        ),
    )


def _template() -> SyntheticTemplate:
    return SyntheticTemplate(
        crop_tvt=np.linspace(1000.0, 1080.0, 8, dtype=np.float32),
        typewell_gr=np.linspace(10.0, 80.0, 8, dtype=np.float32),
        target_bins=np.array([3.0, 3.5, 4.0], dtype=np.float32),
        history_bins=np.array([2.0, 2.5], dtype=np.float32),
        well_id="synthetic_source",
        start_step=10,
        center_tvt=1040.0,
    )


def test_generated_synthetic_path_stays_within_crop() -> None:
    sample = generate_synthetic_sample(_template(), _cfg(), index=0)

    assert np.nanmin(sample.target_bins) >= 0.0
    assert np.nanmax(sample.target_bins) <= 7.0


def test_noiseless_generated_gr_matches_typewell_sampled_at_target_path() -> None:
    template = _template()
    sample = generate_synthetic_sample(template, _cfg(), index=0)

    expected = sample_typewell_gr(template.typewell_gr, sample.target_bins)

    assert sample.horizontal_gr is not None
    np.testing.assert_allclose(sample.horizontal_gr[-3:], expected, atol=1e-5)


def test_no_prior_variant_zeros_all_prior_channels() -> None:
    sample = generate_synthetic_sample(_template(), _cfg(), index=0, prior_kind="no_prior")
    prior_indices = list(range(7, 14))

    assert np.all(sample.x[prior_indices] == 0.0)


def test_bad_prior_changes_anchor_channels() -> None:
    good = generate_synthetic_sample(_template(), _cfg(), index=0, prior_kind="good")
    bad = generate_synthetic_sample(_template(), _cfg(), index=0, prior_kind="bad")

    assert not np.allclose(good.x[7], bad.x[7])
    assert not np.allclose(good.x[12], bad.x[12])


def test_stretch_changes_gr_timing_but_keeps_target_bins_aligned() -> None:
    sequence = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    stretched = stretch_gr_sequence(sequence, factor=1.25)
    sample = generate_synthetic_sample(
        _template(),
        _cfg(stretch_range=(1.25, 1.25), path_families=("real_residual",)),
        index=0,
    )

    assert not np.allclose(stretched, sequence)
    np.testing.assert_allclose(sample.target_bins, _template().target_bins, atol=1e-5)


def test_horizontal_flip_twice_recovers_target_and_gr_order() -> None:
    sample = generate_synthetic_sample(_template(), _cfg(), index=0)
    flipped = flip_synthetic_sample(sample)
    recovered = flip_synthetic_sample(flipped)

    np.testing.assert_allclose(recovered.target_bins, sample.target_bins)
    assert recovered.horizontal_gr is not None
    assert sample.horizontal_gr is not None
    np.testing.assert_allclose(recovered.horizontal_gr, sample.horizontal_gr)


def test_synthetic_dropout_interpolates_gr_like_real_finite_mask() -> None:
    template = SyntheticTemplate(
        crop_tvt=np.linspace(1000.0, 1080.0, 8, dtype=np.float32),
        typewell_gr=np.array([10.0, 11.0, 17.0, 31.0, 50.0, 80.0, 121.0, 170.0], dtype=np.float32),
        target_bins=np.array([3.0, 3.5, 4.0], dtype=np.float32),
        history_bins=np.array([2.0, 2.5], dtype=np.float32),
        well_id="synthetic_source",
        start_step=10,
        center_tvt=1040.0,
    )
    sample = None
    for index in range(30):
        candidate = generate_synthetic_sample(
            template,
            _cfg(
                dropout_max_frac=1.0,
                path_families=("real_residual",),
                noise_std_range=(0.0, 0.0),
            ),
            index=index,
        )
        finite = candidate.x[6, 0] > 0.5
        if finite.any() and (~finite).any():
            sample = candidate
            break
    assert sample is not None
    assert sample.horizontal_gr is not None
    finite = sample.x[6, 0] > 0.5
    steps = np.arange(len(sample.horizontal_gr), dtype=np.float32)
    expected = np.interp(
        steps,
        steps[finite],
        sample.horizontal_gr[finite],
    ).astype(np.float32)

    np.testing.assert_allclose(sample.horizontal_gr, expected, atol=1e-5)
