from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _tiny_well(n: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    tvt = np.arange(n, dtype=np.float32) * 5.0 + 1000.0
    z = -0.4 * tvt + 20.0 * np.sin(np.arange(n, dtype=np.float32) / 5.0)
    gr = np.sin(tvt / 25.0) * 30.0 + np.cos(tvt / 12.0) * 8.0 + 80.0
    tvt_input = tvt.copy()
    tvt_input[10:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(n, dtype=np.float32) * 10.0,
            "X": np.arange(n, dtype=np.float32),
            "Y": np.zeros(n, dtype=np.float32),
            "Z": z,
            "GR": gr,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": tvt, "GR": gr})
    return horizontal, typewell


def test_soft_segment_sample_has_finite_channels_and_soft_targets() -> None:
    from mtpnet.soft_segment import SoftSegmentConfig, build_soft_segment_sample

    horizontal, typewell = _tiny_well()
    sample = build_soft_segment_sample(
        "well_a",
        horizontal,
        typewell,
        SoftSegmentConfig(rows_per_step=2, vertical_bins=32),
    )

    assert sample is not None
    assert sample.image.shape[0] == len(sample.channel_names)
    assert sample.image.shape[1:] == sample.target.shape
    assert sample.image.shape[1] == 32
    assert np.isfinite(sample.image).all()
    assert np.isfinite(sample.target).all()
    assert np.allclose(sample.target.sum(axis=0), 1.0, atol=1e-5)
    assert "TVT" not in sample.channel_names
    assert "Geology" not in sample.channel_names


def test_xcorr_sanity_channels_can_run_without_sdf_shortcuts() -> None:
    from mtpnet.soft_segment import SoftSegmentConfig, build_soft_segment_sample

    horizontal, typewell = _tiny_well()
    sample = build_soft_segment_sample(
        "well_a",
        horizontal,
        typewell,
        SoftSegmentConfig(
            rows_per_step=1,
            vertical_bins=24,
            xcorr_radii=(1, 2),
            use_sdf_channels=False,
            use_pointwise_gr_diff=False,
            use_typewell_gr_channel=False,
        ),
    )

    assert sample is not None
    assert "xcorr_r1" in sample.channel_names
    assert "xcorr_r2" in sample.channel_names
    assert "bridge_sdf" not in sample.channel_names
    assert "plane_prior_sdf" not in sample.channel_names
    assert "typewell_gr" not in sample.channel_names
    xcorr = sample.image[sample.channel_names.index("xcorr_r1")]
    first_true_bin = int(round(float(sample.target_bins[0])))
    assert abs(int(np.argmax(xcorr[:, 0])) - first_true_bin) <= 2


def test_vertical_kl_loss_prefers_logits_near_target() -> None:
    from mtpnet.soft_segment import vertical_kl_loss, vertical_soft_target

    target_bins = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    target = torch.tensor(vertical_soft_target(target_bins, 12, tau_bins=1.0)).unsqueeze(0)
    good = torch.full((1, 12, 3), -5.0)
    bad = torch.full((1, 12, 3), -5.0)
    for step, bin_idx in enumerate([4, 5, 6]):
        good[0, bin_idx, step] = 5.0
        bad[0, 0, step] = 5.0

    assert vertical_kl_loss(good, target) < vertical_kl_loss(bad, target)


def test_viterbi_decode_recovers_smooth_diagonal_and_suppresses_spike() -> None:
    from mtpnet.soft_segment import viterbi_decode_logprobs

    h, w = 20, 8
    logits = np.full((h, w), -6.0, dtype=np.float32)
    true = np.arange(5, 13, dtype=np.int64)
    for step, bin_idx in enumerate(true):
        logits[bin_idx, step] = 5.0
    logits[18, 4] = 7.0  # tempting one-step spike

    decoded = viterbi_decode_logprobs(logits, jump_penalty=0.15, curvature_penalty=0.0)

    assert decoded.shape == (w,)
    assert abs(int(decoded[4]) - int(true[4])) <= 1
    assert np.mean(np.abs(decoded - true)) <= 1.0


def test_viterbi_decode_can_use_logsoftmax_and_hard_jump_window() -> None:
    from mtpnet.soft_segment import viterbi_decode_logprobs

    h, w = 20, 8
    logits = np.full((h, w), -6.0, dtype=np.float32)
    true = np.arange(5, 13, dtype=np.int64)
    for step, bin_idx in enumerate(true):
        logits[bin_idx, step] = 5.0
    logits[18, 4] = 9.0

    greedy = viterbi_decode_logprobs(
        logits,
        jump_penalty=0.0,
        max_jump_bins=None,
        normalize_scores=False,
    )
    decoded = viterbi_decode_logprobs(
        logits,
        jump_penalty=0.03,
        max_jump_bins=4,
        normalize_scores=True,
    )

    assert int(greedy[4]) == 18
    assert abs(int(decoded[4]) - int(true[4])) <= 1


def test_soft_segment_net_forward_shape() -> None:
    from mtpnet.soft_segment import SoftSegmentNet

    model = SoftSegmentNet(in_channels=10, hidden_channels=16)
    logits = model(torch.zeros(2, 10, 32, 9))

    assert logits.shape == (2, 32, 9)


def test_soft_segment_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.soft_segment import SoftSegmentConfig, run_soft_segment

    data = tmp_path / "data" / "train"
    data.mkdir(parents=True)
    for well_id, shift in [("a", 0.0), ("b", 3.0), ("c", -3.0), ("d", 6.0)]:
        horizontal, typewell = _tiny_well()
        horizontal["TVT"] = horizontal["TVT"] + shift
        horizontal["TVT_input"] = horizontal["TVT_input"] + shift
        typewell["TVT"] = typewell["TVT"] + shift
        horizontal.to_csv(data / f"{well_id}__horizontal_well.csv", index=False)
        typewell.to_csv(data / f"{well_id}__typewell.csv", index=False)

    run_soft_segment(
        SoftSegmentConfig(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "soft_segment",
            k_wells=-1,
            rows_per_step=2,
            vertical_bins=32,
            epochs=1,
            batch_size=2,
            hidden_channels=8,
            n_folds=2,
            seed=7,
        )
    )

    metrics = json.loads((tmp_path / "soft_segment" / "metrics.json").read_text())
    assert metrics["num_valid_samples"] > 0
    assert "emission_top10_rate" in metrics
    assert (tmp_path / "soft_segment" / "window_predictions.parquet").exists()
    assert (tmp_path / "soft_segment" / "report.md").exists()


def test_example_figure_handles_xcorr_only_samples(tmp_path: Path) -> None:
    from mtpnet.soft_segment import (
        SoftSegmentConfig,
        SoftSegmentNet,
        _write_example_figure,
        build_soft_segment_sample,
    )

    horizontal, typewell = _tiny_well()
    sample = build_soft_segment_sample(
        "well_a",
        horizontal,
        typewell,
        SoftSegmentConfig(
            rows_per_step=1,
            vertical_bins=24,
            xcorr_radii=(1,),
            use_sdf_channels=False,
            use_pointwise_gr_diff=False,
            use_typewell_gr_channel=False,
        ),
    )
    assert sample is not None

    model = SoftSegmentNet(sample.image.shape[0], hidden_channels=8)
    out = tmp_path / "figure.png"
    _write_example_figure(model, sample, SoftSegmentConfig(), torch.device("cpu"), out)

    assert out.exists()
