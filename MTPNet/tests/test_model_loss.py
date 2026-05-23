import torch
import pytest

from mtpnet.config import CorrelationHeadConfig, LossConfig, ModelConfig
from mtpnet.loss import (
    corr_vertical_kl_loss,
    mode_corr_scores,
    mtp_loss,
    vertical_corr_target,
)
from mtpnet.model import MTPNet


def test_model_forward_shapes() -> None:
    model = MTPNet(
        in_channels=5,
        height=32,
        width=10,
        future_steps=6,
        cfg=ModelConfig(k_modes=4, conv_channels=(8, 16), hidden_dims=(32,)),
    )
    paths, logits = model(torch.randn(3, 5, 32, 10))
    assert paths.shape == (3, 4, 6)
    assert logits.shape == (3, 4)


def test_model_forward_all_returns_corr_logits_when_enabled() -> None:
    model = MTPNet(
        in_channels=5,
        height=32,
        width=10,
        future_steps=6,
        cfg=ModelConfig(k_modes=4, conv_channels=(8, 16), hidden_dims=(32,)),
        corr_head=CorrelationHeadConfig(enabled=True),
    )
    paths, logits = model(torch.randn(3, 5, 32, 10))
    output = model.forward_all(torch.randn(3, 5, 32, 10))

    assert paths.shape == (3, 4, 6)
    assert logits.shape == (3, 4)
    assert output.paths.shape == (3, 4, 6)
    assert output.logits.shape == (3, 4)
    assert output.raw_paths.shape == (3, 4, 6)
    assert output.corr_logits is not None
    assert output.corr_logits.shape == (3, 32, 10)


def test_corr_head_uses_native_resolution_input_branch() -> None:
    model = MTPNet(
        in_channels=14,
        height=96,
        width=24,
        future_steps=16,
        cfg=ModelConfig(
            k_modes=8,
            conv_channels=(16, 32, 64, 128),
            hidden_dims=(64,),
        ),
        corr_head=CorrelationHeadConfig(enabled=True, source="input_shallow"),
    )

    assert model.corr_head_source == "input_shallow"
    assert model.corr_head_native_resolution is True
    output = model.forward_all(torch.randn(2, 14, 96, 24))

    assert output.corr_logits is not None
    assert output.corr_logits.shape == (2, 96, 24)


def test_model_head_supports_train_batch_size_one() -> None:
    model = MTPNet(
        in_channels=5,
        height=32,
        width=10,
        future_steps=6,
        cfg=ModelConfig(k_modes=4, conv_channels=(8, 16), hidden_dims=(32,)),
    )
    model.train()
    paths, logits = model(torch.randn(1, 5, 32, 10))
    assert paths.shape == (1, 4, 6)
    assert logits.shape == (1, 4)


def test_bounded_model_output_stays_inside_vertical_bins() -> None:
    model = MTPNet(
        in_channels=5,
        height=32,
        width=10,
        future_steps=6,
        cfg=ModelConfig(
            k_modes=4,
            conv_channels=(8, 16),
            hidden_dims=(32,),
            bounded_output=True,
        ),
    )
    paths, _ = model(torch.randn(3, 5, 32, 10))
    paths_detached = paths.detach()

    assert float(paths_detached.min()) >= 0.0
    assert float(paths_detached.max()) <= 31.0


def test_mode_bias_initialization_spreads_mode_centers() -> None:
    model = MTPNet(
        in_channels=5,
        height=64,
        width=10,
        future_steps=4,
        cfg=ModelConfig(
            k_modes=5,
            conv_channels=(8,),
            hidden_dims=(16,),
            bounded_output=True,
            mode_bias_init=True,
            mode_bias_span_bins=20.0,
        ),
    )
    bias = model.path_head.bias.detach().reshape(5, 4)
    centers = (63.0 * torch.sigmoid(bias[:, 0])).tolist()

    assert centers == pytest.approx([11.5, 21.5, 31.5, 41.5, 51.5], abs=0.1)
    assert torch.allclose(bias[:, 0], bias[:, -1])


def test_mtp_loss_prefers_closest_mode_and_backpropagates() -> None:
    pred = torch.tensor([[[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]]], requires_grad=True)
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[1.2, 1.1]])
    loss, metrics = mtp_loss(
        pred, logits, target, LossConfig(alpha_cls=0.2, smooth_lambda=0.0)
    )
    assert metrics["best_k"].tolist() == [2]
    loss.backward()
    assert pred.grad is not None
    assert logits.grad is not None


def test_mtp_loss_uses_classification_warmup_epoch() -> None:
    pred = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]], requires_grad=True)
    logits = torch.tensor([[-4.0, 4.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0]])
    cfg = LossConfig(
        alpha_cls=0.2,
        cls_warmup_epochs=2,
        alpha_cls_warmup_value=0.0,
        smooth_lambda=0.0,
    )

    warmup_loss, warmup_metrics = mtp_loss(pred, logits, target, cfg, epoch=1)
    full_loss, full_metrics = mtp_loss(pred, logits, target, cfg, epoch=3)

    assert warmup_metrics["alpha_cls_effective"] == pytest.approx(0.0)
    assert full_metrics["alpha_cls_effective"] == pytest.approx(0.2)
    assert warmup_loss.item() == pytest.approx(0.0)
    assert full_loss.item() > warmup_loss.item()


def test_mtp_loss_entropy_reward_and_diversity_margin() -> None:
    target = torch.tensor([[1.0, 1.0]])
    pred = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], requires_grad=True)
    low_entropy_logits = torch.tensor([[6.0, -6.0]], requires_grad=True)
    high_entropy_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    cfg = LossConfig(
        alpha_cls=0.0,
        smooth_lambda=0.0,
        entropy_lambda=0.1,
        diversity_lambda=0.5,
        diversity_margin_bins=3.0,
    )

    low_loss, low_metrics = mtp_loss(pred, low_entropy_logits, target, cfg, epoch=1)
    high_loss, high_metrics = mtp_loss(pred, high_entropy_logits, target, cfg, epoch=1)

    assert high_metrics["entropy_loss"] > low_metrics["entropy_loss"]
    assert high_loss.item() < low_loss.item()
    assert high_metrics["diversity_loss"] == pytest.approx(3.0)


def test_mtp_loss_soft_probability_calibration_prefers_error_order() -> None:
    target = torch.tensor([[0.0, 0.0]])
    pred = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [6.0, 6.0]]], requires_grad=True)
    aligned_logits = torch.tensor([[3.0, 1.0, -2.0]], requires_grad=True)
    reversed_logits = torch.tensor([[-2.0, 1.0, 3.0]], requires_grad=True)
    cfg = LossConfig(
        alpha_cls=0.0,
        smooth_lambda=0.0,
        soft_prob_alpha=0.5,
        soft_prob_tau_bins=2.0,
    )

    aligned_loss, aligned_metrics = mtp_loss(
        pred, aligned_logits, target, cfg, epoch=1
    )
    reversed_loss, reversed_metrics = mtp_loss(
        pred, reversed_logits, target, cfg, epoch=1
    )

    assert aligned_metrics["soft_prob_loss"] < reversed_metrics["soft_prob_loss"]
    assert aligned_loss.item() < reversed_loss.item()
    aligned_loss.backward()
    assert aligned_logits.grad is not None


def test_mtp_loss_top3_margin_penalizes_best_mode_outside_top3() -> None:
    target = torch.tensor([[0.0, 0.0]])
    pred = torch.tensor(
        [[[0.0, 0.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]],
        requires_grad=True,
    )
    outside_top3_logits = torch.tensor([[0.0, 4.0, 3.0, 2.0, 1.0]], requires_grad=True)
    inside_top3_logits = torch.tensor([[3.0, 4.0, 2.0, 1.0, 0.0]], requires_grad=True)
    cfg = LossConfig(
        alpha_cls=0.0,
        smooth_lambda=0.0,
        top3_margin_alpha=0.5,
        top3_margin=0.0,
    )

    outside_loss, outside_metrics = mtp_loss(
        pred, outside_top3_logits, target, cfg, epoch=1
    )
    inside_loss, inside_metrics = mtp_loss(
        pred, inside_top3_logits, target, cfg, epoch=1
    )

    assert outside_metrics["top3_margin_loss"] > 0.0
    assert inside_metrics["top3_margin_loss"] == pytest.approx(0.0)
    assert outside_loss.item() > inside_loss.item()


def test_mtp_loss_continuation_probability_prefers_smooth_history_extension() -> None:
    target = torch.tensor([[2.0, 2.0]])
    pred = torch.tensor(
        [[[2.0, 2.0], [8.0, 8.0], [20.0, 20.0]]],
        requires_grad=True,
    )
    aligned_logits = torch.tensor([[3.0, 0.0, -3.0]], requires_grad=True)
    reversed_logits = torch.tensor([[-3.0, 0.0, 3.0]], requires_grad=True)
    history_bins = torch.tensor([[0.0, 1.0]])
    cfg = LossConfig(
        alpha_cls=0.0,
        smooth_lambda=0.0,
        continuation_alpha=0.5,
        continuation_tau_bins=2.0,
    )

    aligned_loss, aligned_metrics = mtp_loss(
        pred, aligned_logits, target, cfg, epoch=1, history_bins=history_bins
    )
    reversed_loss, reversed_metrics = mtp_loss(
        pred, reversed_logits, target, cfg, epoch=1, history_bins=history_bins
    )

    assert aligned_metrics["continuation_loss"] < reversed_metrics["continuation_loss"]
    assert aligned_loss.item() < reversed_loss.item()


def test_vertical_corr_target_sums_to_one_and_peaks_at_true_bin() -> None:
    target_bins = torch.tensor([[1.0, 3.0]])

    target = vertical_corr_target(target_bins, height=5, tau_bins=1.0)

    assert target.shape == (1, 5, 2)
    assert torch.allclose(target.sum(dim=1), torch.ones(1, 2))
    assert target[0, :, 0].argmax().item() == 1
    assert target[0, :, 1].argmax().item() == 3


def test_corr_vertical_kl_loss_is_lower_when_logits_peak_at_target_bin() -> None:
    target_bins = torch.tensor([[1.0, 3.0]])
    good_logits = torch.full((1, 5, 2), -4.0)
    good_logits[0, 1, 0] = 4.0
    good_logits[0, 3, 1] = 4.0
    bad_logits = torch.full((1, 5, 2), -4.0)
    bad_logits[0, 4, 0] = 4.0
    bad_logits[0, 0, 1] = 4.0

    good_loss = corr_vertical_kl_loss(good_logits, target_bins, tau_bins=1.0)
    bad_loss = corr_vertical_kl_loss(bad_logits, target_bins, tau_bins=1.0)

    assert good_loss.item() < bad_loss.item()


def test_corr_vertical_kl_loss_masks_out_of_crop_targets_and_supports_per_sample() -> None:
    corr_logits = torch.zeros(2, 5, 2)
    target_bins = torch.tensor([[-10.0, 20.0], [1.0, 3.0]])

    per_sample = corr_vertical_kl_loss(
        corr_logits,
        target_bins,
        tau_bins=1.0,
        reduction="none",
    )

    assert per_sample.shape == (2,)
    assert per_sample[0].item() == pytest.approx(0.0)
    assert per_sample[1].item() > 0.0
    weighted = (per_sample * torch.tensor([10.0, 0.5])).mean()
    assert weighted.item() == pytest.approx(0.5 * per_sample[1].item() / 2.0)


def test_mode_corr_scores_prefer_path_on_high_corr_ridge() -> None:
    corr_logits = torch.full((1, 6, 5), -4.0)
    corr_logits[0, 1, 2] = 4.0
    corr_logits[0, 2, 3] = 4.0
    paths = torch.tensor([[[1.0, 2.0], [4.0, 4.0]]])

    scores = mode_corr_scores(corr_logits, paths, future_start=2)

    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]


def test_mode_corr_scores_uses_bilinear_subbin_sampling() -> None:
    corr_logits = torch.full((1, 4, 3), -8.0)
    corr_logits[0, 1, 1:] = 4.0
    corr_logits[0, 2, 1:] = 0.0
    paths = torch.tensor([[[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]]])

    scores = mode_corr_scores(corr_logits, paths, future_start=1)

    assert scores[0, 0] > scores[0, 1] > scores[0, 2]
