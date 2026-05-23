import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mtpnet.config import AugmentationConfig, MTPConfig, TrainConfig, WindowConfig
from mtpnet.train import (
    _augment_prior_conditioning_samples,
    _epoch_progress_record,
    _evaluate,
    _sanity_samples,
    _static_mode_metrics,
    resolve_device,
)
from mtpnet.windows import WindowSample


class FixedPredictionModel(nn.Module):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        paths = torch.tensor(
            [
                [[10.0, 10.0], [0.0, 2.0]],
                [[9.0, 11.0], [12.0, 12.0]],
            ],
            dtype=torch.float32,
            device=x.device,
        )[:batch]
        logits = torch.tensor(
            [
                [0.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
            device=x.device,
        )[:batch]
        return paths, logits


class OOBPredictionModel(nn.Module):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        paths = torch.tensor(
            [[[-1.0, 0.0], [1.0, 65.0]]],
            dtype=torch.float32,
            device=x.device,
        ).repeat(batch, 1, 1)
        logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32, device=x.device).repeat(
            batch, 1
        )
        return paths, logits


class BoundedRawPredictionModel(nn.Module):
    bounded_output = True

    def forward_raw(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        raw_paths = torch.tensor(
            [[[-1.0, 0.0], [1.0, 65.0]]],
            dtype=torch.float32,
            device=x.device,
        ).repeat(batch, 1, 1)
        logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32, device=x.device).repeat(
            batch, 1
        )
        return raw_paths, logits

    def bound_paths(self, raw_paths: Tensor) -> Tensor:
        return 63.0 * torch.sigmoid(raw_paths)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raw_paths, logits = self.forward_raw(x)
        return self.bound_paths(raw_paths), logits


def make_sample(well_id: str, target: np.ndarray) -> WindowSample:
    crop_tvt = np.arange(64, dtype=np.float32) * 10.0
    return WindowSample(
        x=np.zeros((5, 4, 2), dtype=np.float32),
        target_bins=target.astype(np.float32),
        target_tvt=(target * 10.0).astype(np.float32),
        history_tvt=np.array([0.0], dtype=np.float32),
        crop_tvt=crop_tvt,
        well_id=well_id,
        start_step=0,
        center_tvt=0.0,
    )


def test_evaluate_reports_best_mode_mae_and_classification_accuracy() -> None:
    cfg = MTPConfig(train=TrainConfig(batch_size=2, device="cpu"))
    samples = [
        make_sample("a", np.array([0.0, 0.0])),
        make_sample("b", np.array([10.0, 10.0])),
    ]
    metrics, predictions = _evaluate(FixedPredictionModel(), samples, cfg, torch.device("cpu"))

    assert metrics["oracle_topk_rmse_bins"] == pytest.approx(np.mean([np.sqrt(2.0), 1.0]))
    assert metrics["best_mode_mae_bins"] == pytest.approx(1.0)
    assert metrics["classification_accuracy_best_mode"] == pytest.approx(0.5)
    assert metrics["oracle_topk_rmse_ft"] == pytest.approx(np.mean([np.sqrt(200.0), 10.0]))
    assert metrics["weighted_mean_rmse_ft"] > 0.0
    assert metrics["mode_entropy_mean"] > 0.0
    assert metrics["target_bin_min"] == pytest.approx(0.0)
    assert metrics["target_bin_max"] == pytest.approx(10.0)
    assert "top1_pred_tvt" in predictions.columns
    assert "target_in_crop_rate" in metrics


def test_evaluate_reports_logit_ranking_metrics() -> None:
    cfg = MTPConfig(train=TrainConfig(batch_size=2, device="cpu"))
    samples = [
        make_sample("a", np.array([0.0, 0.0])),
        make_sample("b", np.array([10.0, 10.0])),
    ]

    metrics, predictions = _evaluate(FixedPredictionModel(), samples, cfg, torch.device("cpu"))

    assert metrics["oracle_top3_by_logit_rmse_ft"] == pytest.approx(
        metrics["oracle_topk_rmse_ft"]
    )
    assert metrics["oracle_top5_by_logit_rmse_ft"] == pytest.approx(
        metrics["oracle_topk_rmse_ft"]
    )
    assert metrics["best_mode_rank_by_logit_mean"] == pytest.approx(1.5)
    assert metrics["best_mode_rank_by_logit_median"] == pytest.approx(1.5)
    assert metrics["best_mode_top1_rate"] == pytest.approx(0.5)
    assert metrics["best_mode_top3_rate"] == pytest.approx(1.0)
    assert metrics["best_mode_top5_rate"] == pytest.approx(1.0)
    assert metrics["logit_error_spearman"] == pytest.approx(0.0)
    assert "best_mode_rank_by_logit" in predictions.columns


def test_evaluate_reports_prediction_bin_oob_metrics() -> None:
    cfg = MTPConfig(train=TrainConfig(batch_size=1, device="cpu"))
    samples = [make_sample("a", np.array([0.0, 0.0]))]

    metrics, predictions = _evaluate(
        OOBPredictionModel(), samples, cfg, torch.device("cpu")
    )

    assert metrics["pred_bin_oob_frac"] == pytest.approx(0.5)
    assert metrics["top1_pred_bin_oob_frac"] == pytest.approx(0.5)
    assert metrics["weighted_pred_bin_oob_frac"] == pytest.approx(0.5)
    assert metrics["pred_bin_min"] == pytest.approx(-1.0)
    assert metrics["pred_bin_max"] == pytest.approx(65.0)
    assert predictions.loc[0, "top1_pred_bin_oob_frac"] == pytest.approx(0.5)
    assert predictions.loc[0, "weighted_pred_bin_oob_frac"] == pytest.approx(0.5)


def test_evaluate_reports_raw_prediction_oob_before_bound() -> None:
    cfg = MTPConfig(train=TrainConfig(batch_size=1, device="cpu"))
    samples = [make_sample("a", np.array([0.0, 0.0]))]

    metrics, predictions = _evaluate(
        BoundedRawPredictionModel(), samples, cfg, torch.device("cpu")
    )

    assert metrics["bounded_output"] is True
    assert metrics["pred_bin_oob_frac"] == pytest.approx(0.0)
    assert metrics["raw_path_oob_frac_before_bound"] == pytest.approx(0.5)
    assert predictions.loc[0, "raw_path_oob_frac_before_bound"] == pytest.approx(0.5)


def test_static_mode_baseline_reports_fixed_mode_oracle() -> None:
    cfg = MTPConfig(
        train=TrainConfig(batch_size=2, device="cpu"),
        window=WindowConfig(vertical_bins=64, future_steps=2),
    )
    samples = [
        make_sample("a", np.array([0.0, 0.0])),
        make_sample("b", np.array([63.0, 63.0])),
    ]

    metrics = _static_mode_metrics(samples, cfg, torch.device("cpu"))

    assert metrics["static_top1_rmse_ft"] > metrics["static_oracle_topk_rmse_ft"]
    assert metrics["static_weighted_mean_rmse_ft"] < metrics["static_top1_rmse_ft"]
    assert metrics["static_oracle_topk_rmse_ft"] > 0.0


def test_sanity_samples_can_ablate_gr_history_and_prior_channels() -> None:
    cfg = MTPConfig(
        train=TrainConfig(seed=11),
        window=WindowConfig(
            channels=(
                "gr_diff",
                "history_mask",
                "base_sdf",
                "b2_sdf",
                "a_density",
            )
        ),
    )
    sample = WindowSample(
        x=np.ones((5, 4, 2), dtype=np.float32),
        target_bins=np.array([0.0, 0.0], dtype=np.float32),
        target_tvt=np.array([0.0, 0.0], dtype=np.float32),
        history_tvt=np.array([0.0], dtype=np.float32),
        crop_tvt=np.arange(64, dtype=np.float32),
        well_id="a",
        start_step=0,
        center_tvt=0.0,
    )

    no_gr = _sanity_samples([sample], cfg, kind="no_gr")[0]
    no_prior = _sanity_samples([sample], cfg, kind="no_base_b2_a")[0]
    prior_only = _sanity_samples([sample], cfg, kind="base_b2_a_only")[0]

    assert np.all(no_gr.x[0] == 0.0)
    assert np.all(no_gr.x[1:] == 1.0)
    assert np.all(no_prior.x[2:] == 0.0)
    assert np.all(no_prior.x[:2] == 1.0)
    assert np.all(prior_only.x[:2] == 0.0)
    assert np.all(prior_only.x[2:] == 1.0)


def test_prior_augmentation_can_drop_anchor_and_jitter_anchor_channels() -> None:
    cfg = MTPConfig(
        train=TrainConfig(seed=11),
        window=WindowConfig(
            vertical_bins=4,
            vertical_radius_ft=40.0,
            channels=("anchor_sdf", "anchor_offset_value", "b2_sdf", "a_density"),
        ),
        augmentation=AugmentationConfig(
            enabled=True,
            drop_anchor_sdf_prob=1.0,
            drop_b2_sdf_prob=0.0,
            drop_a_density_prob=0.0,
            drop_all_priors_prob=0.0,
        ),
    )
    sample = WindowSample(
        x=np.ones((4, 4, 2), dtype=np.float32),
        target_bins=np.array([0.0, 0.0], dtype=np.float32),
        target_tvt=np.array([0.0, 0.0], dtype=np.float32),
        history_tvt=np.array([0.0], dtype=np.float32),
        crop_tvt=np.linspace(-40.0, 40.0, 4, dtype=np.float32),
        well_id="a",
        start_step=0,
        center_tvt=0.0,
    )

    augmented = _augment_prior_conditioning_samples([sample], cfg)[0]

    assert np.all(augmented.x[0] == 0.0)
    assert np.all(augmented.x[1] == 0.0)
    assert np.all(augmented.x[2:] == 1.0)


def test_sanity_samples_can_remove_anchor_and_jitter_anchor() -> None:
    cfg = MTPConfig(
        window=WindowConfig(
            vertical_bins=4,
            vertical_radius_ft=40.0,
            channels=("anchor_sdf", "anchor_offset_value", "b2_sdf"),
        )
    )
    x = np.ones((3, 4, 2), dtype=np.float32)
    sample = WindowSample(
        x=x,
        target_bins=np.array([0.0, 0.0], dtype=np.float32),
        target_tvt=np.array([0.0, 0.0], dtype=np.float32),
        history_tvt=np.array([0.0], dtype=np.float32),
        crop_tvt=np.linspace(-40.0, 40.0, 4, dtype=np.float32),
        well_id="a",
        start_step=0,
        center_tvt=0.0,
    )

    no_anchor = _sanity_samples([sample], cfg, kind="no_anchor")[0]
    jittered = _sanity_samples([sample], cfg, kind="anchor_jitter_20ft")[0]

    assert np.all(no_anchor.x[0] == 0.0)
    assert np.all(no_anchor.x[1] == 0.0)
    assert np.all(no_anchor.x[2] == 1.0)
    assert not np.allclose(jittered.x[0], sample.x[0])
    assert not np.allclose(jittered.x[1], sample.x[1])


def test_auto_device_prefers_mps_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("mps")


def test_epoch_progress_record_exposes_training_log_fields() -> None:
    record = _epoch_progress_record(
        epoch=3,
        train_loss=1.25,
        valid_score=2.5,
        valid_metrics={
            "oracle_topk_rmse_bins": 2.0,
            "weighted_mean_rmse_bins": 3.0,
            "top1_rmse_bins": 4.0,
            "oracle_topk_rmse_ft": 10.0,
        },
        is_best=True,
    )

    assert record == {
        "event": "epoch",
        "epoch": 3,
        "train_loss": 1.25,
        "valid_score": 2.5,
        "valid_oracle_topk_rmse_bins": 2.0,
        "valid_weighted_mean_rmse_bins": 3.0,
        "valid_top1_rmse_bins": 4.0,
        "valid_oracle_topk_rmse_ft": 10.0,
        "is_best": True,
    }
