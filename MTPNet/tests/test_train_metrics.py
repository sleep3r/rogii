import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mtpnet.config import MTPConfig, TrainConfig
from mtpnet.train import _evaluate, resolve_device
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


def test_auto_device_prefers_mps_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("mps")
