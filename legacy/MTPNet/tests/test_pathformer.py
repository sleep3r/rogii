from __future__ import annotations

import numpy as np
import pytest
import torch

from pathformer.dataset import N_FEATURES, WellSample
from pathformer.evaluate import evaluate_samples, predict_row_predictions
from pathformer.config import PFModelConfig
from pathformer.model import PathFormer


class FixedPathModel(torch.nn.Module):
    def __init__(self, pred_delta: list[float]):
        super().__init__()
        self.register_buffer(
            "pred_delta", torch.tensor(pred_delta, dtype=torch.float32).unsqueeze(0)
        )

    def forward(
        self, features: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.pred_delta[:, : features.shape[1]].repeat(features.shape[0], 1)


def _sample() -> WellSample:
    features = np.zeros((3, N_FEATURES), dtype=np.float32)
    return WellSample(
        well_id="well_a",
        features=features,
        target_delta=np.array([0.0, 10.0, 20.0], dtype=np.float32),
        hidden_mask=np.array([False, True, True], dtype=bool),
        seq_len=3,
        last_known_tvt=100.0,
        first_hidden_step=1,
        tail_class="G_all_candidates_fail",
        hidden_row_ids=np.array(["well_a_10", "well_a_11", "well_a_20"], dtype=object),
        hidden_row_idx=np.array([10, 11, 20], dtype=np.int32),
        hidden_row_steps=np.array([1, 1, 2], dtype=np.int32),
        hidden_row_tvt=np.array([110.0, 114.0, 120.0], dtype=np.float32),
        hidden_row_gr=np.array([1.0, np.nan, 3.0], dtype=np.float32),
    )


def test_predict_row_predictions_maps_step_predictions_to_hidden_rows() -> None:
    model = FixedPathModel([0.0, 10.0, 30.0])

    rows = predict_row_predictions(
        model,
        [_sample()],
        max_seq_len=8,
        device=torch.device("cpu"),
        candidate="pathformer_direct",
    )

    assert rows["id"].tolist() == ["well_a_10", "well_a_11", "well_a_20"]
    assert rows["candidate"].unique().tolist() == ["pathformer_direct"]
    assert rows["pred_tvt"].tolist() == pytest.approx([110.0, 110.0, 130.0])
    assert rows["TVT"].tolist() == pytest.approx([110.0, 114.0, 120.0])
    assert rows["tail_class"].unique().tolist() == ["G_all_candidates_fail"]


def test_evaluate_samples_reports_row_level_metrics_without_b2_coupling() -> None:
    model = FixedPathModel([0.0, 10.0, 30.0])

    metrics = evaluate_samples(model, [_sample()], max_seq_len=8, device=torch.device("cpu"))

    expected_rmse = np.sqrt((0.0**2 + 4.0**2 + 10.0**2) / 3.0)
    assert metrics["row_rmse_ft"] == pytest.approx(expected_rmse)
    assert metrics["rows"] == 3
    assert metrics["n_wells"] == 1
    assert metrics["tail_class"]["G_all_candidates_fail"]["row_rmse_ft"] == pytest.approx(
        expected_rmse
    )
    assert "gain_vs_b2" not in metrics
    assert "mean_b2_rmse" not in metrics


def test_pathformer_anchor_skip_starts_from_base_delta_when_enabled() -> None:
    model = PathFormer(
        PFModelConfig(
            d_model=32,
            n_heads=4,
            n_encoder_layers=1,
            ffn_dim=64,
            dropout=0.0,
            anchor_skip=True,
            anchor_feature_index=17,
            anchor_valid_index=22,
            anchor_scale=100.0,
        )
    )
    for param in model.parameters():
        torch.nn.init.zeros_(param)
    features = torch.zeros(1, 3, N_FEATURES)
    features[0, :, 17] = torch.tensor([0.5, 1.0, -0.25])
    features[0, :, 22] = torch.tensor([1.0, 1.0, 0.0])

    pred = model(features)

    assert pred.squeeze(0).tolist() == pytest.approx([50.0, 100.0, 0.0])
