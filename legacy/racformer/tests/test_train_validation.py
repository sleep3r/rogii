from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from racformer.train import _oracle_direct_step_rmse, _validate
from torch import nn
from torch.utils.data import DataLoader


class _FixedPredictionModel(nn.Module):
    def forward(self, batch: dict) -> SimpleNamespace:
        return SimpleNamespace(
            s_pred=torch.zeros_like(batch["s_star"]),
            direct_resid_step=torch.ones_like(batch["direct_resid_step"]),
        )

    def materialize(self, batch: dict, out: SimpleNamespace) -> torch.Tensor:
        if torch.equal(out.s_pred, batch["s_star"]) and torch.equal(
            out.direct_resid_step,
            torch.zeros_like(out.direct_resid_step),
        ):
            return batch["oracle_seg_pred_hidden"]
        return batch["pred_tvt_hidden"]


class _ZeroLoss(nn.Module):
    def forward(self, out: SimpleNamespace, batch: dict, pred_tvt_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"total": pred_tvt_hidden.new_zeros(())}


def test_validate_reports_base_rmse_and_gain_vs_base() -> None:
    batch = {
        "pred_tvt_hidden": torch.tensor([10.0, 12.0, 0.0], dtype=torch.float32),
        "base_tvt_hidden": torch.tensor([10.0, 14.0, 0.0], dtype=torch.float32),
        "tvt_hidden": torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32),
        "n_hidden_rows": torch.tensor(2, dtype=torch.long),
        "s_star": torch.tensor([0.0], dtype=torch.float32),
        "direct_resid_step": torch.tensor([0.0], dtype=torch.float32),
        "hidden_row_to_step": torch.tensor([0, 0, 0], dtype=torch.long),
        "oracle_seg_pred_hidden": torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32),
    }
    loader = DataLoader([batch], batch_size=1)

    pooled_rmse, metrics = _validate(_FixedPredictionModel(), loader, _ZeroLoss(), torch.device("cpu"))

    assert pooled_rmse == pytest.approx(2.0**0.5)
    assert metrics["base_rmse"] == pytest.approx(8.0**0.5)
    assert metrics["gain_vs_base"] == pytest.approx((8.0**0.5) - (2.0**0.5))
    assert metrics["oracle_seg_rmse"] == pytest.approx(0.0)
    assert metrics["oracle_direct_step_rmse"] == pytest.approx(2.0)
    assert metrics["pred_delta_abs_mean"] == pytest.approx(1.0)
    assert metrics["pred_delta_abs_max"] == pytest.approx(2.0)


def test_oracle_direct_step_rmse_uses_step_mean_residuals() -> None:
    batch = {
        "base_tvt_hidden": torch.zeros((1, 4), dtype=torch.float32),
        "tvt_hidden": torch.tensor([[0.0, 2.0, 10.0, 14.0]], dtype=torch.float32),
        "n_hidden_rows": torch.tensor([4], dtype=torch.long),
        "hidden_row_to_step": torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
    }

    rmse = _oracle_direct_step_rmse(batch)

    assert rmse == pytest.approx(2.5**0.5)
