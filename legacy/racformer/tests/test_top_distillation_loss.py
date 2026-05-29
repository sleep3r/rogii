import math
from types import SimpleNamespace

import pytest
import torch
from racformer.config import RACModelConfig, RACTrainConfig
from racformer.loss import RACLoss
from racformer.model import RACFormer, RACFormerOutput, load_state_dict_allowing_top_teacher_heads


def _minimal_batch() -> dict:
    return {
        "n_hidden_rows": torch.tensor([1], dtype=torch.long),
        "tvt_hidden": torch.tensor([[10.0]], dtype=torch.float32),
        "base_tvt_hidden": torch.tensor([[10.0]], dtype=torch.float32),
        "dC_hidden": torch.tensor([[0.0]], dtype=torch.float32),
        "s_star": torch.zeros((1, 2), dtype=torch.float32),
        "hidden_mask": torch.tensor([[False, True, False]], dtype=torch.bool),
        "hidden_row_to_step": torch.tensor([[1]], dtype=torch.long),
        "anchor_step": torch.tensor([0], dtype=torch.long),
        "top_teacher_mask": torch.tensor([[True, True, True]], dtype=torch.bool),
        "top_event_step": torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
        "top_state_step": torch.tensor([[0, 1, 2]], dtype=torch.long),
    }


def test_racformer_outputs_top_event_and_direction_logits() -> None:
    cfg = RACModelConfig(d_model=16, n_heads=4, enc_layers=1, dec_layers=1, ff_dim=32, k_seg=2)
    model = RACFormer(cfg)
    batch = {
        "features": torch.zeros((2, 5, 74), dtype=torch.float32),
        "region_ids": torch.zeros((2, 5), dtype=torch.long),
        "pad_mask": torch.zeros((2, 5), dtype=torch.bool),
    }

    out = model(batch)

    assert out.top_event_logits.shape == (2, 5)
    assert out.top_dir_logits.shape == (2, 5, 3)


def test_model_loader_allows_old_checkpoints_without_top_direction_head() -> None:
    model = RACFormer(RACModelConfig(d_model=16, n_heads=4, enc_layers=1, dec_layers=1, ff_dim=32, k_seg=2))
    old_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("top_dir_head.")
    }

    load_state_dict_allowing_top_teacher_heads(model, old_state)


def test_racloss_adds_masked_top_event_and_direction_terms() -> None:
    train_cfg = RACTrainConfig(
        w_tvt_mse=0.0,
        w_tvt_huber=0.0,
        w_endpoint=0.0,
        w_seg=0.0,
        w_local=0.0,
        w_smooth=0.0,
        w_direct_reg=0.0,
        w_event=0.0,
        w_bucket=0.0,
        w_top_event=1.0,
        w_top_dir=1.0,
    )
    model_cfg = RACModelConfig(k_seg=2)
    loss_fn = RACLoss(train_cfg, model_cfg)
    out = RACFormerOutput(
        s_pred=torch.zeros((1, 2), dtype=torch.float32),
        bucket_logits=torch.zeros((1, 3, model_cfg.n_buckets), dtype=torch.float32),
        direct_resid_step=torch.zeros((1, 3), dtype=torch.float32),
        event_logits=torch.zeros((1, 3), dtype=torch.float32),
        top_event_logits=torch.zeros((1, 3), dtype=torch.float32),
        top_dir_logits=torch.zeros((1, 3, 3), dtype=torch.float32),
        encoder_out=torch.zeros((1, 3, model_cfg.d_model), dtype=torch.float32),
    )

    losses = loss_fn(out, _minimal_batch(), pred_tvt_hidden=torch.tensor([[10.0]]))

    assert losses["top_event"].item() == pytest.approx(math.log(2.0))
    assert losses["top_dir"].item() == pytest.approx(math.log(3.0))
    assert losses["total"].item() == pytest.approx(math.log(2.0) + math.log(3.0))


def test_racloss_ignores_top_teacher_when_mask_is_empty() -> None:
    train_cfg = RACTrainConfig(
        w_tvt_mse=0.0,
        w_tvt_huber=0.0,
        w_endpoint=0.0,
        w_seg=0.0,
        w_local=0.0,
        w_smooth=0.0,
        w_direct_reg=0.0,
        w_event=0.0,
        w_bucket=0.0,
        w_top_event=1.0,
        w_top_dir=1.0,
    )
    model_cfg = RACModelConfig(k_seg=2)
    batch = _minimal_batch()
    batch["top_teacher_mask"] = torch.zeros_like(batch["top_teacher_mask"])
    out = RACFormerOutput(
        s_pred=torch.zeros((1, 2), dtype=torch.float32),
        bucket_logits=torch.zeros((1, 3, model_cfg.n_buckets), dtype=torch.float32),
        direct_resid_step=torch.zeros((1, 3), dtype=torch.float32),
        event_logits=torch.zeros((1, 3), dtype=torch.float32),
        top_event_logits=torch.zeros((1, 3), dtype=torch.float32),
        top_dir_logits=torch.zeros((1, 3, 3), dtype=torch.float32),
        encoder_out=torch.zeros((1, 3, model_cfg.d_model), dtype=torch.float32),
    )

    losses = RACLoss(train_cfg, model_cfg)(out, batch, pred_tvt_hidden=torch.tensor([[10.0]]))

    assert losses["top_event"].item() == 0.0
    assert losses["top_dir"].item() == 0.0
    assert losses["total"].item() == 0.0


def test_racloss_tvt_only_does_not_require_auxiliary_batch_fields() -> None:
    train_cfg = RACTrainConfig(
        w_tvt_mse=1.0,
        w_tvt_huber=0.0,
        w_endpoint=0.0,
        w_seg=0.0,
        w_local=0.0,
        w_smooth=0.0,
        w_direct_reg=0.0,
        w_event=0.0,
        w_bucket=0.0,
        w_top_event=0.0,
        w_top_dir=0.0,
    )
    batch = {
        "n_hidden_rows": torch.tensor([2], dtype=torch.long),
        "tvt_hidden": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
    }
    pred = torch.tensor([[12.0, 17.0]], dtype=torch.float32)

    losses = RACLoss(train_cfg, RACModelConfig())(SimpleNamespace(), batch, pred)

    assert losses["tvt_mse"].item() == pytest.approx(0.065)
    assert losses["total"].item() == pytest.approx(0.065)
    assert losses["seg"].item() == 0.0
    assert losses["event"].item() == 0.0
