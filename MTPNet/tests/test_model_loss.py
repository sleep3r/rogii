import torch

from mtpnet.config import LossConfig, ModelConfig
from mtpnet.loss import mtp_loss
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
