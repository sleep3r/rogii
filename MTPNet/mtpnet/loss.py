from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig


def _path_error(pred: Tensor, target: Tensor, path_loss: str) -> Tensor:
    target_expanded = target[:, None, :]
    if path_loss == "mae":
        return torch.abs(pred - target_expanded).mean(dim=-1)
    if path_loss == "mse":
        return ((pred - target_expanded) ** 2).mean(dim=-1)
    raise ValueError(f"Unsupported path_loss: {path_loss}")


def _smoothness(paths: Tensor) -> Tensor:
    if paths.shape[-1] < 3:
        return paths.new_tensor(0.0)
    second = paths[:, 2:] - 2.0 * paths[:, 1:-1] + paths[:, :-2]
    return torch.abs(second).mean()


def mtp_loss(
    pred: Tensor, logits: Tensor, target: Tensor, cfg: LossConfig
) -> tuple[Tensor, dict[str, Any]]:
    errors = _path_error(pred, target, cfg.path_loss)
    best_k = errors.argmin(dim=1)
    batch_index = torch.arange(pred.shape[0], device=pred.device)
    best_paths = pred[batch_index, best_k]
    reg_loss = _path_error(best_paths[:, None, :], target, cfg.path_loss).mean()
    cls_loss = F.cross_entropy(logits, best_k)
    smooth_loss = _smoothness(best_paths)
    loss = reg_loss + cfg.alpha_cls * cls_loss + cfg.smooth_lambda * smooth_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "best_k": best_k.detach().cpu(),
        "best_error": errors[batch_index, best_k].detach().cpu(),
    }
    return loss, metrics
