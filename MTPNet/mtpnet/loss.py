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


def _effective_alpha_cls(cfg: LossConfig, epoch: int | None) -> float:
    if epoch is not None and cfg.cls_warmup_epochs > 0 and epoch <= cfg.cls_warmup_epochs:
        return cfg.alpha_cls_warmup_value
    return cfg.alpha_cls


def _effective_entropy_lambda(cfg: LossConfig, epoch: int | None) -> float:
    if cfg.entropy_warmup_epochs > 0:
        if epoch is None or epoch <= cfg.entropy_warmup_epochs:
            return cfg.entropy_lambda
        return cfg.entropy_final_lambda
    return cfg.entropy_lambda


def _entropy(logits: Tensor) -> Tensor:
    prob = F.softmax(logits, dim=1)
    return -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()


def _diversity_margin(paths: Tensor, margin_bins: float) -> Tensor:
    if paths.shape[1] < 2 or margin_bins <= 0.0:
        return paths.new_tensor(0.0)
    pair_dist = torch.abs(paths[:, :, None, :] - paths[:, None, :, :]).mean(dim=-1)
    k_modes = paths.shape[1]
    pair_mask = torch.triu(
        torch.ones(k_modes, k_modes, dtype=torch.bool, device=paths.device),
        diagonal=1,
    )
    pair_dist = pair_dist[:, pair_mask]
    return F.relu(float(margin_bins) - pair_dist).mean()


def _soft_probability_loss(errors: Tensor, logits: Tensor, tau_bins: float) -> Tensor:
    tau = max(float(tau_bins), 1e-6)
    target_prob = F.softmax(-errors.detach() / tau, dim=1)
    return F.kl_div(F.log_softmax(logits, dim=1), target_prob, reduction="batchmean")


def mtp_loss(
    pred: Tensor,
    logits: Tensor,
    target: Tensor,
    cfg: LossConfig,
    *,
    epoch: int | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    errors = _path_error(pred, target, cfg.path_loss)
    best_k = errors.argmin(dim=1)
    batch_index = torch.arange(pred.shape[0], device=pred.device)
    best_paths = pred[batch_index, best_k]
    reg_loss = _path_error(best_paths[:, None, :], target, cfg.path_loss).mean()
    cls_loss = F.cross_entropy(logits, best_k)
    soft_prob_loss = _soft_probability_loss(
        errors, logits, cfg.soft_prob_tau_bins
    )
    smooth_loss = _smoothness(best_paths)
    entropy_loss = _entropy(logits)
    diversity_loss = _diversity_margin(pred, cfg.diversity_margin_bins)
    alpha_cls = _effective_alpha_cls(cfg, epoch)
    entropy_lambda = _effective_entropy_lambda(cfg, epoch)
    loss = (
        reg_loss
        + alpha_cls * cls_loss
        + cfg.soft_prob_alpha * soft_prob_loss
        + cfg.smooth_lambda * smooth_loss
        - entropy_lambda * entropy_loss
        + cfg.diversity_lambda * diversity_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "soft_prob_loss": float(soft_prob_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "entropy_loss": float(entropy_loss.detach().cpu()),
        "diversity_loss": float(diversity_loss.detach().cpu()),
        "alpha_cls_effective": float(alpha_cls),
        "entropy_lambda_effective": float(entropy_lambda),
        "best_k": best_k.detach().cpu(),
        "best_error": errors[batch_index, best_k].detach().cpu(),
    }
    return loss, metrics
