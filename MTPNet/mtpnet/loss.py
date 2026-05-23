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


def vertical_corr_target(target_bins: Tensor, *, height: int, tau_bins: float) -> Tensor:
    bins = torch.arange(height, device=target_bins.device, dtype=target_bins.dtype)
    distance = torch.abs(bins[None, :, None] - target_bins[:, None, :])
    return F.softmax(-distance / max(float(tau_bins), 1e-6), dim=1)


def corr_vertical_kl_loss(
    corr_logits: Tensor, target_bins: Tensor, *, tau_bins: float
) -> Tensor:
    target_prob = vertical_corr_target(
        target_bins.to(device=corr_logits.device, dtype=corr_logits.dtype),
        height=int(corr_logits.shape[1]),
        tau_bins=tau_bins,
    )
    return F.kl_div(
        F.log_softmax(corr_logits, dim=1),
        target_prob.detach(),
        reduction="batchmean",
    )


def mode_corr_scores(
    corr_logits: Tensor, paths: Tensor, *, future_start: int
) -> Tensor:
    future_steps = paths.shape[-1]
    future_logits = corr_logits[:, :, future_start : future_start + future_steps]
    log_prob = F.log_softmax(future_logits, dim=1)
    bins = paths.round().long().clamp(0, corr_logits.shape[1] - 1)
    log_prob_by_step = log_prob.permute(0, 2, 1)
    expanded = log_prob_by_step[:, None, :, :].expand(
        -1, paths.shape[1], -1, -1
    )
    return torch.gather(expanded, dim=3, index=bins[:, :, :, None]).squeeze(3).mean(
        dim=2
    )


def _top3_margin_loss(
    logits: Tensor, best_k: Tensor, *, margin: float = 0.0, top_k: int = 3
) -> Tensor:
    k_modes = logits.shape[1]
    if k_modes <= top_k:
        return logits.new_tensor(0.0)
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    best_logit = logits[batch_index, best_k]
    competitors = logits.clone()
    competitors[batch_index, best_k] = torch.finfo(logits.dtype).min
    kth_competitor = torch.topk(competitors, k=top_k, dim=1).values[:, -1]
    return F.relu(kth_competitor + float(margin) - best_logit).mean()


def _continuation_probability_loss(
    pred: Tensor,
    logits: Tensor,
    history_bins: Tensor | None,
    tau_bins: float,
) -> Tensor:
    if history_bins is None or history_bins.numel() == 0:
        return logits.new_tensor(0.0)
    history = history_bins.to(device=pred.device, dtype=pred.dtype)
    last = history[:, -1]
    if history.shape[1] >= 2:
        prev = history[:, -2]
        slope = torch.where(torch.isfinite(prev), last - prev, torch.zeros_like(last))
    else:
        slope = torch.zeros_like(last)
    expected_first = last + slope
    valid = torch.isfinite(expected_first)
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    continuation_error = torch.abs(
        pred[valid, :, 0] - expected_first[valid, None]
    )
    tau = max(float(tau_bins), 1e-6)
    target_prob = F.softmax(-continuation_error.detach() / tau, dim=1)
    return F.kl_div(
        F.log_softmax(logits[valid], dim=1),
        target_prob,
        reduction="batchmean",
    )


def mtp_loss(
    pred: Tensor,
    logits: Tensor,
    target: Tensor,
    cfg: LossConfig,
    *,
    epoch: int | None = None,
    history_bins: Tensor | None = None,
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
    top3_margin_loss = _top3_margin_loss(
        logits, best_k, margin=cfg.top3_margin, top_k=3
    )
    continuation_loss = _continuation_probability_loss(
        pred, logits, history_bins, cfg.continuation_tau_bins
    )
    alpha_cls = _effective_alpha_cls(cfg, epoch)
    entropy_lambda = _effective_entropy_lambda(cfg, epoch)
    loss = (
        reg_loss
        + alpha_cls * cls_loss
        + cfg.soft_prob_alpha * soft_prob_loss
        + cfg.top3_margin_alpha * top3_margin_loss
        + cfg.continuation_alpha * continuation_loss
        + cfg.smooth_lambda * smooth_loss
        - entropy_lambda * entropy_loss
        + cfg.diversity_lambda * diversity_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "soft_prob_loss": float(soft_prob_loss.detach().cpu()),
        "top3_margin_loss": float(top3_margin_loss.detach().cpu()),
        "continuation_loss": float(continuation_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "entropy_loss": float(entropy_loss.detach().cpu()),
        "diversity_loss": float(diversity_loss.detach().cpu()),
        "alpha_cls_effective": float(alpha_cls),
        "entropy_lambda_effective": float(entropy_lambda),
        "best_k": best_k.detach().cpu(),
        "best_error": errors[batch_index, best_k].detach().cpu(),
    }
    return loss, metrics
