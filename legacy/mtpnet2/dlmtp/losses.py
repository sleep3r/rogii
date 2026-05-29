"""
dlmtp/losses.py  —  Loss functions for heatmap TVT prediction.

gaussian_ce_loss
    Soft cross-entropy with a Gaussian target centred on the true bin.
    Loss = -sum_j [ p_gauss(j) * log_softmax(logits)[j] ]
    Averaged over valid rows only.

path_smooth_loss
    Penalise rapid bin changes along the L axis (encourages a smooth path).
    Applied to the predicted argmax path from logits.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_ce_loss(
    logits: torch.Tensor,       # (B, L, J)
    true_bins: torch.Tensor,    # (B, L)  int64
    valid_mask: torch.Tensor,   # (B, L)  float32, 1 = valid row
    sigma: float = 3.0,
) -> torch.Tensor:
    """Soft cross-entropy with Gaussian target N(true_bin, sigma^2).

    Returns scalar loss (mean over valid rows across batch).
    """
    B, L, J = logits.shape
    device   = logits.device

    j_idx    = torch.arange(J, dtype=logits.dtype, device=device)  # (J,)
    # true_bins: (B, L) → (B, L, 1) for broadcasting
    centers  = true_bins.float().unsqueeze(-1)                      # (B, L, 1)
    gauss    = torch.exp(-0.5 * ((j_idx - centers) / sigma) ** 2)  # (B, L, J)
    gauss    = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-9)    # normalise

    log_p    = F.log_softmax(logits, dim=-1)                        # (B, L, J)
    ce_per   = -(gauss * log_p).sum(dim=-1)                         # (B, L)

    # Mask and average
    n_valid  = valid_mask.sum().clamp(min=1.0)
    loss     = (ce_per * valid_mask).sum() / n_valid
    return loss


def path_smooth_loss(
    logits: torch.Tensor,       # (B, L, J)
    valid_mask: torch.Tensor,   # (B, L)
    weight: float = 0.05,
) -> torch.Tensor:
    """Penalise expected bin variance along the L axis (smooth path prior).

    Uses soft expected bin E[j | logits] and penalises |E[j_s] - E[j_{s-1}]|^2.
    """
    if weight == 0.0:
        return logits.new_zeros(())

    B, L, J = logits.shape
    device  = logits.device

    j_idx   = torch.arange(J, dtype=logits.dtype, device=device)
    probs   = F.softmax(logits, dim=-1)                  # (B, L, J)
    e_bin   = (probs * j_idx).sum(dim=-1)                # (B, L) expected bin

    # Consecutive differences along L
    diff    = (e_bin[:, 1:] - e_bin[:, :-1]) ** 2       # (B, L-1)
    # Valid only when both consecutive rows are valid
    mask    = valid_mask[:, 1:] * valid_mask[:, :-1]     # (B, L-1)
    n_valid = mask.sum().clamp(min=1.0)
    return weight * (diff * mask).sum() / n_valid
