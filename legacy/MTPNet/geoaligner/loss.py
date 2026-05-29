from __future__ import annotations

import torch
import torch.nn.functional as F


def alignment_ce_loss(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    hidden_mask: torch.Tensor,
    lateral_pad_mask: torch.Tensor,
    typewell_pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    active = hidden_mask & ~lateral_pad_mask & (target_bins >= 0)
    if active.sum() == 0:
        return logits.sum() * 0.0
    masked_logits = logits
    if typewell_pad_mask is not None:
        masked_logits = masked_logits.masked_fill(typewell_pad_mask[:, None, :], -1e9)
    return F.cross_entropy(masked_logits[active], target_bins[active])

