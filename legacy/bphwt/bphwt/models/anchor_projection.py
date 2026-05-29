"""Anchor projection: enforce TVT_input known values in model output.

anchor_project(tvt_raw, tvt_input, known_mask, strength=1.0)

With strength=1.0 (hard): replaces hidden prediction with known value at
known rows. With 0 < strength < 1: soft blend.

Also exposes integrate_velocity_with_anchors for the velocity integration path.
"""

from __future__ import annotations

import torch


def anchor_project(
    tvt_raw: torch.Tensor,  # [B, L]
    tvt_input: torch.Tensor,  # [B, L] — NaN where hidden
    known_mask: torch.Tensor,  # [B, L] bool or float (1=known, 0=hidden)
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Replace model output at known rows with TVT_input value.

    tvt_input must already be NaN-filled (pass last-known-filled version).
    """
    km = known_mask.float()
    # Where mask == 1: use tvt_input; where mask == 0: use tvt_raw
    # Handle NaN in tvt_input: replace with tvt_raw there
    tvt_anchor = torch.where(torch.isnan(tvt_input), tvt_raw, tvt_input)
    if strength >= 1.0:
        return km * tvt_anchor + (1.0 - km) * tvt_raw
    else:
        return km * (strength * tvt_anchor + (1.0 - strength) * tvt_raw) + (1.0 - km) * tvt_raw


def integrate_velocity_with_anchors(
    velocity: torch.Tensor,  # [B, L] dTVT/dMD
    md: torch.Tensor,  # [B, L] measured depth
    tvt_input: torch.Tensor,  # [B, L] NaN where hidden
    known_mask: torch.Tensor,  # [B, L]
) -> torch.Tensor:
    """
    Integrate velocity to get TVT, resetting at known anchor points.

    Integration: TVT[t] = TVT[anchor] + sum_{t'=anchor}^{t} velocity[t'] * dMD[t']
    """
    B, L = velocity.shape
    dmd = torch.diff(md, dim=1, prepend=md[:, :1])  # [B, L]
    dmd = torch.clamp(dmd.abs(), min=0.1)

    integrated = torch.zeros_like(velocity)
    for b in range(B):
        km = known_mask[b]  # [L]
        tv = tvt_input[b]  # [L]
        ve = velocity[b]  # [L]
        dm = dmd[b]  # [L]

        # Find known anchor indices
        anchor_idx = torch.where(km > 0.5)[0]
        if len(anchor_idx) == 0:
            integrated[b] = ve.cumsum(0) * dm
            continue

        out = torch.zeros(L, device=velocity.device)

        # Process segments between anchors
        anchor_list = anchor_idx.tolist()
        # Extend to cover full range
        if anchor_list[0] > 0:
            anchor_list = [0] + anchor_list
        if anchor_list[-1] < L - 1:
            anchor_list = anchor_list + [L - 1]

        for k in range(len(anchor_list) - 1):
            a_start = anchor_list[k]
            a_end = anchor_list[k + 1]
            if km[a_start] > 0.5:
                start_tvt = tv[a_start]
            else:
                start_tvt = out[a_start]
            # Integrate forward
            seg_delta = (ve[a_start : a_end + 1] * dm[a_start : a_end + 1]).cumsum(0)
            out[a_start : a_end + 1] = start_tvt + seg_delta
            # Override anchor endpoints
            if km[a_end] > 0.5:
                out[a_end] = tv[a_end]

        integrated[b] = out

    return integrated
