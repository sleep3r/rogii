"""Typewell cross-attention module.

Encodes the typewell sequence and lets horizontal rows attend to it
using relative-position-biased cross-attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypewellEncoder(nn.Module):
    """Small 1D Conv encoder for the typewell sequence."""

    def __init__(self, in_channels: int = 3, d_model: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        layers = []
        for _ in range(n_layers):
            layers.append(_ConvNeXtBlock1D(d_model))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_tw, M]  →  [B, M, d_model]"""
        h = self.stem(x)  # [B, d_model, M]
        h = self.layers(h)  # [B, d_model, M]
        h = h.permute(0, 2, 1)  # [B, M, d_model]
        return self.norm(h)


class TypewellCrossAttention(nn.Module):
    """
    Each horizontal position attends to a local window in the typewell.
    Query: horizontal token (flattened from encoder feature)
    Key/Value: typewell token at matching TVT neighbourhood

    Relative position bias: uses |TVT_horizontal - TVT_typewell| normalised.
    """

    def __init__(
        self,
        d_horiz: int,
        d_tw: int,
        n_heads: int = 4,
        attn_window: int = 32,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_horiz // n_heads
        self.d_horiz = d_horiz
        self.attn_window = attn_window

        self.q_proj = nn.Linear(d_horiz, d_horiz, bias=False)
        self.k_proj = nn.Linear(d_tw, d_horiz, bias=False)
        self.v_proj = nn.Linear(d_tw, d_horiz, bias=False)
        self.out_proj = nn.Linear(d_horiz, d_horiz)
        self.scale = math.sqrt(self.head_dim)

        # Relative position bias (bucket-based)
        self.rel_bias = nn.Embedding(64, n_heads)

    def forward(
        self,
        horiz: torch.Tensor,  # [B, L, d_horiz]
        typewell: torch.Tensor,  # [B, M, d_tw]
        tvt_horiz: torch.Tensor,  # [B, L] predicted TVT (used for window centering)
        tw_tvt: torch.Tensor,  # [B, M] typewell TVT grid
    ) -> torch.Tensor:
        B, L, _ = horiz.shape
        _, M, _ = typewell.shape

        Q = self.q_proj(horiz)  # [B, L, d_horiz]
        K = self.k_proj(typewell)  # [B, M, d_horiz]
        V = self.v_proj(typewell)  # [B, M, d_horiz]

        # Reshape to multi-head
        def _split(t: torch.Tensor) -> torch.Tensor:
            bsz, seq, dim = t.shape
            return t.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

        Q = _split(Q)  # [B, H, L, head_dim]
        K = _split(K)  # [B, H, M, head_dim]
        V = _split(V)  # [B, H, M, head_dim]

        # Compute attention scores [B, H, L, M]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Relative position bias based on |TVT_h - TVT_tw|
        tvt_diff = (tvt_horiz.unsqueeze(-1) - tw_tvt.unsqueeze(-2)).abs()  # [B, L, M]
        # Bucket into 64 bins (0..200 ft range → 64 bins)
        buckets = torch.clamp((tvt_diff / 3.0).long(), 0, 63)  # [B, L, M]
        rel_b = self.rel_bias(buckets)  # [B, L, M, H]
        rel_b = rel_b.permute(0, 3, 1, 2)  # [B, H, L, M]
        scores = scores + rel_b

        # Window masking: only attend to typewell within ±window ft
        window_mask = tvt_diff > self.attn_window  # [B, L, M] True = mask out
        scores = scores.masked_fill(window_mask.unsqueeze(1), -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B, H, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_horiz)
        return self.out_proj(out)


class _ConvNeXtBlock1D(nn.Module):
    def __init__(self, d: int, expansion: int = 4) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, d * expansion)
        self.pw2 = nn.Linear(d * expansion, d)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(d) * 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        residual = x
        h = self.dw(x)  # [B, C, L]
        h = h.permute(0, 2, 1)  # [B, L, C]
        h = self.norm(h)
        h = self.pw2(self.act(self.pw1(h)))
        h = h * self.gamma
        h = h.permute(0, 2, 1)  # [B, C, L]
        return residual + h
