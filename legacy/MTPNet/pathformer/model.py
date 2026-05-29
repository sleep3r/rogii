"""PathFormer model: full-well TVT prediction via transformer encoder.

Architecture:
  1. Feature stem: Linear(N_FEATURES, d_model) + LayerNorm
  2. Positional encoding: sinusoidal (fixed)
  3. Transformer encoder: n_encoder_layers × (self-attn + FFN), bidirectional
  4. TVT head: Linear(d_model, 1) → squeeze → predicted delta per step

Output: pred_delta of shape (B, seq_len) where
  pred_tvt = last_known_tvt + pred_delta
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import PFModelConfig, N_FEATURES


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# PathFormer
# ---------------------------------------------------------------------------

class PathFormer(nn.Module):
    """Full-well TVT path predictor.

    Input:
        features: (B, T, N_FEATURES) — compressed-step features
        pad_mask:  (B, T)            — True = padding (ignored by attention)

    Output:
        pred_delta: (B, T)  — predicted TVT - last_known_tvt for each step
    """

    def __init__(self, cfg: PFModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.anchor_skip = cfg.anchor_skip
        self.anchor_feature_index = cfg.anchor_feature_index
        self.anchor_valid_index = cfg.anchor_valid_index
        self.anchor_scale = cfg.anchor_scale

        # feature stem
        self.stem = nn.Sequential(
            nn.Linear(N_FEATURES, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # positional encoding
        self.pe = SinusoidalPE(cfg.d_model, max_len=1024, dropout=cfg.dropout)

        # transformer encoder (bidirectional — no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_encoder_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        # prediction head
        self.tvt_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features: (B, T, N_FEATURES)
        pad_mask:  (B, T) bool — True at padding positions
        returns:   (B, T)
        """
        x = self.stem(features)          # (B, T, d_model)
        x = self.pe(x)                   # (B, T, d_model) + sinusoidal PE

        # TransformerEncoder expects src_key_padding_mask shape (B, T)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, T, d_model)

        pred_delta = self.tvt_head(x).squeeze(-1)  # (B, T)
        if self.anchor_skip:
            if self.anchor_feature_index >= features.size(-1) or self.anchor_valid_index >= features.size(-1):
                raise ValueError(
                    "anchor_skip feature indices exceed feature dimension "
                    f"{features.size(-1)}"
                )
            anchor_delta = features[..., self.anchor_feature_index] * self.anchor_scale
            anchor_valid = features[..., self.anchor_valid_index] > 0.5
            pred_delta = torch.where(anchor_valid, anchor_delta + pred_delta, pred_delta)
        return pred_delta


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pathformer_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    hidden_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    huber_delta: float = 5.0,
    smooth_lambda: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Huber loss on hidden steps + smoothness penalty.

    pred_delta, target_delta: (B, T)
    hidden_mask: (B, T) bool — True = hidden step
    pad_mask:    (B, T) bool — True = padding
    Returns (loss, info_dict).
    """
    # mask = hidden AND not padding
    active = hidden_mask & ~pad_mask   # (B, T)

    if active.sum() == 0:
        zero = pred_delta.sum() * 0.0
        return zero, {"huber": 0.0, "smooth": 0.0, "total": 0.0}

    diff = pred_delta[active] - target_delta[active]
    huber = torch.where(
        diff.abs() < huber_delta,
        0.5 * diff ** 2,
        huber_delta * (diff.abs() - 0.5 * huber_delta),
    ).mean()

    # smoothness: second difference on hidden portion per sample
    smooth = torch.tensor(0.0, device=pred_delta.device)
    if smooth_lambda > 0:
        B = pred_delta.size(0)
        smooth_terms = 0
        for b in range(B):
            h_idx = (hidden_mask[b] & ~pad_mask[b]).nonzero(as_tuple=True)[0]
            if len(h_idx) >= 3:
                h_pred = pred_delta[b][h_idx]
                second_diff = h_pred[2:] - 2 * h_pred[1:-1] + h_pred[:-2]
                smooth = smooth + (second_diff ** 2).mean()
                smooth_terms += 1
        if smooth_terms > 0:
            smooth = smooth / smooth_terms

    total = huber + smooth_lambda * smooth
    return total, {
        "huber": float(huber.item()),
        "smooth": float(smooth.item()),
        "total": float(total.item()),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
