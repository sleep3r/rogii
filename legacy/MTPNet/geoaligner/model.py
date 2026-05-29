from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import GAModelConfig
from .dataset import N_LATERAL_FEATURES, N_TYPEWELL_FEATURES


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def _encoder(cfg: GAModelConfig, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.ffn_dim,
        dropout=cfg.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(cfg.d_model))


class GeoAligner(nn.Module):
    def __init__(self, cfg: GAModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lateral_stem = nn.Sequential(
            nn.Linear(N_LATERAL_FEATURES, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.typewell_stem = nn.Sequential(
            nn.Linear(N_TYPEWELL_FEATURES, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.lateral_pe = SinusoidalPE(cfg.d_model, dropout=cfg.dropout)
        self.typewell_pe = SinusoidalPE(cfg.d_model, dropout=cfg.dropout)
        self.lateral_encoder = _encoder(cfg, cfg.lateral_layers)
        self.typewell_encoder = _encoder(cfg, cfg.typewell_layers)
        self.lateral_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.typewell_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.scale = math.sqrt(float(cfg.d_model))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        lateral_features: torch.Tensor,
        typewell_features: torch.Tensor,
        lateral_pad_mask: torch.Tensor | None = None,
        typewell_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lat = self.lateral_pe(self.lateral_stem(lateral_features))
        tw = self.typewell_pe(self.typewell_stem(typewell_features))
        lat = self.lateral_encoder(lat, src_key_padding_mask=lateral_pad_mask)
        tw = self.typewell_encoder(tw, src_key_padding_mask=typewell_pad_mask)
        lat = torch.nn.functional.normalize(self.lateral_proj(lat), dim=-1)
        tw = torch.nn.functional.normalize(self.typewell_proj(tw), dim=-1)
        logits = torch.matmul(lat, tw.transpose(1, 2)) * self.scale
        if typewell_pad_mask is not None:
            logits = logits.masked_fill(typewell_pad_mask[:, None, :], -1e9)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

