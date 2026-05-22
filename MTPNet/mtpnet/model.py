from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MTPNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        future_steps: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.k_modes = cfg.k_modes
        self.future_steps = future_steps
        self.height = height
        self.bounded_output = cfg.bounded_output
        blocks: list[nn.Module] = []
        c_in = in_channels
        for index, c_out in enumerate(cfg.conv_channels):
            blocks.append(ConvBlock(c_in, c_out))
            blocks.append(ConvBlock(c_out, c_out))
            if index < len(cfg.conv_channels) - 1:
                blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
            c_in = c_out
        self.encoder = nn.Sequential(*blocks)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            flat_dim = int(self.encoder(dummy).reshape(1, -1).shape[1])
        head: list[nn.Module] = []
        in_dim = flat_dim
        for hidden_dim in cfg.hidden_dims:
            head.extend(
                [
                    nn.Linear(in_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = hidden_dim
        self.head = nn.Sequential(*head)
        self.path_head = nn.Linear(in_dim, cfg.k_modes * future_steps)
        self.logit_head = nn.Linear(in_dim, cfg.k_modes)
        if cfg.mode_bias_init:
            self._init_mode_bias(cfg.mode_bias_span_bins)

    def _init_mode_bias(self, span_bins: float) -> None:
        max_bin = float(self.height - 1)
        center = max_bin / 2.0
        span = min(float(span_bins), center - 1e-3)
        centers = torch.linspace(
            center - span,
            center + span,
            self.k_modes,
            dtype=self.path_head.bias.dtype,
            device=self.path_head.bias.device,
        )
        if self.bounded_output:
            norm = (centers / max_bin).clamp(0.02, 0.98)
            centers = torch.logit(norm)
        bias = centers[:, None].repeat(1, self.future_steps).reshape(-1)
        with torch.no_grad():
            self.path_head.bias.copy_(bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        features = self.encoder(x).reshape(batch, -1)
        hidden = self.head(features)
        raw_paths = self.path_head(hidden).reshape(
            batch, self.k_modes, self.future_steps
        )
        if self.bounded_output:
            paths = float(self.height - 1) * torch.sigmoid(raw_paths)
        else:
            paths = raw_paths
        logits = self.logit_head(hidden)
        return paths, logits
