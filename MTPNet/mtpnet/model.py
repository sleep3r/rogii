from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
import torch
from torch import Tensor, nn

from .config import CorrelationHeadConfig, ModelConfig


@dataclass(frozen=True)
class MTPForwardOutput:
    paths: Tensor
    logits: Tensor
    raw_paths: Tensor
    corr_logits: Tensor | None = None


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
        corr_head: CorrelationHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.k_modes = cfg.k_modes
        self.future_steps = future_steps
        self.height = height
        self.width = width
        self.bounded_output = cfg.bounded_output
        self.corr_head_enabled = bool(corr_head and corr_head.enabled)
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
            encoded = self.encoder(dummy)
            flat_dim = int(encoded.reshape(1, -1).shape[1])
            encoded_channels = int(encoded.shape[1])
        self.corr_head = (
            nn.Conv2d(encoded_channels, 1, kernel_size=1)
            if self.corr_head_enabled
            else None
        )
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

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward_raw(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        features = self.encode(x).reshape(batch, -1)
        hidden = self.head(features)
        raw_paths = self.path_head(hidden).reshape(
            batch, self.k_modes, self.future_steps
        )
        logits = self.logit_head(hidden)
        return raw_paths, logits

    def forward_all(self, x: Tensor) -> MTPForwardOutput:
        batch = x.shape[0]
        encoded = self.encode(x)
        features = encoded.reshape(batch, -1)
        hidden = self.head(features)
        raw_paths = self.path_head(hidden).reshape(
            batch, self.k_modes, self.future_steps
        )
        logits = self.logit_head(hidden)
        paths = self.bound_paths(raw_paths)
        corr_logits: Tensor | None = None
        if self.corr_head is not None:
            corr_logits = self.corr_head(encoded)
            corr_logits = F.interpolate(
                corr_logits,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        return MTPForwardOutput(
            paths=paths,
            logits=logits,
            raw_paths=raw_paths,
            corr_logits=corr_logits,
        )

    def bound_paths(self, raw_paths: Tensor) -> Tensor:
        if self.bounded_output:
            return float(self.height - 1) * torch.sigmoid(raw_paths)
        return raw_paths

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = self.forward_all(x)
        return output.paths, output.logits
