"""
dlmtp/unet.py  —  Lightweight 2D U-Net for heatmap → bin logits.

Input  : (B, C, L, J)   where C = N_CHANNELS, L = crop length, J = TVT bins
Output : (B, L, J)       logits over TVT bins per row

Architecture (depth=3, base=32):
    Encoder:  DoubleConv(C→32) → Pool → DoubleConv(32→64) → Pool → DoubleConv(64→128)
    Bottleneck:  → Pool → DoubleConv(128→256)
    Decoder:  Up+skip→DoubleConv(384→128) → Up+skip→DoubleConv(192→64) → Up+skip→DoubleConv(96→32)
    Head:     Conv2d(32→1)

~7 M params for base=32, depth=3.
~1.7 M params for base=16, depth=3 (use for CPU-only runs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x  = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class HeatmapUNet(nn.Module):
    """2D U-Net: (B, C, L, J) → (B, L, J) logits.

    Parameters
    ----------
    in_ch : number of input heatmap channels (default 8)
    base  : base channel count (doubled at each level)
    depth : number of down/up-sampling steps
    """

    def __init__(self, in_ch: int = 8, base: int = 32, depth: int = 3):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]   # e.g. [32,64,128,256]

        self.inc    = DoubleConv(in_ch, chs[0])
        self.downs  = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth)])
        # Up layers: skip_ch = chs[depth-1-i], in_ch = chs[depth-i]
        self.ups    = nn.ModuleList([
            Up(chs[depth - i], chs[depth - i - 1], chs[depth - i - 1])
            for i in range(depth)
        ])
        self.head   = nn.Conv2d(chs[0], 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, L, J)  — heatmap tensor (L and J must be divisible by 2^depth)
        returns (B, L, J) logits
        """
        skips = []
        x = self.inc(x)
        skips.append(x)                    # skip 0

        for down in self.downs:
            x = down(x)
            skips.append(x)               # skip 1..depth

        # skips[-1] is the bottleneck — no skip needed for first up
        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]        # going backwards: depth-1 .. 0
            x = up(x, skip)

        return self.head(x).squeeze(1)    # (B, L, J)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
