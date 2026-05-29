"""BPHWT — Bayesian Physics-informed Horizon-Warp Transformer.

Architecture: 1D ConvNeXt U-Net with optional bottleneck attention.
Input:  [B, C_in, L]  — feature channels over lateral sequence
Output: dict with multiple heads at full resolution [B, L]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt-style 1D block: DWConv → LayerNorm → MLP → scale."""

    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, channels * expansion)
        self.pw2 = nn.Linear(channels * expansion, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        residual = x
        h = self.dw(x)
        h = h.permute(0, 2, 1)  # [B, L, C]
        h = self.norm(h)
        h = self.act(self.pw1(h))
        h = self.drop(h)
        h = self.pw2(h)
        h = h * self.gamma
        h = h.permute(0, 2, 1)  # [B, C, L]
        return residual + h


class EncoderStage(nn.Module):
    """Encoder stage: optional stride-2 downsampling + N ConvNeXt blocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int = 2,
        stride: int = 2,
        dropout: float = 0.0,
        stoch_depth: float = 0.0,
    ) -> None:
        super().__init__()
        # Downsampling
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.LayerNorm(in_ch),
                nn.Conv1d(in_ch, out_ch, kernel_size=stride, stride=stride),
            )
        else:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock1D(out_ch, dropout=dropout * (1 - stoch_depth * i / max(n_blocks - 1, 1)))
                for i in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.downsample, "__len__"):
            # Sequential with LayerNorm
            x_perm = x.permute(0, 2, 1)
            x_perm = self.downsample[0](x_perm)
            x = x_perm.permute(0, 2, 1)
            x = self.downsample[1](x)
        else:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class LocalSelfAttention1D(nn.Module):
    """Efficient local self-attention with a sliding window."""

    def __init__(self, d_model: int, n_heads: int = 8, window: int = 256) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] → attend → [B, C, L]
        B, C, L = x.shape
        h = x.permute(0, 2, 1)  # [B, L, C]
        residual = h
        h = self.norm(h)

        QKV = self.qkv(h)  # [B, L, 3C]
        Q, K, V = QKV.chunk(3, dim=-1)

        def _mh(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = _mh(Q), _mh(K), _mh(V)  # [B, H, L, hd]

        # Full attention (for bottleneck, L is small after 3× stride-2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B, H, L, hd]
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out(out)
        out = out + residual
        return out.permute(0, 2, 1)  # [B, C, L]


class DecoderStage(nn.Module):
    """Decoder stage: upsample + merge skip + N ConvNeXt blocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        n_blocks: int = 2,
        scale: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="linear", align_corners=False),
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
        )
        self.skip_proj = nn.Conv1d(skip_ch, out_ch, kernel_size=1) if skip_ch != out_ch else nn.Identity()
        self.merge_norm = nn.Sequential(
            nn.LayerNorm(out_ch),
        )
        self.blocks = nn.ModuleList([ConvNeXtBlock1D(out_ch, dropout=dropout) for _ in range(n_blocks)])
        self._out_ch = out_ch

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, L/2], skip: [B, skip_ch, L]
        x = self.upsample(x)  # [B, out_ch, L]
        # Align lengths (skip might be slightly longer due to odd lengths)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        s = self.skip_proj(skip)  # [B, out_ch, L]
        h = x + s
        h = h.permute(0, 2, 1)
        h = self.merge_norm[0](h)
        h = h.permute(0, 2, 1)
        for blk in self.blocks:
            h = blk(h)
        return h


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class BPHWT(nn.Module):
    """
    Bayesian Physics-informed Horizon-Warp Transformer (BPHWT-lite).

    Input:  [B, C_in, L]
    Output: dict
        tvt_pred        [B, L]   — final TVT prediction (anchor-projected)
        tvt_raw         [B, L]   — raw TVT before anchor projection
        mu_delta        [B, L]   — predicted residual to TVT_base
        log_sigma       [B, L]   — log std of TVT prediction
        dip_sign        [B, 3, L]— dip sign logits (down/flat/up)
        velocity        [B, L]   — predicted dTVT/dMD
        seg_boundary_logits [B,L]— segment boundary logits
        seg_boundary    [B, L]   — segment boundary probability
        tvt_integrated  [B, L]   — velocity-integrated TVT (aux)
    """

    def __init__(
        self,
        in_channels: int,
        stage_channels: list[int] = (96, 160, 256, 384),
        stage_strides: list[int] = (2, 2, 2, 2),
        decoder_channels: list[int] = (256, 160, 96, 64),
        n_blocks: int = 2,
        use_bottleneck_attn: bool = True,
        attn_heads: int = 8,
        dropout: float = 0.05,
        stoch_depth: float = 0.05,
        predict_velocity: bool = True,
        predict_dip_sign: bool = True,
        predict_seg_boundary: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.predict_velocity = predict_velocity
        self.predict_dip_sign = predict_dip_sign
        self.predict_seg_boundary = predict_seg_boundary

        # Stem
        stem_ch = stage_channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(stem_ch, stem_ch, kernel_size=3, padding=1),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = stem_ch
        self.enc_channels = [stem_ch]
        for i, (ch, stride) in enumerate(zip(stage_channels, stage_strides)):
            enc = EncoderStage(
                prev_ch, ch, n_blocks=n_blocks, stride=stride, dropout=dropout, stoch_depth=stoch_depth
            )
            self.encoders.append(enc)
            self.enc_channels.append(ch)
            prev_ch = ch

        # Bottleneck attention
        self.bottleneck_attn = (
            LocalSelfAttention1D(prev_ch, n_heads=attn_heads) if use_bottleneck_attn else nn.Identity()
        )

        # Decoder
        self.decoders = nn.ModuleList()
        dec_chs = list(decoder_channels)
        enc_skip_chs = list(reversed(self.enc_channels[:-1]))  # skip from encoder stages
        enc_strides = list(reversed(stage_strides))

        for i, (dec_ch, skip_ch, stride) in enumerate(zip(dec_chs, enc_skip_chs, enc_strides)):
            dec = DecoderStage(prev_ch, skip_ch, dec_ch, n_blocks=n_blocks, scale=stride, dropout=dropout)
            self.decoders.append(dec)
            prev_ch = dec_ch

        final_ch = prev_ch

        # Output heads
        self.head_mu = nn.Sequential(
            nn.Conv1d(final_ch, final_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(final_ch // 2, 1, kernel_size=1),
        )
        self.head_sigma = nn.Sequential(
            nn.Conv1d(final_ch, final_ch // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(final_ch // 4, 1, kernel_size=1),
        )
        if predict_velocity:
            self.head_velocity = nn.Sequential(
                nn.Conv1d(final_ch, final_ch // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(final_ch // 4, 1, kernel_size=1),
                nn.Tanh(),  # bound to [-1, 1], scale by max_velocity downstream
            )
        if predict_dip_sign:
            self.head_dip_sign = nn.Sequential(
                nn.Conv1d(final_ch, final_ch // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(final_ch // 4, 3, kernel_size=1),  # {down, flat, up}
            )
        if predict_seg_boundary:
            self.head_seg = nn.Sequential(
                nn.Conv1d(final_ch, final_ch // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(final_ch // 4, 1, kernel_size=1),
            )

        self._init_recipe_heads()

    def forward(
        self,
        x: torch.Tensor,  # [B, C_in, L]
        tvt_base: torch.Tensor,  # [B, L]  — TVT prior (linear/HMM blend)
        tvt_input: torch.Tensor,  # [B, L]  — NaN-filled TVT_input
        known_mask: torch.Tensor,  # [B, L]  — 1 = known, 0 = hidden
        md: torch.Tensor,  # [B, L]  — measured depth
        tw_tvt: torch.Tensor | None = None,  # for typewell cross-attention (future)
    ) -> dict[str, torch.Tensor]:

        # ---------- Encoder ----------
        h = self.stem(x)  # [B, stem_ch, L]
        skips = [h]
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # ---------- Bottleneck ----------
        h = self.bottleneck_attn(h)

        # ---------- Decoder ----------
        skips_dec = list(reversed(skips[:-1]))
        for dec, skip in zip(self.decoders, skips_dec):
            h = dec(h, skip)

        # ---------- Heads ----------
        mu_delta = self.head_mu(h).squeeze(1)  # [B, L]
        log_sigma = self.head_sigma(h).squeeze(1)  # [B, L]
        log_sigma = torch.clamp(log_sigma, -4.0, 4.0)

        # TVT prediction = prior + residual
        tvt_raw = tvt_base + mu_delta
        # Anchor projection: replace known rows with TVT_input
        from bphwt.models.anchor_projection import anchor_project

        tvt_pred = anchor_project(tvt_raw, tvt_input, known_mask, strength=1.0)

        out: dict[str, torch.Tensor] = {
            "tvt_pred": tvt_pred,
            "tvt_raw": tvt_raw,
            "mu_delta": mu_delta,
            "log_sigma": log_sigma,
        }

        if self.predict_velocity:
            vel = self.head_velocity(h).squeeze(1) * 0.15  # [B, L], ±0.15 ft/ft
            out["velocity"] = vel
            # Velocity-integrated TVT (auxiliary)
            from bphwt.models.anchor_projection import integrate_velocity_with_anchors

            try:
                tvt_int = integrate_velocity_with_anchors(vel, md, tvt_input, known_mask)
                out["tvt_integrated"] = tvt_int
            except Exception:
                out["tvt_integrated"] = tvt_raw

        if self.predict_dip_sign:
            out["dip_sign"] = self.head_dip_sign(h)  # [B, 3, L]

        if self.predict_seg_boundary:
            seg_logits = self.head_seg(h).squeeze(1)  # [B, L]
            out["seg_boundary_logits"] = seg_logits
            out["seg_boundary"] = torch.sigmoid(seg_logits)  # [B, L]

        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_recipe_heads(self) -> None:
        """Start as the explicit TVT_base baseline, then learn residuals."""
        _zero_last_conv(self.head_mu)
        _zero_last_conv(self.head_sigma)
        if self.predict_velocity:
            _zero_last_conv(self.head_velocity)
        if self.predict_dip_sign:
            _zero_last_conv(self.head_dip_sign)
        if self.predict_seg_boundary:
            _zero_last_conv(self.head_seg)


def _zero_last_conv(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv1d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return
