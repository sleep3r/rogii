"""RAC-Former v1: Residual Anchored C-field Transformer for TVT prediction.

Architecture:
  InputProjection  →  ConvStem (4 dilated depth-sep conv blocks)
                   →  TransformerEncoder (ENC_LAYERS pre-norm layers)
                   →  SegmentQueryDecoder (K_SEG queries, DEC_LAYERS)
                   →  5 output heads (all zero-initialized)
                   →  Row materializer

Zero-init guarantee: all five head Linear projections start at zero, so
  initial s_pred = 0, direct_resid = 0, and pred_tvt = prior_tvt exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    K_SEG,
    N_FEATURES,
    RACModelConfig,
)

# ---------------------------------------------------------------------------
# Stochastic depth (per-sample drop-path)
# ---------------------------------------------------------------------------

def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth.

    Correct implementation: sample U ~ U[0,1), add keep_prob → floor → binary mask.
    Then scale by 1/keep_prob so expectation is preserved.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = random_tensor.floor()   # 0 with prob=drop_prob, 1 otherwise
    return x.div(keep_prob) * binary_mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Input projection + region embedding
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """Linear(N_FEATURES → D_MODEL) + LayerNorm + region embedding.

    region_ids: 0=PAD, 1=KNOWN, 2=ANCHOR, 3=HIDDEN
    """

    N_REGIONS = 4  # max region id (0..3); padding_idx=0

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.06):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Region embedding: small bias on top of projected features
        self.region_emb = nn.Embedding(self.N_REGIONS + 1, d_model, padding_idx=0)
        nn.init.trunc_normal_(self.region_emb.weight, std=0.01)
        nn.init.zeros_(self.region_emb.weight[0])  # keep PAD=0
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N_FEATURES), region_ids: (B, L) long
        h = self.norm(self.proj(x))
        h = h + self.region_emb(region_ids)
        return self.drop(h)


# ---------------------------------------------------------------------------
# ConvStem: 4 dilated depth-sep Conv1d blocks
# ---------------------------------------------------------------------------

class DepthSepConvBlock(nn.Module):
    """Pre-norm depthwise-separable Conv1d with expansion + stoch depth."""

    def __init__(
        self,
        d_model: int,
        dilation: int = 1,
        expand: int = 2,
        dropout: float = 0.06,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        pad = dilation  # kernel_size=3 → same-padding with dilation
        # depthwise
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=pad,
                            dilation=dilation, groups=d_model, bias=False)
        # pointwise: expand then contract
        mid = d_model * expand
        self.pw1 = nn.Conv1d(d_model, mid, kernel_size=1)
        self.pw2 = nn.Conv1d(mid, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        h = self.norm(x).transpose(1, 2)   # (B, D, L)
        if pad_mask is not None:
            h = h * (~pad_mask).unsqueeze(1).float()
        h = self.act(self.dw(h))
        h = self.act(self.pw1(h))
        h = self.drop(self.pw2(h))
        h = h.transpose(1, 2)              # (B, L, D)
        return residual + self.drop_path(h)


# ---------------------------------------------------------------------------
# Transformer Encoder (pre-norm)
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with stochastic depth."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.06,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention (pre-norm)
        normed = self.norm1(x)
        h, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.drop_path(h)
        # Feed-forward (pre-norm)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Segment Query Decoder
# ---------------------------------------------------------------------------

class _DecoderLayer(nn.Module):
    """One decoder layer: self-attn on queries → cross-attn against encoder → FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.06):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q: (B, K, D)  kv (memory): (B, L, D)
        normed = self.norm1(q)
        h, _ = self.self_attn(normed, normed, normed)
        q = q + h
        h, _ = self.cross_attn(self.norm2(q), kv, kv, key_padding_mask=memory_key_padding_mask)
        q = q + h
        q = q + self.ff(self.norm3(q))
        return q


class SegmentQueryDecoder(nn.Module):
    """K_SEG learned queries cross-attend to encoder output.

    Queries start with truncated-normal init; heads are zero-init separately.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        k_seg: int,
        n_layers: int = 2,
        dropout: float = 0.06,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(k_seg, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([
            _DecoderLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # memory: (B, L, D) → returns (B, K_SEG, D)
        B = memory.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        for layer in self.layers:
            q = layer(q, memory, memory_key_padding_mask)
        return self.out_norm(q)


# ---------------------------------------------------------------------------
# Output heads (ALL zero-initialized so initial pred = 0 ≡ prior_tvt)
# ---------------------------------------------------------------------------

def _zero_linear(in_dim: int, out_dim: int) -> nn.Linear:
    lin = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class SegmentSlopeHead(nn.Module):
    """(B, K, D) → (B, K) slopes ft/row, bounded by tanh × bound."""

    def __init__(self, d_model: int, k_seg: int, bound: float = 0.08):
        super().__init__()
        self.proj = _zero_linear(d_model, 1)
        self.bound = bound

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(q).squeeze(-1)) * self.bound  # (B, K)


class BucketSoftHead(nn.Module):
    """(B, L, D) → (B, L, N_BUCKETS) logits."""

    def __init__(self, d_model: int, n_buckets: int):
        super().__init__()
        self.proj = _zero_linear(d_model, n_buckets)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # (B, L, N_BUCKETS)


class DirectResidualHead(nn.Module):
    """(B, L, D) → (B, L) residual ft, bounded by tanh × bound."""

    def __init__(self, d_model: int, bound: float = 6.0):
        super().__init__()
        self.proj = _zero_linear(d_model, 1)
        self.bound = bound

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(h).squeeze(-1)) * self.bound  # (B, L)


class EventHead(nn.Module):
    """(B, L, D) → (B, L) binary event logits."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = _zero_linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).squeeze(-1)  # (B, L)


class TopStateHead(nn.Module):
    """(B, L, D) → (B, L) top-state binary logits."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = _zero_linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).squeeze(-1)  # (B, L)


class TopDirectionHead(nn.Module):
    """(B, L, D) → (B, L, 3) top direction logits."""

    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.proj = _zero_linear(d_model, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


# ---------------------------------------------------------------------------
# RACFormer forward output
# ---------------------------------------------------------------------------

@dataclass
class RACFormerOutput:
    s_pred: torch.Tensor            # (B, K_SEG) segment slopes ft/row
    bucket_logits: torch.Tensor     # (B, L, N_BUCKETS) distributional logits
    direct_resid_step: torch.Tensor # (B, L) per-step direct residual ft
    event_logits: torch.Tensor      # (B, L) event binary logits
    top_event_logits: torch.Tensor  # (B, L) ANCC top-event logits
    top_dir_logits: torch.Tensor    # (B, L, 3) ANCC direction logits
    encoder_out: torch.Tensor       # (B, L, D_MODEL)
    top_state_logits: torch.Tensor | None = None  # backwards-compatible alias for top_event_logits


def load_state_dict_allowing_top_teacher_heads(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load checkpoints created before the ANCC direction head was added.

    Only the new `top_dir_head.*` parameters may be missing. Any other missing
    or unexpected key still indicates an incompatible checkpoint.
    """
    result = model.load_state_dict(state_dict, strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("top_dir_head.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state dict is incompatible: "
            f"missing={bad_missing}, unexpected={result.unexpected_keys}"
        )


# ---------------------------------------------------------------------------
# Row materializer
# ---------------------------------------------------------------------------

def materialize_rows(
    s_pred: torch.Tensor,              # (B, K_SEG)
    direct_resid_step: torch.Tensor,   # (B, L)
    base_tvt_hidden: torch.Tensor,    # (B, H_max) padded
    n_hidden_rows: torch.Tensor,       # (B,) int
    hidden_row_to_step: torch.Tensor,  # (B, H_max) long — abs step index per hidden row
    anchor_step: torch.Tensor,         # (B,) int
    k_seg: int = K_SEG,
    anchor_dr_zero: bool = True,
) -> torch.Tensor:                     # (B, H_max) predicted TVT
    """Convert segment slopes + step-level direct residuals → per-row TVT.

    Design matrix M follows compute_segment_oracle exactly:
        p[h]   = h / max(H-1, 1)
        seg[h] = floor(p[h] * k_seg).clamp(0, k_seg-1)
        M      = cumsum(one_hot(seg, k_seg), dim=0)
        R_seg  = M @ s_pred

    Direct residual uses dataset's exact hidden_row_to_step mapping (handles
    partial anchor step + bin_shift correctly).  Anchor-zeroed: subtract the
    direct_resid_step value at the anchor step so the correction is zero at
    the anchor row.
    """
    B, H_max = base_tvt_hidden.shape
    device = s_pred.device
    pred = base_tvt_hidden.clone()

    L = direct_resid_step.shape[1]
    for b in range(B):
        H_b = int(n_hidden_rows[b].item())
        if H_b <= 0:
            continue

        # Segment design matrix — matches oracle exactly
        p = torch.arange(H_b, device=device, dtype=torch.float32) / max(H_b - 1, 1)
        seg = (p * k_seg).floor().long().clamp(0, k_seg - 1)  # (H_b,)
        one_hot = F.one_hot(seg, k_seg).float()                # (H_b, K_SEG)
        M = one_hot.cumsum(dim=0)                              # (H_b, K_SEG)
        R_seg = M @ s_pred[b]                                  # (H_b,)

        # Direct residual: use exact step mapping from dataset
        t_abs = hidden_row_to_step[b, :H_b].clamp(0, L - 1)
        dr = direct_resid_step[b, t_abs]                       # (H_b,)
        if anchor_dr_zero:
            a_step = int(anchor_step[b].item())
            a_step = max(0, min(a_step, L - 1))
            dr = dr - direct_resid_step[b, a_step]

        pred[b, :H_b] = base_tvt_hidden[b, :H_b] + R_seg + dr

    return pred


# ---------------------------------------------------------------------------
# RACFormer (main model)
# ---------------------------------------------------------------------------

class RACFormer(nn.Module):
    """Residual Anchored C-field Transformer.

    Args:
        cfg: RACModelConfig instance (from config.py)
        n_features: input feature dimension (default N_FEATURES=80)
    """

    def __init__(self, cfg: RACModelConfig, n_features: int = N_FEATURES):
        super().__init__()
        d = cfg.d_model

        # ---- Input ----
        self.input_proj = InputProjection(n_features, d, cfg.dropout)

        # ---- ConvStem: dilations 1, 2, 4, 8 ----
        self.conv_stem = nn.ModuleList([
            DepthSepConvBlock(d, dilation=dil, expand=2, dropout=cfg.dropout,
                              drop_path_rate=cfg.stoch_depth)
            for dil in [1, 2, 4, 8]
        ])

        # ---- Transformer encoder ----
        self.encoder = nn.ModuleList([
            EncoderLayer(d, cfg.n_heads, cfg.ff_dim, cfg.dropout, cfg.stoch_depth)
            for _ in range(cfg.enc_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d)

        # ---- Segment query decoder ----
        self.decoder = SegmentQueryDecoder(
            d_model=d,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            k_seg=cfg.k_seg,
            n_layers=cfg.dec_layers,
            dropout=cfg.dropout,
        )

        # ---- Output heads (zero-initialized) ----
        self.seg_slope_head = SegmentSlopeHead(d, cfg.k_seg, cfg.seg_slope_bound)
        self.bucket_head = BucketSoftHead(d, cfg.n_buckets)
        self.direct_resid_head = DirectResidualHead(d, cfg.direct_resid_bound)
        self.event_head = EventHead(d)
        self.top_state_head = TopStateHead(d)
        self.top_dir_head = TopDirectionHead(d, 3)

        self.cfg = cfg

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> RACFormerOutput:
        """Run forward pass.

        Batch keys (from RACDataset):
          features    (B, L, N_FEATURES)  float32
          region_ids  (B, L)              int64
          pad_mask    (B, L)              bool  True=padding
        """
        features = batch["features"]      # (B, L, N_FEATURES)
        region_ids = batch["region_ids"]  # (B, L)
        pad_mask = batch["pad_mask"]      # (B, L)

        # Input
        h = self.input_proj(features, region_ids)      # (B, L, D)

        # ConvStem
        for block in self.conv_stem:
            h = block(h, pad_mask)

        # Transformer encoder
        for layer in self.encoder:
            h = layer(h, key_padding_mask=pad_mask)
        h = self.encoder_norm(h)
        encoder_out = h  # (B, L, D)

        # Segment decoder
        seg_q = self.decoder(encoder_out, memory_key_padding_mask=pad_mask)  # (B, K, D)

        # Heads
        s_pred = self.seg_slope_head(seg_q)                        # (B, K)
        bucket_logits = self.bucket_head(encoder_out)               # (B, L, N_BUCKETS)
        direct_resid_step = self.direct_resid_head(encoder_out)     # (B, L)
        event_logits = self.event_head(encoder_out)                 # (B, L)
        top_event_logits = self.top_state_head(encoder_out)         # (B, L)
        top_dir_logits = self.top_dir_head(encoder_out)             # (B, L, 3)

        return RACFormerOutput(
            s_pred=s_pred,
            bucket_logits=bucket_logits,
            direct_resid_step=direct_resid_step,
            event_logits=event_logits,
            top_event_logits=top_event_logits,
            top_dir_logits=top_dir_logits,
            encoder_out=encoder_out,
            top_state_logits=top_event_logits,
        )

    def materialize(self, batch: dict, out: RACFormerOutput) -> torch.Tensor:
        """Materialize per-hidden-row TVT predictions from model output.

        Returns (B, H_max) tensor (padded to batch-max hidden rows).
        Requires batch keys: base_tvt_hidden, n_hidden_rows, hidden_row_to_step, anchor_step.
        """
        return materialize_rows(
            s_pred=out.s_pred,
            direct_resid_step=out.direct_resid_step,
            base_tvt_hidden=batch["base_tvt_hidden"],
            n_hidden_rows=batch["n_hidden_rows"],
            hidden_row_to_step=batch["hidden_row_to_step"],
            anchor_step=batch["anchor_step"],
            k_seg=self.cfg.k_seg,
            anchor_dr_zero=True,
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
