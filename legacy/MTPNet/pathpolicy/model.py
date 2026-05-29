"""
pathpolicy/model.py — PolicyFormer architecture.

Two head modes, selected via head_type:

  "classifier"     (default / backward-compat)
      ContextEncoder → ctx_emb [B, d]
      ActionEncoder  → act_emb [B, K, d]
      dot-product (q·k / sqrt(d)) → logits [B, K]   (−inf for unavailable)
      value MLP → value [B]
      forward returns (logits [B, K], value [B])

  "rmse_regressor"
      Same encoders, different head:
      for each action i: cat(ctx_emb, act_emb_i) → MLP → log1p-RMSE scalar
      Returns (pred_log_rmse [B, K], value [B])
      Training loss: Huber(pred_log_rmse, log1p(oracle_rmse_per_action))
      Inference:     argmin(pred_log_rmse[available_actions])
      B2 safety bias option: subtract b2_bonus from b2 slot before argmin
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from pathpolicy.actions import N_ACTION_TYPES, CHUNK_LEN
from pathpolicy.dataset import D_FEAT, PAST_LEN


# ---------------------------------------------------------------------------
# Context encoder
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """Encode the context window [B, T, D_FEAT] → [B, d_model]."""

    def __init__(
        self,
        d_feat: int = D_FEAT,
        d_model: int = 128,
        n_layers: int = 2,
        nhead: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = PAST_LEN + CHUNK_LEN,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learnable positional encoding
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D_FEAT]
        # All rows (including hidden chunk rows with is_hidden=1) are pooled.
        # Hidden rows carry useful forward GR signal — don't exclude them.
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        h = self.input_proj(x) + self.pos_emb(pos)           # [B, T, d_model]
        h = self.transformer(h)                               # [B, T, d_model]
        return h.mean(dim=1)                                  # [B, d_model]


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    """Encode action segments [B, K, L] + type_ids [B, K] → [B, K, d_model]."""

    def __init__(
        self,
        d_model: int = 128,
        n_action_types: int = N_ACTION_TYPES,
        seg_len: int = CHUNK_LEN,
        conv_channels: int = 64,
        conv_kernel: int = 16,
        conv_stride: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1-D conv to downsample the TVT delta segment
        n_tokens = seg_len // conv_stride  # e.g., 256//16 = 16
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=conv_kernel, stride=conv_stride),
            nn.GELU(),
            nn.Conv1d(conv_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.seg_pool = nn.AdaptiveAvgPool1d(1)

        # Type embedding
        self.type_emb = nn.Embedding(n_action_types, d_model)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, segs: Tensor, type_ids: Tensor) -> Tensor:
        # segs: [B, K, L]   — normalized TVT deltas
        # type_ids: [B, K]   — action type indices
        B, K, L = segs.shape

        # Encode segments via conv
        flat = segs.reshape(B * K, 1, L)             # [B*K, 1, L]
        conv_out = self.conv(flat)                    # [B*K, d_model, T']
        seg_emb = self.seg_pool(conv_out).squeeze(-1) # [B*K, d_model]
        seg_emb = seg_emb.reshape(B, K, -1)           # [B, K, d_model]

        # Encode type
        type_emb = self.type_emb(type_ids)            # [B, K, d_model]

        # Fuse
        fused = self.fusion(torch.cat([seg_emb, type_emb], dim=-1))  # [B, K, d_model]
        return fused


# ---------------------------------------------------------------------------
# PolicyFormer
# ---------------------------------------------------------------------------

class PolicyFormer(nn.Module):
    """Context-conditioned action selector.

    head_type="classifier"    — dot-product logits + cross-entropy training
    head_type="rmse_regressor" — per-action log1p-RMSE prediction + Huber training
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        nhead: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        head_type: str = "classifier",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.head_type = head_type

        # ── Shared encoders ────────────────────────────────────────────────
        self.ctx_encoder = ContextEncoder(
            d_feat=D_FEAT,
            d_model=d_model,
            n_layers=n_layers,
            nhead=nhead,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.act_encoder = ActionEncoder(
            d_model=d_model,
            n_action_types=N_ACTION_TYPES,
            seg_len=CHUNK_LEN,
            dropout=dropout,
        )

        # ── Head ───────────────────────────────────────────────────────────
        if head_type == "rmse_regressor":
            # Per-action log1p-RMSE prediction head.
            # Input: cat(ctx_emb, act_emb_i) [B, K, 2*d] → scalar [B, K].
            # No Softplus — we work in log space; argmin is preserved.
            self.rmse_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        else:
            # Original dot-product classification heads
            self.query_proj = nn.Linear(d_model, d_model)
            self.key_proj   = nn.Linear(d_model, d_model)

        # ── Value head (shared) — predicts oracle_rmse scalar for this chunk ─
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # value ≥ 0 (it's an RMSE)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        context: Tensor,          # [B, T, D_FEAT]
        action_segs: Tensor,      # [B, K, CHUNK_LEN] normalized deltas
        action_type_ids: Tensor,  # [B, K] long
        action_mask: Tensor,      # [B, K] bool (True = available)
    ) -> tuple[Tensor, Tensor]:
        """Return (scores [B, K], value [B]).

        head_type="classifier":
            scores = dot-product logits; unavailable slots = −inf.
            Inference: argmax(scores).

        head_type="rmse_regressor":
            scores = predicted log1p(RMSE) per action; no masking applied here.
            Inference: argmin(scores) after masking unavailable slots with large value.
        """
        # ── Shared encoding ────────────────────────────────────────────────
        ctx_emb = self.ctx_encoder(context)                          # [B, d]
        act_emb = self.act_encoder(action_segs, action_type_ids)     # [B, K, d]
        value   = self.value_head(ctx_emb).squeeze(-1)               # [B]

        # ── Head-specific scoring ──────────────────────────────────────────
        if self.head_type == "rmse_regressor":
            B, K, d = act_emb.shape
            ctx_expand = ctx_emb.unsqueeze(1).expand(-1, K, -1)     # [B, K, d]
            pair = torch.cat([ctx_expand, act_emb], dim=-1)          # [B, K, 2*d]
            scores = self.rmse_head(pair).squeeze(-1)                 # [B, K]
            # NOTE: caller is responsible for masking before argmin
        else:
            # Dot-product scoring
            q = self.query_proj(ctx_emb)                             # [B, d]
            k = self.key_proj(act_emb)                               # [B, K, d]
            scale = math.sqrt(self.d_model)
            scores = torch.einsum("bd,bkd->bk", q, k) / scale       # [B, K]
            scores = scores.masked_fill(~action_mask, float("-inf"))

        return scores, value

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
