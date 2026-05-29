"""RAC-Former v1 loss functions.

All loss terms operate on NORMALIZED units so weights in RACTrainConfig
are interpretable and on similar scales:

  TVT err          / 10.0   ft   → unitless ~O(1)
  endpoint err     / 10.0   ft   → unitless
  segment slope    / 0.01   ft/row → unitless
  smoothness diff  / 0.02   ft/row → unitless
  direct resid     / 5.0    ft   → unitless
  local mean-dR    / 0.01   ft/row → unitless

Loss terms and weights (from RACTrainConfig):
  L_tvt_mse    w=1.00  MSE(err/10) on hidden TVT rows
  L_tvt_huber  w=0.25  Smooth-L1 on err/5
  L_endpoint   w=0.15  MSE(err/10) on last hidden row
  L_seg        w=0.40  Smooth-L1 on (s_pred-s_star)/0.01
  L_local      w=0.10  MSE on bucket-expected dR vs true mean dR per hidden step
  L_smooth     w=0.03  Mean |Δs|/0.02 between consecutive segment slopes
  L_direct_reg w=0.02  L2 on (direct/5)² over hidden steps
  L_event      w=0.05  Focal BCE on event detection (|dC| > threshold)
  L_bucket     w=0.03  KL: soft-bucket distribution vs Gaussian target around mean dR

Notes:
  - Local + bucket heads target the RESIDUAL derivative dR/row (TVT - prior),
    not full dTVT — matches the C-field residual parameterization.
  - top-event/top-direction heads use train-only ANCC teacher labels when
    available; unavailable labels are masked out.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    BUCKET_HI,
    BUCKET_LO,
    EVENT_THRESHOLD,
    N_BUCKETS,
    RACModelConfig,
    RACTrainConfig,
)
from .model import RACFormerOutput

# Normalization scales (keep all loss terms on ~O(1))
SCALE_TVT = 10.0        # ft
SCALE_DIRECT = 5.0      # ft
SCALE_SLOPE = 0.01      # ft/row
SCALE_SLOPE_DIFF = 0.02  # ft/row


# ---------------------------------------------------------------------------
# Focal binary cross-entropy
# ---------------------------------------------------------------------------

def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * (1.0 - p_t) ** gamma * bce
    return focal.mean()


# ---------------------------------------------------------------------------
# Soft-bucket distribution helpers (operate in dR space, ft/row)
# ---------------------------------------------------------------------------

def _bucket_centers(n_buckets: int = N_BUCKETS, lo: float = BUCKET_LO, hi: float = BUCKET_HI) -> torch.Tensor:
    return torch.linspace(lo, hi, n_buckets)


def soft_bucket_target(
    mean_dr: torch.Tensor,   # (N,) mean dR/row (RESIDUAL derivative)
    sigma: float,
    n_buckets: int = N_BUCKETS,
    lo: float = BUCKET_LO,
    hi: float = BUCKET_HI,
) -> torch.Tensor:
    centers = _bucket_centers(n_buckets, lo, hi).to(mean_dr.device)
    diff = mean_dr.unsqueeze(-1) - centers.unsqueeze(0)
    log_probs = -0.5 * (diff / sigma) ** 2
    return F.softmax(log_probs, dim=-1)


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

class RACLoss(nn.Module):
    """All loss terms, normalized to ~O(1) so config weights are interpretable."""

    def __init__(self, train_cfg: RACTrainConfig, model_cfg: RACModelConfig):
        super().__init__()
        self.cfg = train_cfg
        self.mcfg = model_cfg

    def forward(
        self,
        out: RACFormerOutput,
        batch: dict,
        pred_tvt_hidden: torch.Tensor,  # (B, H_max)
        weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        B = pred_tvt_hidden.shape[0]
        device = pred_tvt_hidden.device

        n_hidden = batch["n_hidden_rows"]           # (B,) long
        tvt_hidden = batch["tvt_hidden"]            # (B, H_max)

        H_max = tvt_hidden.shape[1]

        # Row-validity mask (B, H_max)
        row_idx = torch.arange(H_max, device=device).unsqueeze(0)
        row_valid = row_idx < n_hidden.unsqueeze(1)
        valid_f = row_valid.float()
        n_valid = valid_f.sum().clamp(min=1.0)

        # ---- 1. TVT MSE (normalized) ----
        err = pred_tvt_hidden - tvt_hidden
        err_n = err / SCALE_TVT
        L_tvt_mse = ((err_n ** 2) * valid_f).sum() / n_valid

        # ---- 2. TVT Huber (smooth-L1 on err/SCALE) ----
        if cfg.w_tvt_huber != 0:
            # smooth_l1_loss with manual masking
            hub_per = F.smooth_l1_loss(
                (err / SCALE_DIRECT), torch.zeros_like(err), beta=1.0, reduction="none"
            )
            L_tvt_huber = (hub_per * valid_f).sum() / n_valid
        else:
            L_tvt_huber = pred_tvt_hidden.new_zeros(())

        # ---- 3. Endpoint loss (last hidden row, normalized) ----
        if cfg.w_endpoint != 0:
            ep_terms = []
            for b in range(B):
                H_b = int(n_hidden[b].item())
                if H_b <= 0:
                    continue
                ep_err = (pred_tvt_hidden[b, H_b - 1] - tvt_hidden[b, H_b - 1]) / SCALE_TVT
                ep_terms.append(ep_err ** 2)
            L_endpoint = torch.stack(ep_terms).mean() if ep_terms else pred_tvt_hidden.new_zeros(())
        else:
            L_endpoint = pred_tvt_hidden.new_zeros(())

        # ---- 4. Segment slope loss (normalized smooth-L1) ----
        if cfg.w_seg != 0:
            s_star = batch["s_star"]                    # (B, K_SEG)
            seg_err_n = (out.s_pred - s_star) / SCALE_SLOPE
            L_seg = F.smooth_l1_loss(seg_err_n, torch.zeros_like(seg_err_n), beta=1.0)
        else:
            L_seg = pred_tvt_hidden.new_zeros(())

        # ---- 5. Smoothness (normalized L1 on consecutive slopes) ----
        if cfg.w_smooth != 0:
            slope_diff = (out.s_pred[:, 1:] - out.s_pred[:, :-1]) / SCALE_SLOPE_DIFF
            L_smooth = slope_diff.abs().mean()
        else:
            L_smooth = pred_tvt_hidden.new_zeros(())

        # ---- 6. Direct residual L2 (normalized) ----
        if cfg.w_direct_reg != 0:
            hidden_mask = batch["hidden_mask"]          # (B, L) bool
            direct_n = out.direct_resid_step / SCALE_DIRECT
            L_direct_reg_t = (direct_n ** 2) * hidden_mask.float()
            n_hidden_steps_total = hidden_mask.float().sum().clamp(min=1.0)
            L_direct_reg = L_direct_reg_t.sum() / n_hidden_steps_total
        else:
            L_direct_reg = pred_tvt_hidden.new_zeros(())

        # ---- 7 + 8 + 9.  Per-hidden-step targets (local / bucket / event) ----
        # All three iterate over the same set of (b, t_abs, r_start, r_end) tuples,
        # so compute them in a single pass.
        need_step_aux = cfg.w_local != 0 or cfg.w_bucket != 0 or cfg.w_event != 0
        if need_step_aux:
            base_hidden = batch["base_tvt_hidden"]    # (B, H_max)
            dC_hidden = batch["dC_hidden"]              # (B, H_max)
            hidden_row_to_step = batch["hidden_row_to_step"]  # (B, H_max) long
            L = out.event_logits.shape[1]
            centers = _bucket_centers(self.mcfg.n_buckets, self.mcfg.bucket_lo, self.mcfg.bucket_hi).to(device)
            local_terms = []
            bucket_terms = []
            event_logits_list = []
            event_labels_list = []

            for b in range(B):
                H_b = int(n_hidden[b].item())
                if H_b <= 0:
                    continue
                # Per-step partition: group hidden rows by their absolute step index
                steps_b = hidden_row_to_step[b, :H_b]  # (H_b,) long
                # Iterate unique steps in order
                unique_steps, counts = torch.unique_consecutive(steps_b, return_counts=True)
                offset = 0
                for t_abs_t, cnt_t in zip(unique_steps, counts):
                    t_abs = int(t_abs_t.item())
                    cnt = int(cnt_t.item())
                    r_start = offset
                    r_end = offset + cnt
                    offset = r_end
                    if t_abs < 0 or t_abs >= L:
                        continue
                    if cnt < 2:
                        continue

                    # True residual derivative dR/row over this step
                    tvt_step = tvt_hidden[b, r_start:r_end]
                    base_step = base_hidden[b, r_start:r_end]
                    R_step = tvt_step - base_step
                    mean_dR = (R_step[-1] - R_step[0]) / (cnt - 1)

                    if cfg.w_local != 0 or cfg.w_bucket != 0:
                        logits_t = out.bucket_logits[b, t_abs]

                    # ---- local: bucket expectation vs mean_dR ----
                    if cfg.w_local != 0:
                        probs_t = F.softmax(logits_t, dim=-1)
                        pred_mean_dR = (probs_t * centers).sum()
                        local_terms.append(((pred_mean_dR - mean_dR) / SCALE_SLOPE) ** 2)

                    # ---- bucket KL ----
                    if cfg.w_bucket != 0:
                        target_dist = soft_bucket_target(
                            mean_dR.detach().unsqueeze(0), cfg.bucket_soft_sigma,
                            self.mcfg.n_buckets, self.mcfg.bucket_lo, self.mcfg.bucket_hi,
                        )  # (1, B)
                        log_pred = F.log_softmax(logits_t.unsqueeze(0), dim=-1)
                        bucket_terms.append(F.kl_div(log_pred, target_dist, reduction="batchmean"))

                    # ---- event: |dC| > threshold (uses dC_hidden, train-only) ----
                    if cfg.w_event != 0:
                        dC_step = dC_hidden[b, r_start:r_end]
                        mean_abs_dC = dC_step.abs().mean()
                        event_labels_list.append((mean_abs_dC > EVENT_THRESHOLD).float())
                        event_logits_list.append(out.event_logits[b, t_abs])

            L_local = torch.stack(local_terms).mean() if local_terms else pred_tvt_hidden.new_zeros(())
            L_bucket = torch.stack(bucket_terms).mean() if bucket_terms else pred_tvt_hidden.new_zeros(())
            if event_logits_list:
                ev_logits = torch.stack(event_logits_list)
                ev_targets = torch.stack(event_labels_list)
                L_event = focal_bce_loss(ev_logits, ev_targets, cfg.event_focal_alpha, cfg.event_focal_gamma)
            else:
                L_event = pred_tvt_hidden.new_zeros(())
        else:
            L_local = pred_tvt_hidden.new_zeros(())
            L_bucket = pred_tvt_hidden.new_zeros(())
            L_event = pred_tvt_hidden.new_zeros(())

        # ---- 10 + 11. ANCC top teacher distillation (train-only targets) ----
        need_top = cfg.w_top_event != 0 or cfg.w_top_dir != 0
        if need_top:
            top_teacher_mask = batch.get("top_teacher_mask")
            if top_teacher_mask is not None:
                top_mask = top_teacher_mask.to(device).bool()
                top_event_target = batch["top_event_step"].to(device).float()
                top_state_target = batch["top_state_step"].to(device).long()
            else:
                top_mask = torch.zeros_like(out.top_event_logits, dtype=torch.bool)
                top_event_target = torch.zeros_like(out.top_event_logits)
                top_state_target = torch.full_like(out.top_event_logits, -100, dtype=torch.long)

            if cfg.w_top_event != 0 and top_mask.any():
                top_event_per = F.binary_cross_entropy_with_logits(
                    out.top_event_logits,
                    top_event_target,
                    reduction="none",
                )
                L_top_event = top_event_per[top_mask].mean()
            else:
                L_top_event = pred_tvt_hidden.new_zeros(())

            top_dir_mask = top_mask & (top_event_target > 0.5) & (top_state_target != -100)
            if cfg.w_top_dir != 0 and top_dir_mask.any():
                L_top_dir = F.cross_entropy(out.top_dir_logits[top_dir_mask], top_state_target[top_dir_mask])
            else:
                L_top_dir = pred_tvt_hidden.new_zeros(())
        else:
            L_top_event = pred_tvt_hidden.new_zeros(())
            L_top_dir = pred_tvt_hidden.new_zeros(())

        # ---- Combine ----
        total = (
            cfg.w_tvt_mse * L_tvt_mse
            + cfg.w_tvt_huber * L_tvt_huber
            + cfg.w_endpoint * L_endpoint
            + cfg.w_seg * L_seg
            + cfg.w_local * L_local
            + cfg.w_smooth * L_smooth
            + cfg.w_direct_reg * L_direct_reg
            + cfg.w_event * L_event
            + cfg.w_bucket * L_bucket
            + cfg.w_top_event * L_top_event
            + cfg.w_top_dir * L_top_dir
        ) * weight

        # Diagnostic: raw row-RMSE (in ft, not normalized)
        with torch.no_grad():
            raw_sq = (err ** 2 * valid_f).sum()
            row_rmse = (raw_sq / n_valid).sqrt()

        return {
            "total": total,
            "tvt_mse": L_tvt_mse.detach(),
            "tvt_huber": L_tvt_huber.detach(),
            "endpoint": L_endpoint.detach(),
            "seg": L_seg.detach(),
            "local": L_local.detach(),
            "smooth": L_smooth.detach(),
            "direct_reg": L_direct_reg.detach(),
            "event": L_event.detach(),
            "bucket": L_bucket.detach(),
            "top_event": L_top_event.detach(),
            "top_dir": L_top_dir.detach(),
            "row_rmse": row_rmse.detach(),
        }


# ---------------------------------------------------------------------------
# Per-well RMSE for checkpoint scoring (unchanged from prior version)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_oof_rmse(
    pred_tvt_hidden: torch.Tensor,
    tvt_hidden: torch.Tensor,
    n_hidden_rows: torch.Tensor,
) -> tuple[float, list[float]]:
    B, H_max = pred_tvt_hidden.shape

    all_sq_err = []
    per_well = []
    for b in range(B):
        H_b = int(n_hidden_rows[b].item())
        if H_b <= 0:
            per_well.append(0.0)
            continue
        err = pred_tvt_hidden[b, :H_b] - tvt_hidden[b, :H_b]
        mse_b = (err ** 2).mean().item()
        per_well.append(mse_b ** 0.5)
        all_sq_err.append(err ** 2)

    if not all_sq_err:
        return 0.0, per_well

    pooled_sq = torch.cat(all_sq_err)
    pooled_rmse = float(pooled_sq.mean().sqrt().item())
    return pooled_rmse, per_well
