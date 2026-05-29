"""All loss functions for BPHWT training.

L = w_tvt * L_tvt
  + w_forward_gr * L_forward_gr
  + w_anchor * L_anchor
  + w_velocity * L_velocity
  + w_dip_sign * L_dip_sign
  + w_seg_boundary * L_seg_boundary
  + w_smooth * L_smooth
  + w_nll * L_nll
  + w_distill * L_distill
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------


def tvt_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 5.0) -> torch.Tensor:
    """Huber loss on TVT."""
    return F.huber_loss(pred, target, delta=delta, reduction="none")


def anchor_loss(
    tvt_pred: torch.Tensor,  # [B, L]
    tvt_input: torch.Tensor,  # [B, L]  NaN where hidden
    known_mask: torch.Tensor,  # [B, L]
    delta: float = 2.0,
) -> torch.Tensor:
    """Enforce TVT_input at known rows."""
    km = known_mask.bool()
    if not km.any():
        return torch.tensor(0.0, device=tvt_pred.device)
    diff = tvt_pred[km] - tvt_input[km]
    return F.huber_loss(diff, torch.zeros_like(diff), delta=delta)


def forward_gr_loss(
    tvt_pred: torch.Tensor,  # [B, L]
    gr_obs: torch.Tensor,  # [B, L]  NaN where invalid
    gr_valid: torch.Tensor,  # [B, L]  bool/float mask
    tw_tvt: torch.Tensor,  # [B, M]  typewell TVT (monotone increasing)
    tw_gr: torch.Tensor,  # [B, M]  typewell GR
    beta: torch.Tensor | None = None,  # [B, 2] (beta0, beta1); None = per-call LS
    delta: float = 10.0,
    w_grad: float = 0.5,
    w_corr: float = 0.2,
    corr_window: int = 64,
) -> torch.Tensor:
    """
    Physics-informed GR loss:

    GR_forward[t] = beta1 * typewell_GR(TVT_pred[t]) + beta0
    Compare GR_forward with observed GR on valid rows.

    Uses differentiable linear interpolation.
    """
    B, L = tvt_pred.shape
    total = torch.tensor(0.0, device=tvt_pred.device)
    count = 0

    for b in range(B):
        gv = gr_valid[b]  # [L]
        if gv.float().sum() < 5:
            continue

        # Forward-interpolate typewell GR at predicted TVT
        gr_pred_raw = _differentiable_interp(tvt_pred[b], tw_tvt[b], tw_gr[b])  # [L]

        # Estimate beta per sample
        if beta is not None:
            b0, b1 = beta[b, 0], beta[b, 1]
        else:
            b0, b1 = _estimate_beta(gr_pred_raw.detach(), gr_obs[b], gv)

        gr_forward = b1 * gr_pred_raw + b0  # [L]

        # Mask to valid rows only
        gv_bool = gv.bool()
        gf_valid = gr_forward[gv_bool]
        go_valid = gr_obs[b][gv_bool]

        # Raw Huber loss
        l_raw = F.huber_loss(gf_valid, go_valid, delta=delta)
        total = total + l_raw

        # Gradient (derivative) loss
        if w_grad > 0:
            dgf = torch.diff(gr_forward)
            dgo = torch.diff(gr_obs[b])
            dgv = gv_bool[1:] & gv_bool[:-1]
            if dgv.float().sum() > 2:
                l_grad = F.huber_loss(dgf[dgv], dgo[dgv], delta=delta / 2)
                total = total + w_grad * l_grad

        # Windowed correlation loss (1 - r)
        if w_corr > 0 and gv_bool.float().sum() > corr_window // 2:
            l_corr = _windowed_corr_loss(gr_forward, gr_obs[b], gv_bool, corr_window)
            total = total + w_corr * l_corr

        count += 1

    return total / max(count, 1)


def velocity_loss(
    vel_pred: torch.Tensor,  # [B, L]
    tvt_true: torch.Tensor,  # [B, L]
    md: torch.Tensor,  # [B, L]
    hidden_mask: torch.Tensor,  # [B, L]
    delta: float = 0.02,
) -> torch.Tensor:
    vel_true = torch.diff(tvt_true, dim=1) / (torch.diff(md, dim=1).abs().clamp(min=0.1) + _EPS)
    vel_true = torch.cat([vel_true[:, :1], vel_true], dim=1)  # [B, L]
    # Only on rows where target is available
    hm = hidden_mask.bool()
    if not hm.any():
        return torch.tensor(0.0, device=vel_pred.device)
    return F.huber_loss(vel_pred[hm], vel_true[hm], delta=delta)


def dip_sign_loss(
    logits: torch.Tensor,  # [B, 3, L]  {0:down, 1:flat, 2:up}
    tvt_true: torch.Tensor,  # [B, L]
    md: torch.Tensor,  # [B, L]
    hidden_mask: torch.Tensor,  # [B, L]
    threshold: float = 0.005,
) -> torch.Tensor:
    vel_true = torch.diff(tvt_true, dim=1) / (torch.diff(md, dim=1).abs().clamp(min=0.1) + _EPS)
    vel_true = torch.cat([vel_true[:, :1], vel_true], dim=1)  # [B, L]
    # 0=down, 1=flat, 2=up
    labels = torch.ones_like(vel_true, dtype=torch.long)
    labels[vel_true < -threshold] = 0
    labels[vel_true > threshold] = 2
    hm = hidden_mask.bool()
    if not hm.any():
        return torch.tensor(0.0, device=logits.device)
    log_probs = logits.permute(0, 2, 1)  # [B, L, 3]
    return F.cross_entropy(log_probs[hm], labels[hm])


def seg_boundary_loss(
    seg_logits: torch.Tensor,  # [B, L]  logits
    tvt_true: torch.Tensor,  # [B, L]
    md: torch.Tensor,  # [B, L]
    hidden_mask: torch.Tensor,  # [B, L]
    curvature_thresh: float = 0.01,
) -> torch.Tensor:
    # Segment boundary = large second derivative of true TVT
    dmd = torch.diff(md, dim=1).clamp(min=0.1)
    d1 = torch.diff(tvt_true, dim=1) / dmd
    d1 = torch.cat([d1[:, :1], d1], dim=1)
    d2 = torch.diff(d1, dim=1).abs()
    d2 = torch.cat([d2[:, :1], d2], dim=1)
    target = (d2 > curvature_thresh).float()

    hm = hidden_mask.bool()
    if not hm.any():
        return torch.tensor(0.0, device=seg_logits.device)
    return F.binary_cross_entropy_with_logits(seg_logits[hm], target[hm])


def smooth_loss(
    tvt_pred: torch.Tensor,  # [B, L]
    hidden_mask: torch.Tensor,  # [B, L]
    seg_prob: torch.Tensor | None = None,  # [B, L] — relax smoothness at boundaries
) -> torch.Tensor:
    d2 = tvt_pred[:, 2:] - 2 * tvt_pred[:, 1:-1] + tvt_pred[:, :-2]  # [B, L-2]
    hm = hidden_mask[:, 1:-1].bool()
    if not hm.any():
        return torch.tensor(0.0, device=tvt_pred.device)
    penalty = d2.abs()
    if seg_prob is not None:
        relax = 1.0 - seg_prob[:, 1:-1].clamp(0, 1)
        penalty = penalty * relax
    return penalty[hm].mean()


def nll_loss_student(
    tvt_pred: torch.Tensor,  # [B, L]
    log_sigma: torch.Tensor,  # [B, L]
    tvt_true: torch.Tensor,  # [B, L]
    hidden_mask: torch.Tensor,  # [B, L]
    nu: float = 4.0,
) -> torch.Tensor:
    """Student-t NLL loss."""
    hm = hidden_mask.bool()
    if not hm.any():
        return torch.tensor(0.0, device=tvt_pred.device)
    sigma = torch.exp(log_sigma[hm]).clamp(0.5, 50.0)
    z = (tvt_true[hm] - tvt_pred[hm]) / sigma
    nll = (nu + 1.0) / 2.0 * torch.log(1.0 + z**2 / nu) + torch.log(sigma)
    return nll.mean()


def distill_loss(
    tvt_pred: torch.Tensor,  # [B, L]
    tvt_candidate: torch.Tensor,  # [B, L] — best prior (e.g. HMM or linear)
    confidence: torch.Tensor,  # [B, L] — 0..1, how much to trust candidate
    hidden_mask: torch.Tensor,  # [B, L]
    delta: float = 5.0,
) -> torch.Tensor:
    hm = hidden_mask.bool()
    if not hm.any():
        return torch.tensor(0.0, device=tvt_pred.device)
    w = confidence[hm]
    diff = tvt_pred[hm] - tvt_candidate[hm]
    loss_values = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction="none")
    return (loss_values * w).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


class BPHWTLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        lc = self.cfg
        device = preds["tvt_pred"].device

        tvt_p = preds["tvt_pred"]
        tvt_r = preds["tvt_raw"]
        hm = batch["hidden_mask"]  # [B, L]
        km = batch["known_mask"]  # [B, L]
        tvt_t = batch["tvt_true"]  # [B, L]
        tvt_in = batch["tvt_input_filled"]  # [B, L] (NaN-free)
        md = batch["md"]  # [B, L]

        total = torch.tensor(0.0, device=device)
        log: dict[str, float] = {}

        # Row weights
        row_w = hm.float() * lc.hidden_weight + km.float() * lc.known_weight

        # --- L_tvt ---
        if lc.w_tvt > 0:
            loss_tvt = (tvt_huber(tvt_p, tvt_t, lc.tvt_huber_delta) * row_w).sum() / (row_w.sum() + _EPS)
            total = total + lc.w_tvt * loss_tvt
            log["loss_tvt"] = loss_tvt.item()

        # --- L_anchor ---
        if lc.w_anchor > 0:
            loss_anchor = anchor_loss(tvt_p, tvt_in, km)
            total = total + lc.w_anchor * loss_anchor
            log["loss_anchor"] = loss_anchor.item()

        # --- L_forward_gr ---
        if lc.w_forward_gr > 0 and "tw_tvt" in batch:
            loss_forward_gr = forward_gr_loss(
                tvt_p,
                batch["gr_obs"],
                batch["gr_valid"],
                batch["tw_tvt"],
                batch["tw_gr"],
                delta=lc.gr_huber_delta,
                w_grad=lc.w_gr_grad,
                w_corr=lc.w_gr_corr,
                corr_window=lc.gr_corr_window,
            )
            total = total + lc.w_forward_gr * loss_forward_gr
            log["loss_fwd_gr"] = loss_forward_gr.item()

        # --- L_velocity ---
        if lc.w_velocity > 0 and "velocity" in preds and hm.any():
            loss_velocity = velocity_loss(preds["velocity"], tvt_t, md, hm)
            total = total + lc.w_velocity * loss_velocity
            log["loss_velocity"] = loss_velocity.item()

        # --- L_dip_sign ---
        if lc.w_dip_sign > 0 and "dip_sign" in preds and hm.any():
            loss_dip_sign = dip_sign_loss(preds["dip_sign"], tvt_t, md, hm, lc.dip_sign_threshold)
            total = total + lc.w_dip_sign * loss_dip_sign
            log["loss_dip_sign"] = loss_dip_sign.item()

        # --- L_seg_boundary ---
        if lc.w_seg_boundary > 0 and "seg_boundary_logits" in preds and hm.any():
            loss_seg = seg_boundary_loss(preds["seg_boundary_logits"], tvt_t, md, hm)
            total = total + lc.w_seg_boundary * loss_seg
            log["loss_seg"] = loss_seg.item()

        # --- L_smooth ---
        if lc.w_smooth > 0:
            seg_p = preds.get("seg_boundary")
            loss_smooth = smooth_loss(tvt_r, hm, seg_p)
            total = total + lc.w_smooth * loss_smooth
            log["loss_smooth"] = loss_smooth.item()

        # --- L_nll ---
        if lc.w_nll > 0 and hm.any():
            loss_nll = nll_loss_student(tvt_p, preds["log_sigma"], tvt_t, hm, lc.nll_nu)
            total = total + lc.w_nll * loss_nll
            log["loss_nll"] = loss_nll.item()

        # --- L_distill ---
        if lc.w_distill > 0 and "tvt_candidate" in batch and hm.any():
            conf = batch.get("candidate_confidence", torch.ones_like(hm, dtype=torch.float))
            loss_distill = distill_loss(tvt_p, batch["tvt_candidate"], conf, hm)
            total = total + lc.w_distill * loss_distill
            log["loss_distill"] = loss_distill.item()

        log["loss_total"] = total.item()
        return total, log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _differentiable_interp(
    x: torch.Tensor,  # [L] query points
    xp: torch.Tensor,  # [M] known x (monotone increasing)
    fp: torch.Tensor,  # [M] known values
) -> torch.Tensor:
    """
    Differentiable 1-D linear interpolation using searchsorted.
    Clamps to boundary values outside range.
    """
    M = xp.shape[0]
    # Find upper index for each query point
    idx = torch.searchsorted(xp.contiguous(), x.contiguous())  # [L]
    idx = idx.clamp(1, M - 1)
    lo = idx - 1
    hi = idx

    x_lo = xp[lo]  # [L]
    x_hi = xp[hi]
    f_lo = fp[lo]
    f_hi = fp[hi]

    dx = (x_hi - x_lo).clamp(min=1e-6)
    t = (x - x_lo) / dx  # [L]
    t = t.clamp(0.0, 1.0)
    return f_lo + t * (f_hi - f_lo)


def _estimate_beta(
    gr_pred: torch.Tensor,  # [L] detached
    gr_obs: torch.Tensor,  # [L] with NaN
    gr_valid: torch.Tensor,  # [L] bool
) -> tuple[float, float]:
    gv = gr_valid.bool()
    if gv.sum() < 5:
        return 0.0, 1.0
    gp = gr_pred[gv].float()
    go = gr_obs[gv].float()
    # LS: go = beta1 * gp + beta0
    ones = torch.ones_like(gp)
    A = torch.stack([gp, ones], dim=1)
    try:
        sol, _, _, _ = torch.linalg.lstsq(A, go.unsqueeze(1))
        beta1, beta0 = float(sol[0]), float(sol[1])
        if not (0.2 < abs(beta1) < 5.0):
            return 0.0, 1.0
        return beta0, beta1
    except Exception:
        return 0.0, 1.0


def _windowed_corr_loss(
    a: torch.Tensor,  # [L]
    b: torch.Tensor,  # [L] with NaN
    valid: torch.Tensor,  # [L] bool
    window: int,
) -> torch.Tensor:
    """Compute 1 - mean(windowed Pearson correlation) as a loss."""
    L = a.shape[0]
    correlations = []
    step = window // 2
    for start in range(0, L - window, step):
        end = start + window
        v = valid[start:end]
        if v.float().sum() < window // 4:
            continue
        a_w = a[start:end][v]
        b_w = b[start:end][v]
        if a_w.numel() < 4:
            continue
        a_c = a_w - a_w.mean()
        b_c = b_w - b_w.mean()
        num = (a_c * b_c).sum()
        denom = (a_c.pow(2).sum() * b_c.pow(2).sum()).sqrt().clamp(min=_EPS)
        correlations.append(num / denom)
    if not correlations:
        return torch.tensor(0.0, device=a.device)
    return 1.0 - torch.stack(correlations).mean()
