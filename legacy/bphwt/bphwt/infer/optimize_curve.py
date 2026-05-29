"""Per-well B-spline post-optimization.

Minimises a joint objective:
  L = Σ_gr  ρ(beta1 * tw_GR(TVT_opt) + beta0 - GR_obs)  [physics]
    + λ_prior  ||TVT_opt - TVT_nn||² / sigma²             [NN prior]
    + λ_hmm    ||TVT_opt - TVT_hmm||²                     [HMM prior]
    + λ_smooth ||D2(TVT_opt)||₁                           [smoothness]
    + λ_anchor ||TVT_opt[known] - TVT_input[known]||²     [anchors]

TVT_opt is parameterised as:
    TVT_opt = anchor_clamp(TVT_nn + B_spline @ c)
where c are n_knots free coefficients.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def optimize_well_curve(
    tvt_nn: np.ndarray,  # [L] NN prediction
    log_sigma: np.ndarray,  # [L] NN log std
    md: np.ndarray,  # [L] measured depth
    known_mask: np.ndarray,  # [L] 1=known
    tvt_input: np.ndarray,  # [L] NaN where hidden
    tvt_input_filled: np.ndarray,  # [L] filled
    gr_obs: np.ndarray,  # [L] NaN where invalid
    gr_valid: np.ndarray,  # [L] bool
    tw_tvt: np.ndarray,  # [M] typewell TVT
    tw_gr: np.ndarray,  # [M] typewell GR
    tvt_hmm: np.ndarray | None = None,  # [L] HMM prior
    n_knots: int = 48,
    n_steps: int = 150,
    lr: float = 0.5,
    lambda_prior: float = 1.0,
    lambda_hmm: float = 0.5,
    lambda_smooth: float = 0.02,
    lambda_anchor: float = 50.0,
    gr_huber_delta: float = 15.0,
    clip_correction: float = 20.0,
) -> np.ndarray:
    """
    Run per-well curve optimisation. Returns optimised TVT [L].
    """
    try:
        import torch
        import torch.nn.functional as F

        L = len(tvt_nn)

        tvt_nn_t = torch.from_numpy(tvt_nn).float()
        sigma_t = torch.exp(torch.from_numpy(np.clip(log_sigma, -4, 4)).float()).clamp(0.5, 50.0)
        km_t = torch.from_numpy(known_mask).float()
        tvt_anchor_t = torch.from_numpy(tvt_input_filled).float()
        gr_obs_t = torch.from_numpy(np.where(np.isfinite(gr_obs), gr_obs, 0.0)).float()
        gv_t = torch.from_numpy(gr_valid).float()
        tw_tvt_t = torch.from_numpy(tw_tvt).float()
        tw_gr_t = torch.from_numpy(tw_gr).float()

        # B-spline basis: uniform knots over row indices
        knot_positions = torch.linspace(0, L - 1, n_knots)
        row_idx = torch.arange(L).float()
        # RBF-style basis: Gaussian bumps at knot positions
        bandwidth = float(L) / n_knots * 1.5
        basis = torch.exp(-0.5 * ((row_idx.unsqueeze(1) - knot_positions.unsqueeze(0)) / bandwidth) ** 2)
        basis = basis / basis.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [L, K]

        # Learnable coefficients (zero-init → start from NN prediction)
        c = torch.zeros(n_knots, requires_grad=True)
        optimizer = torch.optim.Adam([c], lr=lr)

        # Estimate beta0, beta1 from current prediction (updated periodically)
        beta0, beta1 = _estimate_beta_np(tvt_nn, gr_obs, gr_valid, tw_tvt, tw_gr)

        best_loss = float("inf")
        best_c = c.data.clone()

        for step in range(n_steps):
            optimizer.zero_grad()

            # TVT_opt = TVT_nn + spline_correction (clamped)
            correction = (basis @ c).clamp(-clip_correction, clip_correction)
            tvt_opt = tvt_nn_t + correction

            # Hard anchor clamp at known rows
            tvt_opt = km_t * tvt_anchor_t + (1.0 - km_t) * tvt_opt

            # Physics: forward GR
            gr_forward = _interp_torch(tvt_opt, tw_tvt_t, tw_gr_t) * beta1 + beta0
            gr_residual = (gr_forward - gr_obs_t) * gv_t
            l_gr = F.huber_loss(
                gr_residual, torch.zeros_like(gr_residual), delta=gr_huber_delta, reduction="none"
            )
            l_gr = (l_gr * gv_t).sum() / (gv_t.sum() + 1e-6)

            # NN prior
            l_prior = ((tvt_opt - tvt_nn_t) ** 2 / sigma_t**2).mean()

            # HMM prior (if provided)
            l_hmm = torch.tensor(0.0)
            if tvt_hmm is not None:
                tvt_hmm_t = torch.from_numpy(tvt_hmm).float()
                l_hmm = ((tvt_opt - tvt_hmm_t) ** 2).mean()

            # Smoothness
            d2 = tvt_opt[2:] - 2 * tvt_opt[1:-1] + tvt_opt[:-2]
            l_smooth = d2.abs().mean()

            # Anchor
            km_bool = km_t.bool()
            if km_bool.any():
                l_anchor = F.mse_loss(tvt_opt[km_bool], tvt_anchor_t[km_bool])
            else:
                l_anchor = torch.tensor(0.0)

            total = (
                l_gr
                + lambda_prior * l_prior
                + lambda_hmm * l_hmm
                + lambda_smooth * l_smooth
                + lambda_anchor * l_anchor
            )

            total.backward()
            optimizer.step()

            # Reestimate beta every 25 steps
            if step % 25 == 24:
                with torch.no_grad():
                    tvt_cur = tvt_nn_t + (basis @ c).clamp(-clip_correction, clip_correction)
                    beta0, beta1 = _estimate_beta_np(tvt_cur.numpy(), gr_obs, gr_valid, tw_tvt, tw_gr)

            if total.item() < best_loss:
                best_loss = total.item()
                best_c = c.data.clone()

        # Final prediction with best c
        with torch.no_grad():
            correction = (basis @ best_c).clamp(-clip_correction, clip_correction)
            tvt_opt = tvt_nn_t + correction
            tvt_opt = km_t * tvt_anchor_t + (1.0 - km_t) * tvt_opt

        return tvt_opt.numpy().astype(np.float32)

    except Exception as e:
        logger.warning(f"Post-optimization failed ({e}), returning NN prediction")
        return tvt_nn.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interp_torch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Differentiable linear interpolation."""
    M = xp.shape[0]
    idx = torch.searchsorted(xp.contiguous(), x.contiguous())
    idx = idx.clamp(1, M - 1)
    lo, hi = idx - 1, idx
    dx = (xp[hi] - xp[lo]).clamp(min=1e-6)
    t = ((x - xp[lo]) / dx).clamp(0.0, 1.0)
    return fp[lo] + t * (fp[hi] - fp[lo])


def _estimate_beta_np(
    tvt_pred: np.ndarray,
    gr_obs: np.ndarray,
    gr_valid: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
) -> tuple[float, float]:
    """Least-squares estimate of (beta0, beta1) on valid GR rows."""
    gv = gr_valid > 0.5
    if gv.sum() < 5:
        return 0.0, 1.0
    gr_fwd = np.interp(tvt_pred[gv], tw_tvt, tw_gr).astype(np.float32)
    go = gr_obs[gv].astype(np.float32)
    # Remove NaN from go
    valid2 = np.isfinite(go)
    if valid2.sum() < 5:
        return 0.0, 1.0
    A = np.column_stack([gr_fwd[valid2], np.ones(valid2.sum())])
    try:
        sol, _, _, _ = np.linalg.lstsq(A, go[valid2], rcond=None)
        b1, b0 = float(sol[0]), float(sol[1])
        if not (0.2 < abs(b1) < 5.0):
            return 0.0, 1.0
        return b0, b1
    except Exception:
        return 0.0, 1.0
