"""HMM typewell alignment prior.

Runs a banded Viterbi + forward-backward over a TVT bin grid to compute:
    hmm_viterbi    — MAP TVT path
    hmm_tvt_mu     — posterior mean TVT
    hmm_tvt_std    — posterior std TVT
    hmm_velocity   — smoothed dTVT/dMD along Viterbi path
    hmm_entropy    — per-step state entropy (bits)
    hmm_loglik     — total log-likelihood of the MAP path

Key design choices
------------------
* Non-monotonic: TVT can go up, down, or stay flat (dip-up, dip-down, geosteering).
  Standard DTW assumes monotonic; this HMM does not.
* Banded transitions: only ±band bins per step are allowed (controls max velocity).
* Known TVT_input anchors become hard observation potentials (log p = +100 if match).
* GR observation model: N(beta1 * typewell_GR(tvt_bin) + beta0, sigma_gr).
  beta0, beta1 are estimated per-well via LS on observed GR rows.
* If GR NaN ratio > gr_bad_thresh, fall back to linear-anchor-only prior.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

_EPS = 1e-8
_NEG_INF = -1e30


def run_hmm(
    gr_obs: np.ndarray,  # [L] float, NaN where invalid
    tw_tvt: np.ndarray,  # [M] typewell TVT grid
    tw_gr: np.ndarray,  # [M] typewell GR
    anchor_tvt: np.ndarray,  # [L] float, NaN where hidden
    md: np.ndarray,  # [L] measured depth
    tvt_step: float = 2.0,
    tvt_margin: float = 100.0,
    velocity_sigma: float = 0.025,
    gr_sigma: float = 15.0,
    max_velocity: float = 0.12,
    gr_bad_thresh: float = 0.90,
    anchor_log_weight: float = 80.0,
) -> dict[str, np.ndarray]:
    """
    Run HMM Viterbi + forward-backward for one well.

    Returns dict with:
        hmm_viterbi, hmm_tvt_mu, hmm_tvt_std, hmm_velocity,
        hmm_entropy, hmm_loglik
    All arrays are float32 of length L.
    """
    L = len(gr_obs)
    gr_valid = np.isfinite(gr_obs)
    gr_valid_ratio = gr_valid.mean()
    gr_nan_ratio = 1.0 - gr_valid_ratio
    anchor_known = np.isfinite(anchor_tvt)

    # ------------------------------------------------------------------
    # 1. Determine TVT grid
    # ------------------------------------------------------------------
    anchor_vals = anchor_tvt[anchor_known]
    if len(anchor_vals) == 0:
        tvt_center = 11000.0  # fallback
    else:
        tvt_center = anchor_vals.mean()

    # Extend range to cover all anchor values + margin
    if len(anchor_vals) > 0:
        tvt_lo = anchor_vals.min() - tvt_margin
        tvt_hi = anchor_vals.max() + (L * max_velocity * 1.5) + tvt_margin
    else:
        tvt_lo = tvt_center - tvt_margin
        tvt_hi = tvt_center + L * max_velocity * 2.0 + tvt_margin

    # Also ensure typewell coverage
    tvt_lo = max(tvt_lo, tw_tvt.min())
    tvt_hi = min(tvt_hi, tw_tvt.max())
    if tvt_hi <= tvt_lo:
        tvt_hi = tvt_lo + 200.0

    tvt_grid = np.arange(tvt_lo, tvt_hi + tvt_step / 2, tvt_step, dtype=np.float64)
    S = len(tvt_grid)
    if S < 2:
        return _fallback_prior(anchor_tvt, L)

    # ------------------------------------------------------------------
    # 2. Precompute typewell GR at each bin
    # ------------------------------------------------------------------
    tw_gr_at_bins = np.interp(tvt_grid, tw_tvt, tw_gr).astype(np.float64)  # [S]

    # ------------------------------------------------------------------
    # 3. Estimate beta0, beta1 via LS on observed GR rows
    # ------------------------------------------------------------------
    beta0, beta1 = 0.0, 1.0
    if gr_valid_ratio > 0.05 and gr_nan_ratio < gr_bad_thresh:
        gr_obs_valid = gr_obs[gr_valid]
        # Use linear anchor TVT to get initial bin estimate
        lin_tvt = _linear_interp(anchor_tvt)
        lin_tvt_valid = lin_tvt[gr_valid]
        tw_gr_lin = np.interp(lin_tvt_valid, tw_tvt, tw_gr)
        # LS: gr_obs = beta1 * tw_gr + beta0
        A = np.stack([tw_gr_lin, np.ones_like(tw_gr_lin)], axis=1)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, gr_obs_valid, rcond=None)
            beta1_est, beta0_est = coeffs[0], coeffs[1]
            # Sanity check
            if 0.3 < abs(beta1_est) < 3.0:
                beta1, beta0 = beta1_est, beta0_est
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 4. Build log-observation matrix log_obs[t, s]
    # ------------------------------------------------------------------
    log_obs = np.zeros((L, S), dtype=np.float64)

    if gr_valid_ratio > 0.05 and gr_nan_ratio < gr_bad_thresh:
        # Smoothed GR for more robust comparison
        gr_filled = _fill_gr(gr_obs)
        gr_smooth = gaussian_filter1d(gr_filled, sigma=5.0)

        # Predicted GR for each bin: beta1 * tw_gr_at_bin + beta0
        predicted_gr = beta1 * tw_gr_at_bins + beta0  # [S]
        predicted_gr_smooth = gaussian_filter1d(predicted_gr, sigma=5.0)

        for t in range(L):
            if gr_valid[t]:
                diff = gr_smooth[t] - predicted_gr_smooth  # [S]
                log_obs[t] = -0.5 * (diff / gr_sigma) ** 2

    # Anchor potentials: hard likelihood at known TVT_input rows
    for t in range(L):
        if anchor_known[t]:
            bin_idx = int(np.round((anchor_tvt[t] - tvt_lo) / tvt_step))
            bin_idx = np.clip(bin_idx, 0, S - 1)
            # Soft Gaussian anchor potential (width = 1 bin)
            bin_range = np.arange(S)
            log_obs[t] += anchor_log_weight * np.exp(-0.5 * ((bin_range - bin_idx) / 2.0) ** 2)

    # ------------------------------------------------------------------
    # 5. Banded Viterbi
    # ------------------------------------------------------------------
    dmd = np.diff(md, prepend=md[0])
    dmd = np.where(np.abs(dmd) < _EPS, 1.0, dmd)
    band = max(1, int(np.ceil(max_velocity * np.median(np.abs(dmd)) / tvt_step))) + 1
    band = min(band, 8)

    # Transition cost for each offset ds: -0.5 * (ds * tvt_step / (velocity_sigma * dmd))^2
    ds_range = np.arange(-band, band + 1, dtype=np.float64)  # [2B+1]

    delta = np.full((L, S), _NEG_INF, dtype=np.float64)
    psi = np.zeros((L, S), dtype=np.int32)

    # Initialise: from known anchor or uniform over typewell range
    if anchor_known[0]:
        init_bin = int(np.round((anchor_tvt[0] - tvt_lo) / tvt_step))
        init_bin = np.clip(init_bin, 0, S - 1)
        init_log = -0.5 * ((np.arange(S) - init_bin) / 3.0) ** 2 * anchor_log_weight / 4
    else:
        init_log = np.zeros(S, dtype=np.float64)  # uniform
    delta[0] = init_log + log_obs[0]

    for t in range(1, L):
        step = abs(dmd[t])
        trans_sigma = velocity_sigma * step
        trans_costs = -0.5 * (ds_range * tvt_step / (trans_sigma + _EPS)) ** 2  # [2B+1]

        # For each state s, find best previous state in [s-band, s+band]
        best = np.full(S, _NEG_INF, dtype=np.float64)
        best_from = np.zeros(S, dtype=np.int32)

        for k, ds in enumerate(range(-band, band + 1)):
            tc = trans_costs[k]
            if ds >= 0:
                # state s comes from state s - ds (s_from = s - ds)
                s_to = np.arange(ds, S)
                s_from = np.arange(0, S - ds)
                val = delta[t - 1, s_from] + tc
                improve = val > best[s_to]
                best[s_to[improve]] = val[improve]
                best_from[s_to[improve]] = s_from[improve]
            else:
                # ds < 0 => s_to = s_from + |ds|
                s_from = np.arange(-ds, S)
                s_to = np.arange(0, S + ds)
                val = delta[t - 1, s_from] + tc
                improve = val > best[s_to]
                best[s_to[improve]] = val[improve]
                best_from[s_to[improve]] = s_from[improve]

        delta[t] = best + log_obs[t]
        psi[t] = best_from

    # Backtrack
    path = np.zeros(L, dtype=np.int32)
    path[L - 1] = int(np.argmax(delta[L - 1]))
    for t in range(L - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    viterbi_tvt = tvt_grid[path].astype(np.float32)
    hmm_loglik = float(delta[L - 1, path[L - 1]])
    viterbi_gr = beta1 * np.interp(viterbi_tvt, tw_tvt, tw_gr).astype(np.float64) + beta0
    mismatch = np.zeros(L, dtype=np.float32)
    if gr_valid.any():
        mismatch[gr_valid] = np.abs(viterbi_gr[gr_valid] - gr_obs[gr_valid]).astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Forward-backward for posterior mean/std
    # ------------------------------------------------------------------
    # Forward pass (already computed as delta after removing obs)
    # For efficiency, use the delta as a proxy for alpha (log-scale)
    alpha = delta.copy()  # approximate; re-use Viterbi delta

    # Backward pass
    beta_fb = np.zeros((L, S), dtype=np.float64)
    beta_fb[L - 1] = 0.0  # log(1)

    for t in range(L - 2, -1, -1):
        step = abs(dmd[t + 1])
        trans_sigma = velocity_sigma * step
        trans_costs = -0.5 * (ds_range * tvt_step / (trans_sigma + _EPS)) ** 2

        for k, ds in enumerate(range(-band, band + 1)):
            tc = trans_costs[k]
            if ds >= 0:
                s_to_arr = np.arange(ds, S)
                s_from_arr = np.arange(0, S - ds)
                contrib = beta_fb[t + 1, s_to_arr] + log_obs[t + 1, s_to_arr] + tc
                beta_fb[t, s_from_arr] = np.logaddexp(beta_fb[t, s_from_arr], contrib)
            else:
                s_from_arr = np.arange(-ds, S)
                s_to_arr = np.arange(0, S + ds)
                contrib = beta_fb[t + 1, s_to_arr] + log_obs[t + 1, s_to_arr] + tc
                beta_fb[t, s_from_arr] = np.logaddexp(beta_fb[t, s_from_arr], contrib)

    # Posterior (log scale)
    log_posterior = alpha + beta_fb  # [L, S]
    # Normalise each row
    log_z = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
    log_z = np.where(np.isinf(log_z), 0.0, log_z)
    log_posterior = log_posterior - log_z
    posterior = np.exp(np.clip(log_posterior, -500, 0))  # [L, S]
    posterior = posterior / (posterior.sum(axis=1, keepdims=True) + _EPS)

    tvt_grid_f = tvt_grid.astype(np.float32)
    mu = (posterior @ tvt_grid_f).astype(np.float32)
    var = (posterior @ (tvt_grid_f**2)) - mu**2
    std = np.sqrt(np.clip(var, 0, None)).astype(np.float32)

    # Entropy
    entropy = -(posterior * np.log(posterior + _EPS)).sum(axis=1).astype(np.float32)

    # Velocity along Viterbi path
    velocity = np.gradient(viterbi_tvt.astype(np.float64), md).astype(np.float32)

    return {
        "hmm_viterbi": viterbi_tvt,
        "hmm_tvt_mu": mu,
        "hmm_tvt_std": std,
        "hmm_velocity": velocity,
        "hmm_entropy": entropy,
        "hmm_loglik": np.full(L, float(hmm_loglik), dtype=np.float32),
        "hmm_gr_mismatch": mismatch,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_interp(tvt_input: np.ndarray) -> np.ndarray:
    n = len(tvt_input)
    known = np.where(np.isfinite(tvt_input))[0]
    if len(known) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.interp(np.arange(n), known, tvt_input[known]).astype(np.float32)


def _fill_gr(gr: np.ndarray) -> np.ndarray:
    gr = gr.copy()
    valid = np.isfinite(gr)
    if valid.sum() == 0:
        return np.zeros_like(gr)
    return np.interp(np.arange(len(gr)), np.where(valid)[0], gr[valid]).astype(np.float32)


def _fallback_prior(anchor_tvt: np.ndarray, L: int) -> dict[str, np.ndarray]:
    """Return linear-prior-only features when HMM can't run."""
    lin = _linear_interp(anchor_tvt)
    return {
        "hmm_viterbi": lin,
        "hmm_tvt_mu": lin,
        "hmm_tvt_std": np.full(L, 20.0, dtype=np.float32),
        "hmm_velocity": np.zeros(L, dtype=np.float32),
        "hmm_entropy": np.full(L, 3.0, dtype=np.float32),
        "hmm_loglik": np.full(L, -999.0, dtype=np.float32),
        "hmm_gr_mismatch": np.full(L, 999.0, dtype=np.float32),
    }
