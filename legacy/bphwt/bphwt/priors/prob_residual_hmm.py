"""Standalone Bayesian residual HMM for typewell-based TVT paths.

This module deliberately does not depend on the existing BPHWT cache, NN code,
or older absolute-TVT HMM prior. It is a small per-well probabilistic path model:

    TVT_t = base_t + residual_t

where ``base_t`` is built only from visible ``TVT_input`` anchors and the HMM
infers a smooth residual path from GR/typewell agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

_NEG_INF = -1.0e300
_EPS = 1.0e-12


@dataclass(frozen=True)
class ProbResidualHMMConfig:
    row_stride: int = 5
    residual_range_ft: float = 80.0
    residual_step_ft: float = 1.0

    sigma_gr_z: float = 0.75
    sigma_gr_raw: float = 15.0
    student_nu: float = 4.0

    sigma_rw: float = 1.25
    sigma_base: float = 35.0
    sigma_anchor: float = 0.75
    transition_band: int | None = None

    use_zscore_gr: bool = True
    use_calibrated_gr_if_possible: bool = True
    zscore_weight: float = 1.0
    calibrated_weight: float = 1.0

    min_gr_valid_ratio: float = 0.10
    min_zscore_valid: int = 5
    min_calibration_points: int = 30
    beta1_min_abs: float = 0.2
    beta1_max_abs: float = 5.0

    blend_alpha: float = 0.70
    posterior_std_gate: float = 20.0


@dataclass(frozen=True)
class CalibrationResult:
    enabled: bool
    beta0: float = 0.0
    beta1: float = 1.0
    sigma: float = 1.0
    n_points: int = 0


def make_base_tvt(tvt_input: np.ndarray, md: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear/last-known TVT baseline from visible anchors only."""
    tvt = np.asarray(tvt_input, dtype=np.float64)
    n = int(tvt.size)
    known = np.isfinite(tvt)
    if n == 0:
        return np.zeros(0, dtype=np.float64), known

    if md is None:
        x = np.arange(n, dtype=np.float64)
    else:
        x = np.asarray(md, dtype=np.float64)
        if x.shape != tvt.shape or not np.isfinite(x).all():
            x = np.arange(n, dtype=np.float64)

    if not known.any():
        return np.zeros(n, dtype=np.float64), known

    base = np.interp(x, x[known], tvt[known]).astype(np.float64)
    base[known] = tvt[known]
    return base, known


def robust_zscore(
    x: np.ndarray,
    mask: np.ndarray | None = None,
    min_valid: int = 5,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, bool, float, float]:
    """Median/IQR z-score with std fallback.

    Invalid positions are returned as zero so callers can safely keep using a
    separate validity mask.
    """
    arr = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    if int(valid.sum()) < int(min_valid):
        return np.zeros_like(arr, dtype=np.float64), False, 0.0, 1.0

    vals = arr[valid]
    center = float(np.median(vals))
    q25, q75 = np.percentile(vals, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.std(vals))
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(arr, dtype=np.float64), False, center, 1.0

    z = (arr - center) / scale
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float64), True, center, scale


def make_stride_index(n: int, stride: int, known_mask: np.ndarray | None = None) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    step = max(1, int(stride))
    idx = set(range(0, int(n), step))
    idx.add(int(n) - 1)
    if known_mask is not None:
        known = np.asarray(known_mask, dtype=bool)
        idx.update(int(i) for i in np.flatnonzero(known))
    return np.array(sorted(idx), dtype=np.int64)


def make_residual_grid(cfg: ProbResidualHMMConfig) -> np.ndarray:
    step = float(cfg.residual_step_ft)
    radius = float(cfg.residual_range_ft)
    if step <= 0.0:
        raise ValueError("residual_step_ft must be positive")
    if radius < 0.0:
        raise ValueError("residual_range_ft must be non-negative")
    return np.arange(-radius, radius + 0.5 * step, step, dtype=np.float64)


def build_log_obs_residual_grid(
    base_s: np.ndarray,
    gr_s: np.ndarray,
    gr_valid_s: np.ndarray,
    tvt_input_s: np.ndarray,
    known_s: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    r_grid: np.ndarray,
    cfg: ProbResidualHMMConfig | dict[str, Any] | None = None,
    calibration: CalibrationResult | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build log p(observation_t | residual_state_j)."""
    c = _coerce_config(cfg)
    base = np.asarray(base_s, dtype=np.float64)
    gr = np.asarray(gr_s, dtype=np.float64)
    gr_valid = np.asarray(gr_valid_s, dtype=bool) & np.isfinite(gr)
    tvt_input = np.asarray(tvt_input_s, dtype=np.float64)
    known = np.asarray(known_s, dtype=bool) & np.isfinite(tvt_input)
    residuals = np.asarray(r_grid, dtype=np.float64)
    tw_x, tw_y = _prepare_typewell(tw_tvt, tw_gr)

    tvt_grid = base[:, None] + residuals[None, :]
    tw_gr_at_grid = np.interp(tvt_grid, tw_x, tw_y)
    log_obs = np.zeros((base.size, residuals.size), dtype=np.float64)
    info: dict[str, Any] = {
        "zscore_enabled": False,
        "calibrated_enabled": False,
        "gr_valid_count": int(gr_valid.sum()),
    }

    if c.use_zscore_gr and int(gr_valid.sum()) >= c.min_zscore_valid:
        gr_z, gr_ok, _, _ = robust_zscore(gr, gr_valid, min_valid=c.min_zscore_valid)
        tw_z, tw_ok, _, _ = robust_zscore(tw_y, min_valid=c.min_zscore_valid)
        if gr_ok and tw_ok:
            tw_z_at_grid = np.interp(tvt_grid, tw_x, tw_z)
            ll = student_t_logpdf(gr_z[:, None] - tw_z_at_grid, sigma=c.sigma_gr_z, nu=c.student_nu)
            log_obs[gr_valid] += float(c.zscore_weight) * ll[gr_valid]
            info["zscore_enabled"] = True

    if calibration is not None and calibration.enabled:
        pred_gr = calibration.beta0 + calibration.beta1 * tw_gr_at_grid
        ll = student_t_logpdf(gr[:, None] - pred_gr, sigma=calibration.sigma, nu=c.student_nu)
        log_obs[gr_valid] += float(c.calibrated_weight) * ll[gr_valid]
        info["calibrated_enabled"] = True
        info["calibration_beta0"] = calibration.beta0
        info["calibration_beta1"] = calibration.beta1
        info["calibration_sigma"] = calibration.sigma
        info["calibration_n_points"] = calibration.n_points

    if c.sigma_anchor > 0.0 and known.any():
        diff = tvt_grid[known] - tvt_input[known, None]
        log_obs[known] += -0.5 * (diff / float(c.sigma_anchor)) ** 2

    if c.sigma_base > 0.0:
        log_obs += -0.5 * (residuals[None, :] / float(c.sigma_base)) ** 2

    log_obs = np.nan_to_num(log_obs, nan=0.0, posinf=0.0, neginf=_NEG_INF)
    return log_obs, info


def fit_calibration(
    gr: np.ndarray,
    tvt_input: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cfg: ProbResidualHMMConfig | dict[str, Any] | None = None,
) -> CalibrationResult:
    c = _coerce_config(cfg)
    if not c.use_calibrated_gr_if_possible:
        return CalibrationResult(enabled=False)

    gr_arr = np.asarray(gr, dtype=np.float64)
    tvt_arr = np.asarray(tvt_input, dtype=np.float64)
    valid = np.isfinite(gr_arr) & np.isfinite(tvt_arr)
    n_points = int(valid.sum())
    if n_points < int(c.min_calibration_points):
        return CalibrationResult(enabled=False, n_points=n_points)

    tw_x, tw_y = _prepare_typewell(tw_tvt, tw_gr)
    x = np.interp(tvt_arr[valid], tw_x, tw_y)
    y = gr_arr[valid]
    design = np.column_stack([x, np.ones_like(x)])
    try:
        beta1, beta0 = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return CalibrationResult(enabled=False, n_points=n_points)

    beta1 = float(beta1)
    beta0 = float(beta0)
    if not (c.beta1_min_abs <= abs(beta1) <= c.beta1_max_abs):
        return CalibrationResult(enabled=False, beta0=beta0, beta1=beta1, n_points=n_points)

    resid = y - (beta0 + beta1 * x)
    _, sigma_ok, _, sigma_est = robust_zscore(resid, min_valid=max(3, min(c.min_calibration_points, n_points)))
    sigma = float(c.sigma_gr_raw) if c.sigma_gr_raw > 0.0 else sigma_est
    if not sigma_ok and c.sigma_gr_raw <= 0.0:
        sigma = 1.0
    sigma = max(float(sigma), 1.0e-3)
    return CalibrationResult(enabled=True, beta0=beta0, beta1=beta1, sigma=sigma, n_points=n_points)


def student_t_logpdf(residual: np.ndarray, sigma: float, nu: float) -> np.ndarray:
    s = max(float(sigma), 1.0e-6)
    v = max(float(nu), 1.0e-6)
    z = np.asarray(residual, dtype=np.float64) / s
    return -0.5 * (v + 1.0) * np.log1p((z * z) / v) - np.log(s)


def forward_backward_banded(
    log_obs: np.ndarray,
    r_grid: np.ndarray,
    sigma_rw: float,
    band: int | None = None,
) -> dict[str, np.ndarray | float]:
    obs = _clean_log_obs(log_obs)
    residuals = np.asarray(r_grid, dtype=np.float64)
    if obs.ndim != 2:
        raise ValueError("log_obs must have shape [L, S]")
    if obs.shape[1] != residuals.size:
        raise ValueError("r_grid length must match log_obs.shape[1]")
    L, S = obs.shape
    if L == 0 or S == 0:
        raise ValueError("log_obs must be non-empty")

    offsets, costs = _transition_offsets(residuals, sigma_rw, band)

    alpha = np.full((L, S), _NEG_INF, dtype=np.float64)
    alpha[0] = obs[0]
    for t in range(1, L):
        carried = np.full(S, _NEG_INF, dtype=np.float64)
        prev = alpha[t - 1]
        for offset, cost in zip(offsets, costs, strict=True):
            src, dst = _offset_slices(S, int(offset))
            carried[dst] = np.logaddexp(carried[dst], prev[src] + cost)
        alpha[t] = carried + obs[t]

    log_evidence = float(logsumexp(alpha[-1]))

    beta = np.full((L, S), _NEG_INF, dtype=np.float64)
    beta[-1] = 0.0
    for t in range(L - 2, -1, -1):
        carried = np.full(S, _NEG_INF, dtype=np.float64)
        nxt = beta[t + 1] + obs[t + 1]
        for offset, cost in zip(offsets, costs, strict=True):
            src, dst = _offset_slices(S, int(offset))
            carried[src] = np.logaddexp(carried[src], nxt[dst] + cost)
        beta[t] = carried

    log_post = alpha + beta - log_evidence
    posterior = np.exp(np.clip(log_post, -745.0, 0.0))
    row_sum = posterior.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0.0
    if bad.any():
        posterior[bad] = 1.0 / float(S)
        row_sum = posterior.sum(axis=1, keepdims=True)
    posterior /= row_sum

    mean = posterior @ residuals
    second = posterior @ (residuals * residuals)
    std = np.sqrt(np.clip(second - mean * mean, 0.0, None))

    return {
        "alpha": alpha,
        "beta": beta,
        "posterior": posterior,
        "posterior_mean": mean.astype(np.float64),
        "posterior_std": std.astype(np.float64),
        "log_evidence": log_evidence,
    }


def viterbi_banded(
    log_obs: np.ndarray,
    r_grid: np.ndarray,
    sigma_rw: float,
    band: int | None = None,
) -> tuple[np.ndarray, float]:
    obs = _clean_log_obs(log_obs)
    residuals = np.asarray(r_grid, dtype=np.float64)
    L, S = obs.shape
    offsets, costs = _transition_offsets(residuals, sigma_rw, band)

    delta = np.full((L, S), _NEG_INF, dtype=np.float64)
    psi = np.zeros((L, S), dtype=np.int32)
    delta[0] = obs[0]

    for t in range(1, L):
        best = np.full(S, _NEG_INF, dtype=np.float64)
        best_from = np.zeros(S, dtype=np.int32)
        prev = delta[t - 1]
        for offset, cost in zip(offsets, costs, strict=True):
            src, dst = _offset_slices(S, int(offset))
            src_idx, dst_idx = _offset_indices(S, int(offset))
            vals = prev[src] + cost
            improve = vals > best[dst]
            if improve.any():
                best[dst_idx[improve]] = vals[improve]
                best_from[dst_idx[improve]] = src_idx[improve]
        delta[t] = best + obs[t]
        psi[t] = best_from

    state = int(np.argmax(delta[-1]))
    path_idx = np.zeros(L, dtype=np.int32)
    path_idx[-1] = state
    for t in range(L - 2, -1, -1):
        path_idx[t] = psi[t + 1, path_idx[t + 1]]

    return residuals[path_idx].astype(np.float64), float(delta[-1, state])


def infer_residual_hmm(
    md: np.ndarray,
    gr: np.ndarray,
    tvt_input: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cfg: ProbResidualHMMConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    c = _coerce_config(cfg)
    md_arr = np.asarray(md, dtype=np.float64)
    gr_arr = np.asarray(gr, dtype=np.float64)
    tvt_arr = np.asarray(tvt_input, dtype=np.float64)
    tw_x, tw_y = _prepare_typewell(tw_tvt, tw_gr)
    if not (md_arr.shape == gr_arr.shape == tvt_arr.shape):
        raise ValueError("md, gr, and tvt_input must have the same shape")

    base, known = make_base_tvt(tvt_arr, md_arr)
    if not known.any():
        base = np.full_like(base, float(np.median(tw_x)), dtype=np.float64)

    idx = make_stride_index(len(md_arr), c.row_stride, include_known_anchor_rows(known))
    r_grid = make_residual_grid(c)
    calibration = fit_calibration(gr_arr, tvt_arr, tw_x, tw_y, c)
    gr_valid = np.isfinite(gr_arr)

    log_obs, obs_info = build_log_obs_residual_grid(
        base[idx],
        gr_arr[idx],
        gr_valid[idx],
        tvt_arr[idx],
        known[idx],
        tw_x,
        tw_y,
        r_grid,
        c,
        calibration=calibration,
    )
    fb = forward_backward_banded(log_obs, r_grid, sigma_rw=c.sigma_rw, band=c.transition_band)
    viterbi_r_s, viterbi_loglik = viterbi_banded(log_obs, r_grid, sigma_rw=c.sigma_rw, band=c.transition_band)
    r_mean_s = np.asarray(fb["posterior_mean"], dtype=np.float64)

    axis = _interp_axis(md_arr)
    r_mean = _interp_strided(axis, idx, r_mean_s)
    r_std = _interp_strided(axis, idx, np.asarray(fb["posterior_std"], dtype=np.float64))
    r_viterbi = _interp_strided(axis, idx, viterbi_r_s)

    pred_mean = base + r_mean
    pred_viterbi = base + r_viterbi
    posterior_std = r_std
    if known.any():
        pred_mean[known] = tvt_arr[known]
        pred_viterbi[known] = tvt_arr[known]
        posterior_std[known] = 0.0

    ll_base = _path_log_score(np.zeros(log_obs.shape[0], dtype=np.float64), log_obs, r_grid, c.sigma_rw)
    mean_path_loglik = _path_log_score(r_mean_s, log_obs, r_grid, c.sigma_rw)
    ll_gain_vs_base = float(mean_path_loglik - ll_base)
    gr_valid_ratio = float(gr_valid.mean()) if gr_valid.size else 0.0
    hidden = ~known
    confidence = confidence_from(posterior_std, gr_valid_ratio, ll_gain_vs_base, hidden, c)
    pred_blend = blend_predictions(base, pred_mean, alpha=c.blend_alpha, confidence=confidence, tvt_input=tvt_arr)

    return {
        "pred_base": base.astype(np.float64),
        "pred_mean": pred_mean.astype(np.float64),
        "pred_viterbi": pred_viterbi.astype(np.float64),
        "pred_blend": pred_blend.astype(np.float64),
        "posterior_std": posterior_std.astype(np.float64),
        "log_evidence": float(fb["log_evidence"]),
        "ll_base": ll_base,
        "ll_gain_vs_base": ll_gain_vs_base,
        "mean_path_loglik": float(mean_path_loglik),
        "viterbi_loglik": float(viterbi_loglik),
        "gr_valid_ratio": gr_valid_ratio,
        "confidence": float(confidence),
        "stride_index": idx,
        "r_grid": r_grid,
        "calibration": calibration,
        "obs_info": obs_info,
    }


def confidence_from(
    posterior_std: np.ndarray,
    gr_valid_ratio: float,
    ll_gain_vs_base: float,
    hidden_mask: np.ndarray | None = None,
    cfg: ProbResidualHMMConfig | dict[str, Any] | None = None,
) -> float:
    c = _coerce_config(cfg)
    conf = 1.0
    if float(gr_valid_ratio) < float(c.min_gr_valid_ratio):
        conf = 0.0

    std = np.asarray(posterior_std, dtype=np.float64)
    mask = np.isfinite(std)
    if hidden_mask is not None:
        mask &= np.asarray(hidden_mask, dtype=bool)
    std_mean = float(np.mean(std[mask])) if mask.any() else float("inf")
    if std_mean > float(c.posterior_std_gate):
        conf *= 0.3
    if float(ll_gain_vs_base) < 0.0:
        conf *= 0.3
    return float(conf)


def blend_predictions(
    base: np.ndarray,
    hmm_mean: np.ndarray,
    alpha: float,
    confidence: float,
    tvt_input: np.ndarray | None = None,
) -> np.ndarray:
    base_arr = np.asarray(base, dtype=np.float64)
    hmm_arr = np.asarray(hmm_mean, dtype=np.float64)
    pred = base_arr + float(alpha) * float(confidence) * (hmm_arr - base_arr)
    if tvt_input is not None:
        tvt = np.asarray(tvt_input, dtype=np.float64)
        known = np.isfinite(tvt)
        pred[known] = tvt[known]
    return pred.astype(np.float64)


def include_known_anchor_rows(known_mask: np.ndarray) -> np.ndarray:
    return np.asarray(known_mask, dtype=bool)


def _coerce_config(cfg: ProbResidualHMMConfig | dict[str, Any] | None) -> ProbResidualHMMConfig:
    if cfg is None:
        return ProbResidualHMMConfig()
    if isinstance(cfg, ProbResidualHMMConfig):
        return cfg
    if isinstance(cfg, dict):
        return ProbResidualHMMConfig(**cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def _prepare_typewell(tw_tvt: np.ndarray, tw_gr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tvt = np.asarray(tw_tvt, dtype=np.float64)
    gr = np.asarray(tw_gr, dtype=np.float64)
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if int(valid.sum()) < 2:
        raise ValueError("typewell must contain at least two finite TVT/GR rows")
    tvt = tvt[valid]
    gr = gr[valid]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    unique_tvt, unique_idx = np.unique(tvt, return_index=True)
    unique_gr = gr[unique_idx]
    if unique_tvt.size < 2:
        raise ValueError("typewell TVT grid must contain at least two unique finite values")
    return unique_tvt.astype(np.float64), unique_gr.astype(np.float64)


def _clean_log_obs(log_obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(log_obs, dtype=np.float64)
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=_NEG_INF)


def _transition_offsets(
    r_grid: np.ndarray,
    sigma_rw: float,
    band: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(r_grid, dtype=np.float64)
    if residuals.size == 1:
        return np.array([0], dtype=np.int64), np.array([0.0], dtype=np.float64)
    if float(sigma_rw) <= 0.0:
        raise ValueError("sigma_rw must be positive")
    step = float(np.median(np.diff(residuals)))
    if step <= 0.0:
        raise ValueError("r_grid must be strictly increasing")
    if band is None:
        width = int(np.ceil(5.0 * float(sigma_rw) / step))
    else:
        width = int(band)
    width = max(0, min(width, residuals.size - 1))
    offsets = np.arange(-width, width + 1, dtype=np.int64)
    deltas = offsets.astype(np.float64) * step
    costs = -0.5 * (deltas / float(sigma_rw)) ** 2
    return offsets, costs.astype(np.float64)


def _offset_slices(S: int, offset: int) -> tuple[slice, slice]:
    if offset >= 0:
        return slice(0, S - offset), slice(offset, S)
    return slice(-offset, S), slice(0, S + offset)


def _offset_indices(S: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    if offset >= 0:
        return np.arange(0, S - offset, dtype=np.int32), np.arange(offset, S, dtype=np.int32)
    return np.arange(-offset, S, dtype=np.int32), np.arange(0, S + offset, dtype=np.int32)


def _path_log_score(r_path: np.ndarray, log_obs: np.ndarray, r_grid: np.ndarray, sigma_rw: float) -> float:
    path = np.asarray(r_path, dtype=np.float64)
    obs = _clean_log_obs(log_obs)
    residuals = np.asarray(r_grid, dtype=np.float64)
    if path.size != obs.shape[0]:
        raise ValueError("r_path length must match log_obs.shape[0]")
    state_idx = np.abs(path[:, None] - residuals[None, :]).argmin(axis=1)
    score = float(obs[np.arange(path.size), state_idx].sum())
    if path.size > 1:
        dr = np.diff(residuals[state_idx])
        score += float((-0.5 * (dr / float(sigma_rw)) ** 2).sum())
    return score


def _interp_axis(md: np.ndarray) -> np.ndarray:
    arr = np.asarray(md, dtype=np.float64)
    if arr.size > 1 and np.isfinite(arr).all() and np.all(np.diff(arr) > 0.0):
        return arr
    return np.arange(arr.size, dtype=np.float64)


def _interp_strided(axis: np.ndarray, idx: np.ndarray, values_s: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return np.zeros(axis.size, dtype=np.float64)
    if idx.size == 1:
        return np.full(axis.size, float(values_s[0]), dtype=np.float64)
    xp = np.asarray(axis, dtype=np.float64)[idx]
    return np.interp(axis, xp, np.asarray(values_s, dtype=np.float64)).astype(np.float64)


__all__ = [
    "CalibrationResult",
    "ProbResidualHMMConfig",
    "blend_predictions",
    "build_log_obs_residual_grid",
    "confidence_from",
    "fit_calibration",
    "forward_backward_banded",
    "infer_residual_hmm",
    "make_base_tvt",
    "make_residual_grid",
    "make_stride_index",
    "robust_zscore",
    "student_t_logpdf",
    "viterbi_banded",
]
