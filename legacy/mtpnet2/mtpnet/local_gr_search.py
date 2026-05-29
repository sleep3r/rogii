"""
local_gr_search.py  —  GR-guided TVT path decoder
==================================================
Greedy (multi-lookahead) and beam-search decoders over offset-state space.

Physical convention:
    dtvt[k] = -dz[k] + offset
    pred_tvt = anchor_tvt + cumsum(dtvt)

Both decoders accept:
    sample    – OffsetSample (.hidden_rows, .z, .gr, .anchor_tvt, .anchor_row)
    k_prior   – (K,) array of per-segment offset priors
    tw_tvt    – typewell TVT array
    tw_gr     – typewell GR array
    cfg       – LocalSearchConfig

and return:
    pred_tvt  – (n_hidden,) predicted TVT
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats  import iqr as scipy_iqr


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalSearchConfig:
    # Lookahead windows (rows). Multi-la: pick (la, offset) with min score.
    lookaheads: tuple = (300, 500, 800)

    # Fraction of lookahead window to commit
    commit_frac: float = 0.20

    # Offset candidate grid
    offset_lo: float = -0.16
    offset_hi: float =  0.16
    offset_n:  int   = 161

    # Hand-score weights
    lambda_offset: float = 40.0   # |offset - k_prior|
    lambda_smooth: float =  1.0   # |offset - prev_offset|

    # Beam-search
    beam_width:  int   = 16
    tvt_eps:     float =  1.0    # diversity: min |Δlast_tvt| to keep beam
    offset_eps:  float =  0.005  # diversity: min |Δprev_offset| to keep beam
    expand_k:    int   = 8       # top-k offsets to expand per beam state

    # GR smoothing (Savitzky-Golay)
    gr_smooth_window: int = 101
    gr_smooth_poly:   int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def smooth_gr(gr: np.ndarray, window: int = 101, poly: int = 3) -> np.ndarray:
    gr = gr.copy()
    # Replace non-finite values with linear interpolation / median fallback
    bad = ~np.isfinite(gr)
    if bad.any():
        idx = np.arange(len(gr))
        gr[bad] = np.interp(idx[bad], idx[~bad], gr[~bad]) if (~bad).any() else 0.0
    if len(gr) < window:
        return gr
    return savgol_filter(gr, window_length=window, polyorder=poly)


def _prior_at(pos: int, nh: int, k_prior: np.ndarray) -> float:
    K = len(k_prior)
    return float(k_prior[int(np.clip(pos * K // nh, 0, K - 1))])


def _score_one_window(
    z: np.ndarray,
    sgr: np.ndarray,
    anchor_row: int,
    hidden_rows: np.ndarray,
    pos: int,
    la: int,
    last_tvt: float,
    prev_offset: float,
    prior_offset: float,
    offset_grid: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cfg: LocalSearchConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Score all offset candidates for one (pos, la) window.

    Returns
    -------
    scores  : (NOFF,) lower = better
    tvc     : (NOFF, ns)  candidate TVT arrays
    nc      : commit length (rows)
    """
    hr  = hidden_rows
    nh  = len(hr)
    ep  = min(pos + la, nh)
    shi = hr[pos:ep]
    ns  = len(shi)
    pr  = hr[pos - 1] if pos > 0 else anchor_row

    dz   = np.diff(np.concatenate([[z[pr]], z[shi]]))
    dtvt = -dz[None, :] + offset_grid[:, None]          # (NOFF, ns)
    tvc  = last_tvt + np.cumsum(dtvt, axis=1)            # (NOFF, ns)

    grc  = np.interp(tvc.ravel(), tw_tvt, tw_gr).reshape(len(offset_grid), ns)
    gr_r = np.sqrt(np.mean((grc - sgr[shi][None, :]) ** 2, axis=1))
    med  = float(np.median(gr_r))
    iq   = float(scipy_iqr(gr_r))
    gr_z = (gr_r - med) / iq if iq > 1e-9 else np.zeros(len(offset_grid))

    scores = (
        gr_z
        + cfg.lambda_offset * np.abs(offset_grid - prior_offset)
        + cfg.lambda_smooth * np.abs(offset_grid - prev_offset)
    )
    nc = max(1, int(cfg.commit_frac * ns))
    return scores, tvc, nc


# ─────────────────────────────────────────────────────────────────────────────
# Greedy decoder (multi-lookahead)
# ─────────────────────────────────────────────────────────────────────────────

def run_greedy(
    sample,
    k_prior: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cfg: LocalSearchConfig,
) -> np.ndarray:
    """
    Multi-lookahead greedy. At each step picks (la, offset) with min score.
    Commits commit_frac * ns rows.
    """
    hr   = sample.hidden_rows
    nh   = len(hr)
    if nh == 0:
        return np.array([], dtype=np.float64)

    z    = sample.z.astype(np.float64)
    sgr  = smooth_gr(sample.gr.astype(np.float64),
                     cfg.gr_smooth_window, cfg.gr_smooth_poly)
    grid = np.linspace(cfg.offset_lo, cfg.offset_hi, cfg.offset_n)

    last     = float(sample.anchor_tvt)
    prev_off = float(k_prior[0])
    pred     = np.empty(nh, dtype=np.float64)
    pos      = 0

    while pos < nh:
        prior           = _prior_at(pos, nh, k_prior)
        best_score      = np.inf
        best_bj = best_nc = 0
        best_tvc        = None

        for la in cfg.lookaheads:
            ep = min(pos + la, nh)
            if ep == pos:
                continue
            sc, tvc, nc = _score_one_window(
                z, sgr, sample.anchor_row, hr,
                pos, la, last, prev_off, prior, grid,
                tw_tvt, tw_gr, cfg,
            )
            m = float(np.min(sc))
            if m < best_score:
                best_score = m
                best_bj    = int(np.argmin(sc))
                best_tvc   = tvc
                best_nc    = nc

        pred[pos:pos + best_nc] = best_tvc[best_bj, :best_nc]
        last     = float(best_tvc[best_bj, best_nc - 1])
        prev_off = float(grid[best_bj])
        pos     += best_nc

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Beam state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _BeamState:
    last_tvt:    float
    prev_offset: float
    total_cost:  float   # sum over steps of (min_score * nc)
    total_rows:  int
    pred:        np.ndarray  # accumulated prediction so far

    @property
    def score(self) -> float:
        return self.total_cost / max(self.total_rows, 1)


def _prune(beams: list[_BeamState], width: int,
           tvt_eps: float, offset_eps: float) -> list[_BeamState]:
    """Sort by score; keep up to `width` diverse beams."""
    beams_sorted = sorted(beams, key=lambda b: b.score)
    kept: list[_BeamState] = []
    for b in beams_sorted:
        if len(kept) >= width:
            break
        diverse = all(
            abs(b.last_tvt    - k.last_tvt)    > tvt_eps
            or abs(b.prev_offset - k.prev_offset) > offset_eps
            for k in kept
        )
        if not kept or diverse:
            kept.append(b)
    return kept or [beams_sorted[0]]


# ─────────────────────────────────────────────────────────────────────────────
# Beam decoder
# ─────────────────────────────────────────────────────────────────────────────

def run_beam(
    sample,
    k_prior: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cfg: LocalSearchConfig,
) -> np.ndarray:
    """
    Beam-search decoder.

    Invariant: all live beams are always at the same position `pos`.
    At each step:
      1. For each beam state × each lookahead: score all NOFF offsets.
      2. Pick best (la, offset) → commit nc rows (same nc for all).
      3. Expand: top-expand_k offsets from each beam's best la.
      4. Prune to beam_width with TVT/offset diversity.
    """
    hr   = sample.hidden_rows
    nh   = len(hr)
    if nh == 0:
        return np.array([], dtype=np.float64)

    z    = sample.z.astype(np.float64)
    sgr  = smooth_gr(sample.gr.astype(np.float64),
                     cfg.gr_smooth_window, cfg.gr_smooth_poly)
    grid = np.linspace(cfg.offset_lo, cfg.offset_hi, cfg.offset_n)

    # Initialise
    initial_pred = np.empty(nh, dtype=np.float64)
    beams: list[_BeamState] = [
        _BeamState(
            last_tvt    = float(sample.anchor_tvt),
            prev_offset = float(k_prior[0]),
            total_cost  = 0.0,
            total_rows  = 0,
            pred        = initial_pred.copy(),
        )
    ]
    pos = 0

    while pos < nh:
        prior    = _prior_at(pos, nh, k_prior)
        new_beams: list[_BeamState] = []
        commit_nc = None   # same for all beams this step

        for state in beams:
            # Find best lookahead for this state
            best_la_score = np.inf
            best_la_tvc   = None
            best_la_nc    = 1
            best_la_sc    = None

            for la in cfg.lookaheads:
                ep = min(pos + la, nh)
                if ep == pos:
                    continue
                sc, tvc, nc = _score_one_window(
                    z, sgr, sample.anchor_row, hr,
                    pos, la, state.last_tvt, state.prev_offset,
                    prior, grid, tw_tvt, tw_gr, cfg,
                )
                m = float(np.min(sc))
                if m < best_la_score:
                    best_la_score = m
                    best_la_tvc   = tvc
                    best_la_nc    = nc
                    best_la_sc    = sc

            # On first beam, fix commit_nc for this step
            if commit_nc is None:
                commit_nc = best_la_nc

            # Expand: try top expand_k offsets from this beam's best la
            top_js = np.argsort(best_la_sc)[: cfg.expand_k]
            for bj in top_js:
                chunk         = best_la_tvc[bj, :commit_nc]
                step_score    = float(best_la_sc[bj])
                new_pred      = state.pred.copy()
                new_pred[pos:pos + commit_nc] = chunk
                new_beams.append(_BeamState(
                    last_tvt    = float(chunk[-1]),
                    prev_offset = float(grid[bj]),
                    total_cost  = state.total_cost + step_score * commit_nc,
                    total_rows  = state.total_rows + commit_nc,
                    pred        = new_pred,
                ))

        beams = _prune(new_beams, cfg.beam_width, cfg.tvt_eps, cfg.offset_eps)
        pos  += commit_nc

    return beams[0].pred
