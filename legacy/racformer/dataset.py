"""RAC-Former v1 dataset and feature engineering (No-Prior Anchored C-Former).

Key design principles:
  - C = TVT + Z is the core structural field (nearly constant along track)
  - base_tvt = anchor_tvt - (Z - Z_anchor) + c0 * (i - anchor_row)    [no external priors]
  - All features are engineered at step resolution (ROWS_PER_STEP raw rows → 1 token)
  - Segment oracle s_star computed via ridge-smoothed LS for training supervision
  - No leakage: only test-safe features (MD, X, Y, Z, GR, TVT_input)

Feature layout (N_RAW_FEATURES = 42):
  [0]  step_frac                  — step_idx / (T-1)
  [1]  rel_step_from_anchor       — (step_idx - anchor_step) / 100
  [2]  rel_row_start_from_anchor  — (row_start - anchor_row) / 5000
  [3]  rel_row_end_from_anchor    — (row_end - anchor_row) / 5000
  [4]  MD_mean / 10000
  [5]  MD_std / 100
  [6]  MD_slope / 100
  [7]  X_mean / 10000
  [8]  X_std / 100
  [9]  X_slope / 100
  [10] Y_mean / 10000
  [11] Y_std / 100
  [12] Y_slope / 100
  [13] Z_mean / 10000
  [14] Z_std / 100
  [15] Z_slope / 1.0              — ft/row mean dZ
  [16] dZ_mean / 1.0
  [17] dZ_std / 0.5
  [18] dZ_min / 2.0
  [19] dZ_max / 2.0
  [20] dMD_mean / 1.0
  [21] dXY_mean / 1.0
  [22] GR_mean / 100
  [23] GR_std / 50
  [24] GR_min / 100
  [25] GR_max / 100
  [26] GR_valid_frac
  [27] dGR_mean / 50
  [28] dGR_std / 50
  [29] known_frac_step            — 1 if known, 0 if hidden
  [30] tvt_input_mean_known / 10000  — mean TVT_input in step (0 if hidden)
  [31] anchor_tvt / 10000         — broadcast from anchor row
  [32] anchor_z / 10000           — broadcast
  [33] anchor_C / 10000           — (anchor_tvt + anchor_z) / 10000
  [34] known_dC_median / 0.05     — median forward dC over last 512 known rows
  [35] known_dC_mean / 0.05
  [36] known_dC_std / 0.05
  [37] known_dC_slope / 0.05      — linear trend of dC over last 512 known rows
  [38] known_GR_mean / 100        — GR mean over last 512 known rows
  [39] known_GR_std / 50
  [40] hidden_n_log               — log1p(n_hidden_rows) / 8
  [41] hidden_z_span / 100        — Z_last_hidden - Z_anchor

Fourier features (N_FOURIER = 32) appended after raw features:
  4 continuous inputs × 4 frequencies × 2 (sin, cos)
  inputs: step_frac, rel_step_from_anchor (clipped), MD_rel, Z_rel

Total N_FEATURES = 42 + 32 = 74.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    K_SEG,
    N_FEATURES,
    N_RAW_FEATURES,
    ROWS_PER_STEP,
    RACFormerConfig,
)

FEATURE_NAMES = [
    "step_frac",
    "rel_step_from_anchor",
    "rel_row_start_from_anchor",
    "rel_row_end_from_anchor",
    "MD_mean",
    "MD_std",
    "MD_slope",
    "X_mean",
    "X_std",
    "X_slope",
    "Y_mean",
    "Y_std",
    "Y_slope",
    "Z_mean",
    "Z_std",
    "Z_slope",
    "dZ_mean",
    "dZ_std",
    "dZ_min",
    "dZ_max",
    "dMD_mean",
    "dXY_mean",
    "GR_mean",
    "GR_std",
    "GR_min",
    "GR_max",
    "GR_valid_frac",
    "dGR_mean",
    "dGR_std",
    "known_frac_step",
    "tvt_input_mean_known",
    "anchor_tvt",
    "anchor_z",
    "anchor_C",
    "known_dC_median",
    "known_dC_mean",
    "known_dC_std",
    "known_dC_slope",
    "known_GR_mean",
    "known_GR_std",
    "hidden_n_log",
    "hidden_z_span",
]
assert len(FEATURE_NAMES) == N_RAW_FEATURES, f"Expected {N_RAW_FEATURES}, got {len(FEATURE_NAMES)}"


# ---------------------------------------------------------------------------
# Region IDs (for region embedding in model)
# ---------------------------------------------------------------------------
REGION_PAD = 0
REGION_KNOWN = 1
REGION_ANCHOR = 2
REGION_HIDDEN = 3


# ---------------------------------------------------------------------------
# Prior loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step aggregation helpers
# ---------------------------------------------------------------------------

def _step_agg(arr: np.ndarray, rows_per_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (mean, std, min, max) per step.  NaN-safe.  Returns (T,) arrays."""
    n = len(arr)
    usable = (n // rows_per_step) * rows_per_step
    if usable == 0:
        return (
            np.empty(0, np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
        )
    a = np.asarray(arr[:usable], np.float32).reshape(-1, rows_per_step)
    finite = np.isfinite(a)
    with np.errstate(all="ignore"):
        counts = finite.sum(axis=1).astype(np.float32)
        sums = np.where(finite, a, 0.0).sum(axis=1)
        means = np.where(counts > 0, sums / counts, np.nan)
        # std
        sums2 = np.where(finite, a ** 2, 0.0).sum(axis=1)
        variance = np.where(counts > 1, (sums2 - sums ** 2 / np.maximum(counts, 1)) / np.maximum(counts - 1, 1), 0.0)
        stds = np.sqrt(np.maximum(variance, 0.0))
        # min / max
        filled_min = np.where(finite, a, np.inf)
        filled_max = np.where(finite, a, -np.inf)
        mins = filled_min.min(axis=1)
        maxs = filled_max.max(axis=1)
        mins = np.where(counts > 0, mins, np.nan)
        maxs = np.where(counts > 0, maxs, np.nan)
    return means, stds, mins, maxs


def _step_mean(arr: np.ndarray, rows_per_step: int) -> np.ndarray:
    m, _, _, _ = _step_agg(arr, rows_per_step)
    return m


def _step_valid_frac(arr: np.ndarray, rows_per_step: int) -> np.ndarray:
    n = len(arr)
    usable = (n // rows_per_step) * rows_per_step
    if usable == 0:
        return np.empty(0, np.float32)
    a = np.isfinite(np.asarray(arr[:usable], np.float32)).reshape(-1, rows_per_step)
    return a.mean(axis=1).astype(np.float32)


def _step_slope(arr: np.ndarray, rows_per_step: int) -> np.ndarray:
    """Per-step linear slope via (last - first) / step_width.  Finite values only."""
    n = len(arr)
    usable = (n // rows_per_step) * rows_per_step
    if usable == 0:
        return np.empty(0, np.float32)
    a = np.asarray(arr[:usable], np.float32).reshape(-1, rows_per_step)
    # use first and last finite per step
    out = np.zeros(a.shape[0], np.float32)
    for i in range(a.shape[0]):
        row = a[i]
        finite_idx = np.where(np.isfinite(row))[0]
        if len(finite_idx) >= 2:
            out[i] = (row[finite_idx[-1]] - row[finite_idx[0]]) / (finite_idx[-1] - finite_idx[0] + 1e-6)
    return out


# ---------------------------------------------------------------------------
# Segment oracle computation
# ---------------------------------------------------------------------------

def compute_segment_oracle(R_true: np.ndarray, k_seg: int = K_SEG) -> np.ndarray:
    """Compute k_seg segment slope targets via ridge-smoothed LS.

    R_true: (H,) residual = TVT_hidden - base_tvt_hidden
    Returns: s_star (k_seg,) ft/row
    """
    H = len(R_true)
    if H < 2:
        return np.zeros(k_seg, np.float32)

    # Segment assignment
    p_arr = np.arange(H, dtype=np.float64) / max(H - 1, 1)
    seg_arr = np.clip(np.floor(p_arr * k_seg).astype(int), 0, k_seg - 1)

    # One-hot → cumsum to get design matrix M[H, K_SEG]
    one_hot = np.zeros((H, k_seg), np.float64)
    one_hot[np.arange(H), seg_arr] = 1.0
    M = np.cumsum(one_hot, axis=0)   # (H, k_seg)

    # First-difference matrix D
    D = np.zeros((k_seg - 1, k_seg), np.float64)
    for j in range(k_seg - 1):
        D[j, j] = -1.0
        D[j, j + 1] = 1.0

    # Ridge-smoothed LS
    lhs = M.T @ M + 1e-3 * D.T @ D + 1e-6 * np.eye(k_seg)
    rhs = M.T @ R_true.astype(np.float64)
    s_star = np.linalg.solve(lhs, rhs).astype(np.float32)
    return s_star


# ---------------------------------------------------------------------------
# Physical base TVT decomposition (no external priors)
# ---------------------------------------------------------------------------

def compute_base_tvt(
    Z_rows: np.ndarray,
    anchor_row: int,
    anchor_tvt: float,
    c0: float = 0.0,
) -> np.ndarray:
    """Anchored physical baseline TVT.

        base_tvt[i] = anchor_tvt - (Z[i] - Z_anchor) + c0 * (i - anchor_row)

    This is *not* an external prior — it's the hard physical decomposition
    `TVT ≈ -Z + C` plus a small constant C-drift estimate `c0` from the
    well's own known prefix.  Anchored exactly: base_tvt[anchor_row] = anchor_tvt.
    """
    row_offset = np.arange(len(Z_rows)) - anchor_row
    base = anchor_tvt - (Z_rows - Z_rows[anchor_row]) + c0 * row_offset
    return base.astype(np.float32)


# ---------------------------------------------------------------------------
# WellSample dataclass
# ---------------------------------------------------------------------------

@dataclass
class WellSample:
    well_id: str

    # Step-level tensors
    features: np.ndarray       # (T, N_FEATURES) float32 — includes Fourier
    region_ids: np.ndarray     # (T,) int32: REGION_PAD/KNOWN/ANCHOR/HIDDEN
    hidden_mask: np.ndarray    # (T,) bool — True = hidden step

    # Row-level tensors (for loss and materialization)
    base_tvt_rows: np.ndarray    # (n_rows,) float32 — anchored physical baseline (no external priors)
    tvt_rows: np.ndarray         # (n_rows,) float32 — true TVT for all rows (train only; NaN in test)
    tvt_input_rows: np.ndarray   # (n_rows,) float32 — NaN for hidden
    z_rows: np.ndarray           # (n_rows,) float32

    # Indices
    anchor_row: int
    anchor_step: int
    n_hidden_rows: int
    n_rows: int

    # Step ↔ row mapping
    row_to_step: np.ndarray     # (n_rows,) int32 — which step each row belongs to
    step_offset: int            # steps trimmed from front (if well was truncated)
    seq_len: int                # actual step count (before padding)

    # Oracle targets (training only; zeros for test/valid)
    s_star: np.ndarray           # (K_SEG,) float32 — segment slope oracle (residual to base_tvt)
    dC_forward: np.ndarray       # (n_hidden_rows,) float32 — dC_true for hidden rows
    top_state_step: np.ndarray   # (T,) int64: 0 flat, 1 down, 2 up, -100 unavailable
    top_event_step: np.ndarray   # (T,) float32: |dANCC| > eps
    top_teacher_mask: np.ndarray # (T,) bool: ANCC teacher is available

    # Row IDs for submission
    row_ids: list[str]           # len = n_rows
    hidden_row_ids: list[str]    # len = n_hidden_rows

    # Scalars
    last_known_tvt: float
    last_known_z: float
    c0: float                    # median dC slope over last 512 known rows

    # Tail class label (for oversampling)
    tail_class: str = "unknown"


# ---------------------------------------------------------------------------
# Main well feature builder
# ---------------------------------------------------------------------------

TOP_STATE_IGNORE = -100


def _build_top_teacher_steps(
    horizontal: pd.DataFrame,
    row_to_step: np.ndarray,
    n_steps: int,
    eps: float,
    has_tvt: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train-only ANCC top-event/top-direction labels per model step."""
    top_state = np.full(n_steps, TOP_STATE_IGNORE, dtype=np.int64)
    top_event = np.zeros(n_steps, dtype=np.float32)
    top_mask = np.zeros(n_steps, dtype=bool)

    if not has_tvt or "ANCC" not in horizontal.columns:
        return top_state, top_event, top_mask

    ancc = pd.to_numeric(horizontal["ANCC"], errors="coerce").to_numpy(np.float32)
    if len(ancc) == 0:
        return top_state, top_event, top_mask

    d_ancc = np.empty_like(ancc)
    d_ancc[1:] = ancc[1:] - ancc[:-1]
    d_ancc[0] = 0.0

    valid = np.isfinite(d_ancc)
    for t in range(n_steps):
        mask = (row_to_step == t) & valid
        vals = d_ancc[mask]
        if len(vals) == 0:
            continue
        mean_d = float(np.mean(vals))
        top_mask[t] = True
        if mean_d > eps:
            top_state[t] = 1
            top_event[t] = 1.0
        elif mean_d < -eps:
            top_state[t] = 2
            top_event[t] = 1.0
        else:
            top_state[t] = 0

    return top_state, top_event, top_mask


def _build_well_sample(
    well_id: str,
    horizontal: pd.DataFrame,
    rows_per_step: int,
    max_seq_len: int,
    last_known_window: int,
    tail_class: str,
    k_seg: int = K_SEG,
    bin_shift: int = 0,
    use_c0_drift: bool = True,
    top_teacher_eps: float = 0.10,
) -> WellSample | None:
    """Build one WellSample from raw CSV.  No external priors are used —
    the physical baseline `base_tvt[i] = anchor_tvt - (Z[i] - Z_anchor) + c0*(i-anchor)`
    is computed from the well's own anchor + known prefix."""

    n_raw = len(horizontal)
    if n_raw < 4 * rows_per_step:
        return None

    # ---- raw columns ----
    MD = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float32)
    X = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(np.float32)
    Y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(np.float32)
    Z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float32)
    GR = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float32)
    TVT_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)

    # TVT is train-only.  In test split column is absent.  We must never use it
    # for any feature construction; only as training target / oracle.
    if "TVT" in horizontal.columns:
        TVT = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float32)
        has_tvt = True
    else:
        TVT = np.full(n_raw, np.nan, dtype=np.float32)
        has_tvt = False

    # C_rows (TVT + Z) is leakage-bearing — only valid for training.
    # For anchor statistics we use TVT_input + Z, which is safe.
    C_input_rows = TVT_input + Z   # NaN where TVT_input is NaN

    # ---- identify anchor ----
    known_mask_raw = np.isfinite(TVT_input)
    hidden_mask_raw = ~known_mask_raw
    known_indices = np.flatnonzero(known_mask_raw)
    hidden_indices = np.flatnonzero(hidden_mask_raw)

    if len(known_indices) == 0 or len(hidden_indices) == 0:
        return None  # nothing to predict or no context

    anchor_row = int(known_indices[-1])
    anchor_tvt = float(TVT_input[anchor_row])
    anchor_z = float(Z[anchor_row])
    anchor_C = anchor_tvt + anchor_z
    if not np.isfinite(anchor_tvt) or not np.isfinite(anchor_z):
        return None

    n_hidden_rows = len(hidden_indices)

    # ---- anchor statistics (last_known_window rows) ----
    win_start = max(0, anchor_row - last_known_window + 1)
    known_in_window = known_indices[known_indices >= win_start]

    if len(known_in_window) >= 2:
        # Use C_input_rows (TVT_input + Z) — safe for both train and test
        C_window = C_input_rows[known_in_window]
        dC_window = C_window[1:] - C_window[:-1]   # forward differences
        dC_window_finite = dC_window[np.isfinite(dC_window)]
        if len(dC_window_finite) > 0:
            c0 = float(np.median(dC_window_finite))
            dC_mean = float(np.mean(dC_window_finite))
            dC_std = float(np.std(dC_window_finite))
            # linear slope of dC: regress dC on index
            n_dc = len(dC_window_finite)
            if n_dc >= 3:
                xs = np.arange(n_dc, dtype=np.float64)
                dC_slope = float(np.polyfit(xs, dC_window_finite, 1)[0])
            else:
                dC_slope = 0.0
        else:
            c0 = dC_mean = dC_std = dC_slope = 0.0
    else:
        c0 = dC_mean = dC_std = dC_slope = 0.0

    GR_window = GR[known_in_window] if len(known_in_window) > 0 else np.array([])
    GR_window_finite = GR_window[np.isfinite(GR_window)]
    known_GR_mean = float(np.mean(GR_window_finite)) if len(GR_window_finite) > 0 else 0.0
    known_GR_std = float(np.std(GR_window_finite)) if len(GR_window_finite) > 1 else 0.0

    # ---- physical baseline TVT (per-row) ----
    # base_tvt = anchor_tvt - (Z - Z_anchor) + c0 * (i - anchor_row)
    # c0=0 disables the small known-prefix drift correction.
    base_c0 = c0 if use_c0_drift else 0.0
    base_tvt_raw = compute_base_tvt(Z, anchor_row, anchor_tvt, c0=base_c0)

    # ---- hidden span ----
    hidden_z_span = float(Z[hidden_indices[-1]] - anchor_z) if len(hidden_indices) > 0 else 0.0

    # ---- step assignment with bin_shift ----
    effective_step = (np.arange(n_raw) + bin_shift) // rows_per_step
    n_steps_total = int(effective_step[-1]) + 1

    # Trim to max_seq_len (keep hidden section + context)
    step_offset = 0
    if n_steps_total > max_seq_len:
        trim_rows = n_steps_total * rows_per_step - max_seq_len * rows_per_step
        trim_rows = min(trim_rows, anchor_row)  # don't trim anchor
        step_offset = trim_rows // rows_per_step * rows_per_step  # align to step boundary
        effective_step = effective_step - (step_offset // rows_per_step if False else 0)

    # Recompute with offset
    row_to_step = np.clip(
        (np.arange(n_raw) + bin_shift) // rows_per_step - (step_offset + bin_shift) // rows_per_step,
        0,
        max_seq_len - 1,
    ).astype(np.int32)

    # Actual anchor step after mapping
    anchor_step = int(row_to_step[anchor_row])
    n_steps = int(row_to_step[-1]) + 1
    n_steps = min(n_steps, max_seq_len)

    top_state_step, top_event_step, top_teacher_mask = _build_top_teacher_steps(
        horizontal=horizontal,
        row_to_step=row_to_step,
        n_steps=n_steps,
        eps=top_teacher_eps,
        has_tvt=has_tvt,
    )

    # ---- step-level aggregations ----

    # Helper to aggregate per-step from row array
    def _step_from_map(arr_: np.ndarray, func="mean") -> np.ndarray:
        """Aggregate arr_ into n_steps using row_to_step assignment."""
        out = np.zeros(n_steps, np.float32)
        counts = np.zeros(n_steps, np.float32)
        valid = np.isfinite(arr_)
        for t in range(n_steps):
            in_step = row_to_step == t
            vals = arr_[in_step & valid]
            if len(vals) > 0:
                if func == "mean":
                    out[t] = np.mean(vals)
                elif func == "std":
                    out[t] = np.std(vals) if len(vals) > 1 else 0.0
                elif func == "min":
                    out[t] = np.min(vals)
                elif func == "max":
                    out[t] = np.max(vals)
                elif func == "sum":
                    out[t] = np.sum(vals)
                counts[t] = len(vals)
        return out

    # Vectorized step aggregation (faster)
    def _vagg(arr_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean, std, min, max, valid_frac) per step. Shape (n_steps,)."""
        out_mean = np.full(n_steps, np.nan, np.float32)
        out_std = np.zeros(n_steps, np.float32)
        out_min = np.full(n_steps, np.nan, np.float32)
        out_max = np.full(n_steps, np.nan, np.float32)
        out_vfrac = np.zeros(n_steps, np.float32)
        valid = np.isfinite(arr_)
        for t in range(n_steps):
            mask = row_to_step == t
            total_in_step = mask.sum()
            vals = arr_[mask & valid]
            if len(vals) > 0:
                out_mean[t] = np.mean(vals)
                out_std[t] = np.std(vals) if len(vals) > 1 else 0.0
                out_min[t] = np.min(vals)
                out_max[t] = np.max(vals)
                out_vfrac[t] = len(vals) / max(1, total_in_step)
        return out_mean, out_std, out_min, out_max, out_vfrac

    # Geometry aggregations
    MD_m, MD_s, _, _, _ = _vagg(MD)
    X_m, X_s, _, _, _ = _vagg(X)
    Y_m, Y_s, _, _, _ = _vagg(Y)
    Z_m, Z_s, _, _, _ = _vagg(Z)

    # Deltas (forward diff per row, then aggregate)
    def _fwd_diff(arr_: np.ndarray) -> np.ndarray:
        d = np.empty_like(arr_)
        d[:-1] = arr_[1:] - arr_[:-1]
        d[-1] = np.nan
        return d

    dZ = _fwd_diff(Z)
    dMD = _fwd_diff(MD)
    dXY = np.sqrt((_fwd_diff(X)) ** 2 + (_fwd_diff(Y)) ** 2)

    dZ_m, dZ_s, dZ_n, dZ_x, _ = _vagg(dZ)
    dMD_m, _, _, _, _ = _vagg(dMD)
    dXY_m, _, _, _, _ = _vagg(dXY)

    # Slopes (per step: use first-to-last finite value span)
    MD_slope = np.zeros(n_steps, np.float32)
    X_slope = np.zeros(n_steps, np.float32)
    Y_slope = np.zeros(n_steps, np.float32)
    Z_slope = np.zeros(n_steps, np.float32)
    for t in range(n_steps):
        mask = row_to_step == t
        for arr_, out_ in [(MD, MD_slope), (X, X_slope), (Y, Y_slope), (Z, Z_slope)]:
            in_step = arr_[mask]
            finite_idx = np.where(np.isfinite(in_step))[0]
            if len(finite_idx) >= 2:
                out_[t] = (in_step[finite_idx[-1]] - in_step[finite_idx[0]]) / (
                    finite_idx[-1] - finite_idx[0] + 1e-6
                )

    # GR aggregations
    GR_m, GR_s, GR_n, GR_x, GR_vf = _vagg(GR)
    dGR = _fwd_diff(GR)
    dGR_m, dGR_s, _, _, _ = _vagg(dGR)

    # TVT_input aggregation (known steps)
    tvt_input_m, _, _, _, _ = _vagg(TVT_input)

    # ---- step region ids ----
    region_ids = np.full(n_steps, REGION_KNOWN, dtype=np.int32)
    for t in range(n_steps):
        mask = row_to_step == t
        if np.all(~known_mask_raw[mask]):
            region_ids[t] = REGION_HIDDEN
    region_ids[anchor_step] = REGION_ANCHOR

    hidden_step_mask = region_ids == REGION_HIDDEN   # (n_steps,) bool

    # ---- build feature matrix ----
    feats = np.zeros((n_steps, N_RAW_FEATURES), np.float32)

    T_minus1 = max(n_steps - 1, 1)
    step_idx = np.arange(n_steps, dtype=np.float32)
    row_start_step = step_idx * rows_per_step + step_offset   # approximate row start
    row_end_step = row_start_step + rows_per_step

    # [0] step_frac
    feats[:, 0] = step_idx / T_minus1
    # [1] rel_step_from_anchor
    feats[:, 1] = (step_idx - anchor_step) / 100.0
    # [2] rel_row_start_from_anchor
    feats[:, 2] = (row_start_step - anchor_row) / 5000.0
    # [3] rel_row_end_from_anchor
    feats[:, 3] = (row_end_step - anchor_row) / 5000.0

    # Geometry
    feats[:, 4] = np.nan_to_num(MD_m) / 10000.0
    feats[:, 5] = np.nan_to_num(MD_s) / 100.0
    feats[:, 6] = MD_slope / 100.0
    feats[:, 7] = np.nan_to_num(X_m) / 10000.0
    feats[:, 8] = np.nan_to_num(X_s) / 100.0
    feats[:, 9] = X_slope / 100.0
    feats[:, 10] = np.nan_to_num(Y_m) / 10000.0
    feats[:, 11] = np.nan_to_num(Y_s) / 100.0
    feats[:, 12] = Y_slope / 100.0
    feats[:, 13] = np.nan_to_num(Z_m) / 10000.0
    feats[:, 14] = np.nan_to_num(Z_s) / 100.0
    feats[:, 15] = Z_slope / 1.0
    feats[:, 16] = np.nan_to_num(dZ_m) / 1.0
    feats[:, 17] = np.nan_to_num(dZ_s) / 0.5
    feats[:, 18] = np.nan_to_num(dZ_n) / 2.0
    feats[:, 19] = np.nan_to_num(dZ_x) / 2.0
    feats[:, 20] = np.nan_to_num(dMD_m) / 1.0
    feats[:, 21] = np.nan_to_num(dXY_m) / 1.0

    # GR
    feats[:, 22] = np.nan_to_num(GR_m) / 100.0
    feats[:, 23] = np.nan_to_num(GR_s) / 50.0
    feats[:, 24] = np.nan_to_num(GR_n) / 100.0
    feats[:, 25] = np.nan_to_num(GR_x) / 100.0
    feats[:, 26] = GR_vf
    feats[:, 27] = np.nan_to_num(dGR_m) / 50.0
    feats[:, 28] = np.nan_to_num(dGR_s) / 50.0

    # TVT/anchor context
    feats[:, 29] = (region_ids == REGION_KNOWN).astype(np.float32)
    # tvt_input_mean_known: use mean for known steps
    tvt_input_for_feat = np.nan_to_num(tvt_input_m)
    feats[:, 30] = np.where(region_ids == REGION_KNOWN, tvt_input_for_feat, 0.0) / 10000.0
    # Broadcast anchors
    feats[:, 31] = anchor_tvt / 10000.0
    feats[:, 32] = anchor_z / 10000.0
    feats[:, 33] = anchor_C / 10000.0
    feats[:, 34] = c0 / 0.05
    feats[:, 35] = dC_mean / 0.05
    feats[:, 36] = dC_std / 0.05
    feats[:, 37] = dC_slope / 0.05
    feats[:, 38] = known_GR_mean / 100.0
    feats[:, 39] = known_GR_std / 50.0
    feats[:, 40] = float(np.log1p(n_hidden_rows) / 8.0)
    feats[:, 41] = hidden_z_span / 100.0

    # ---- Fourier features ----
    fourier_inputs = np.stack([
        step_idx / T_minus1,
        np.clip((step_idx - anchor_step) / 100.0, -3.0, 3.0),
        np.nan_to_num(MD_m - MD_m[anchor_step]) / 10000.0,
        np.nan_to_num(Z_m - Z_m[anchor_step]) / 1000.0,
    ], axis=1)   # (T, 4)

    freqs = np.array([1.0, 2.0, 4.0, 8.0])
    fourier = np.concatenate([
        np.sin(fourier_inputs[:, :, None] * freqs[None, None, :] * 2 * np.pi).reshape(n_steps, -1),
        np.cos(fourier_inputs[:, :, None] * freqs[None, None, :] * 2 * np.pi).reshape(n_steps, -1),
    ], axis=1)   # (T, 32)

    features = np.concatenate([feats, fourier.astype(np.float32)], axis=1)  # (T, 80)
    assert features.shape == (n_steps, N_FEATURES), f"Feature shape mismatch: {features.shape}"

    # ---- oracle targets (train-only; zeros for test) ----
    # Residuals are now relative to base_tvt (no external prior).
    hidden_row_idxs = hidden_indices

    if has_tvt:
        TVT_hidden = TVT[hidden_row_idxs]
        base_hidden = base_tvt_raw[hidden_row_idxs]
        if np.all(np.isfinite(TVT_hidden)):
            R_true = (TVT_hidden - base_hidden).astype(np.float64)
            s_star = compute_segment_oracle(R_true, k_seg)
            # dC forward for hidden rows (uses true TVT — train only)
            C_all = TVT + Z
            dC_all = np.empty_like(C_all)
            dC_all[1:] = C_all[1:] - C_all[:-1]
            dC_all[0] = 0.0
            dC_forward_hidden = dC_all[hidden_row_idxs].astype(np.float32)
        else:
            s_star = np.zeros(k_seg, np.float32)
            dC_forward_hidden = np.zeros(len(hidden_row_idxs), np.float32)
    else:
        s_star = np.zeros(k_seg, np.float32)
        dC_forward_hidden = np.zeros(len(hidden_row_idxs), np.float32)

    # ---- row IDs ----
    row_ids = [f"{well_id}_{i}" for i in range(n_raw)]
    hidden_row_ids = [row_ids[i] for i in hidden_row_idxs]

    return WellSample(
        well_id=well_id,
        features=features,
        region_ids=region_ids,
        hidden_mask=hidden_step_mask,
        base_tvt_rows=base_tvt_raw,
        tvt_rows=TVT,
        tvt_input_rows=TVT_input,
        z_rows=Z,
        anchor_row=anchor_row,
        anchor_step=anchor_step,
        n_hidden_rows=n_hidden_rows,
        n_rows=n_raw,
        row_to_step=row_to_step,
        step_offset=step_offset,
        seq_len=n_steps,
        s_star=s_star,
        dC_forward=dC_forward_hidden,
        top_state_step=top_state_step,
        top_event_step=top_event_step,
        top_teacher_mask=top_teacher_mask,
        row_ids=row_ids,
        hidden_row_ids=hidden_row_ids,
        last_known_tvt=anchor_tvt,
        last_known_z=anchor_z,
        c0=c0,
        tail_class=tail_class,
    )


# ---------------------------------------------------------------------------
# Load all wells
# ---------------------------------------------------------------------------

def discover_wells(data_dir: Path, k_wells: int = -1) -> list[tuple[str, Path, Path]]:
    """Return [(well_id, horizontal_path, typewell_path), ...].

    Looks for *__horizontal_well.csv files at the top level of data_dir.
    Caller is responsible for picking the right subdir (train / test /
    public_train / public_test / dataset root).
    """
    data_dir = Path(data_dir)
    horizontal_paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if not horizontal_paths:
        raise FileNotFoundError(f"No horizontal well CSVs in {data_dir}")
    if k_wells > 0:
        horizontal_paths = horizontal_paths[:k_wells]
    result = []
    for hp in horizontal_paths:
        well_id = hp.name.replace("__horizontal_well.csv", "")
        tp = hp.with_name(f"{well_id}__typewell.csv")
        result.append((well_id, hp, tp))
    return result


def _resolve_split_dir(data_dir: Path, split: str) -> Path:
    """Locate the directory containing CSVs for the requested split.

    Tries (in order):
      1.  data_dir/<split>                  e.g. MTPNet/data/train
      2.  data_dir/public_<split>           e.g. ClearML cache / old/data/public_test
      3.  data_dir                          for ClearML upload (CSVs at root → train)

    Returns the first existing directory containing *__horizontal_well.csv,
    otherwise raises FileNotFoundError.
    """
    data_dir = Path(data_dir)
    candidates = [data_dir / split, data_dir / f"public_{split}", data_dir]
    for c in candidates:
        if c.is_dir() and any(c.glob("*__horizontal_well.csv")):
            return c
    raise FileNotFoundError(
        f"No horizontal well CSVs found for split={split} under {data_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_all_wells(
    cfg: RACFormerConfig,
    split: str = "train",
    bin_shift: int = 0,
) -> list[WellSample]:
    """Load all wells for the given split (train / test).

    Resolves the data directory via `_resolve_split_dir` so the same code
    works with both layouts:
      - MTPNet/data/{train,test}/*.csv
      - ClearML cache: /tmp/.../<dataset>/*.csv  +  /tmp/.../<dataset>/public_test/*.csv
    """
    well_dir = _resolve_split_dir(Path(cfg.data_dir), split)

    wells = discover_wells(well_dir, cfg.k_wells)
    if not wells:
        raise FileNotFoundError(f"No wells found in {well_dir}")

    samples: list[WellSample] = []
    skipped = 0
    for well_id, hp, tp in wells:
        try:
            horizontal = pd.read_csv(hp)
        except Exception as e:
            print(f"[dataset] skip {well_id}: {e}", file=sys.stderr)
            skipped += 1
            continue

        sample = _build_well_sample(
            well_id=well_id,
            horizontal=horizontal,
            rows_per_step=cfg.data.rows_per_step,
            max_seq_len=cfg.data.max_seq_len,
            last_known_window=cfg.data.last_known_window,
            tail_class="unknown",
            k_seg=cfg.model.k_seg,
            bin_shift=bin_shift,
            use_c0_drift=cfg.data.use_c0_drift,
            top_teacher_eps=cfg.data.top_teacher_eps,
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if skipped:
        print(f"[dataset] skipped {skipped} wells", file=sys.stderr)
    print(f"[dataset] loaded {len(samples)} wells (split={split}, bin_shift={bin_shift})", file=sys.stderr)
    return samples


# ---------------------------------------------------------------------------
# Pseudo-anchor augmentation
# ---------------------------------------------------------------------------

def apply_pseudo_anchor(
    sample: WellSample,
    rng: np.random.Generator,
    lo: float = 0.45,
    hi: float = 0.85,
    k_seg: int = K_SEG,
    rows_per_step: int = ROWS_PER_STEP,
) -> WellSample | None:
    """Create a pseudo-anchor version of the sample for augmentation.

    Randomly picks a pseudo-anchor within [lo, hi] fraction of known rows.
    Masks TVT_input after pseudo-anchor as hidden.
    Returns a new WellSample or None if not enough room.
    """
    n_raw = sample.n_rows
    known_indices = np.where(np.isfinite(sample.tvt_input_rows))[0]
    if len(known_indices) < 4:
        return None

    # Choose pseudo anchor position
    lo_idx = int(lo * len(known_indices))
    hi_idx = int(hi * len(known_indices))
    if hi_idx <= lo_idx:
        return None

    pseudo_anchor_pos = int(rng.integers(lo_idx, hi_idx))
    pseudo_anchor_row = int(known_indices[pseudo_anchor_pos])

    # Build new tvt_input: mask rows after pseudo_anchor
    new_tvt_input = sample.tvt_input_rows.copy()
    new_tvt_input[pseudo_anchor_row + 1:] = np.nan

    # We cannot rebuild without the raw horizontal DataFrame, so update the
    # existing tensors consistently for the pseudo-hidden section.

    anchor_step_new = sample.row_to_step[pseudo_anchor_row]
    region_ids_new = sample.region_ids.copy()

    # Mark steps after pseudo-anchor as hidden
    for t in range(sample.seq_len):
        if t > anchor_step_new:
            region_ids_new[t] = REGION_HIDDEN
        elif t == anchor_step_new:
            region_ids_new[t] = REGION_ANCHOR
        else:
            region_ids_new[t] = REGION_KNOWN

    hidden_mask_new = (region_ids_new == REGION_HIDDEN)

    # Recompute s_star for the new pseudo-hidden section.
    # base_tvt must be rebuilt against the new anchor.
    new_anchor_tvt = float(sample.tvt_input_rows[pseudo_anchor_row])
    pseudo_hidden_rows = np.arange(pseudo_anchor_row + 1, n_raw)
    new_base_tvt_rows = compute_base_tvt(
        sample.z_rows, pseudo_anchor_row, new_anchor_tvt, c0=sample.c0,
    )

    TVT_pseudo_hidden = sample.tvt_rows[pseudo_hidden_rows]
    base_pseudo_hidden = new_base_tvt_rows[pseudo_hidden_rows]
    if len(pseudo_hidden_rows) > 1 and np.all(np.isfinite(TVT_pseudo_hidden)):
        R_pseudo = (TVT_pseudo_hidden - base_pseudo_hidden).astype(np.float64)
        s_star_new = compute_segment_oracle(R_pseudo, k_seg)
    else:
        s_star_new = np.zeros(k_seg, np.float32)

    # Update features: clear tvt_input_mean_known for pseudo-hidden steps
    feats_new = sample.features.copy()
    for t in range(sample.seq_len):
        if region_ids_new[t] == REGION_HIDDEN:
            feats_new[t, 29] = 0.0   # known_frac_step
            feats_new[t, 30] = 0.0   # tvt_input_mean_known

    pseudo_hidden_ids = [sample.row_ids[i] for i in pseudo_hidden_rows]

    # Recompute dC_forward for the pseudo-hidden section so it matches
    # the new anchor_row (fix for issue #1 consistency).
    C_all = sample.tvt_rows + sample.z_rows
    dC_all = np.empty_like(C_all)
    dC_all[1:] = C_all[1:] - C_all[:-1]
    dC_all[0] = 0.0
    dC_new = dC_all[pseudo_hidden_rows].astype(np.float32) if len(pseudo_hidden_rows) > 0 else np.zeros(0, np.float32)

    return WellSample(
        well_id=sample.well_id + "_pseudo",
        features=feats_new,
        region_ids=region_ids_new,
        hidden_mask=hidden_mask_new,
        base_tvt_rows=new_base_tvt_rows,
        tvt_rows=sample.tvt_rows,
        tvt_input_rows=new_tvt_input,
        z_rows=sample.z_rows,
        anchor_row=pseudo_anchor_row,
        anchor_step=int(anchor_step_new),
        n_hidden_rows=len(pseudo_hidden_rows),
        n_rows=n_raw,
        row_to_step=sample.row_to_step,
        step_offset=sample.step_offset,
        seq_len=sample.seq_len,
        s_star=s_star_new,
        dC_forward=dC_new,
        top_state_step=sample.top_state_step,
        top_event_step=sample.top_event_step,
        top_teacher_mask=sample.top_teacher_mask,
        row_ids=sample.row_ids,
        hidden_row_ids=pseudo_hidden_ids,
        last_known_tvt=new_anchor_tvt,
        last_known_z=float(sample.z_rows[pseudo_anchor_row]),
        c0=sample.c0,
        tail_class="pseudo_" + sample.tail_class,
    )


# ---------------------------------------------------------------------------
# Dataset with padding and augmentation
# ---------------------------------------------------------------------------

def _pad_or_trim(arr: np.ndarray, target_len: int, pad_val: float = 0.0) -> np.ndarray:
    n = len(arr)
    if n == target_len:
        return arr
    if n > target_len:
        return arr[:target_len]
    pad_shape = (target_len - n,) + arr.shape[1:]
    return np.concatenate([arr, np.full(pad_shape, pad_val, dtype=arr.dtype)], axis=0)


class RACDataset(Dataset):
    """PyTorch Dataset.

    Returns a batch dict with keys:
      features:    (L, N_FEATURES) float32
      region_ids:  (L,) int64
      hidden_mask: (L,) bool
      pad_mask:    (L,) bool  — True = padding (ignored by attention)
      s_star:      (K_SEG,) float32  — segment oracle
      seq_len:     int
      well_id:     str
      anchor_step: int
      anchor_tvt:  float
      anchor_z:    float
      c0:          float
    """

    def __init__(
        self,
        samples: list[WellSample],
        max_seq_len: int,
        augment: bool = False,
        aug_cfg=None,
        seed: int = 42,
        k_seg: int = K_SEG,
        rows_per_step: int = ROWS_PER_STEP,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.aug_cfg = aug_cfg
        self._rng = np.random.default_rng(seed)
        self.k_seg = k_seg
        self.rows_per_step = rows_per_step

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        feats = sample.features.copy()
        region_ids = sample.region_ids.copy()
        hidden_mask = sample.hidden_mask.copy()
        s_star = sample.s_star.copy()

        # Active sample reference — used by hidden-row tensor builders below.
        # If pseudo-anchor fires, we replace this with the pseudo-sample so that
        # features, anchor_row, anchor_step, base_tvt_hidden, tvt_hidden, dC_hidden
        # all come from the SAME sample (fix for issue #1: label corruption).
        active = sample

        if self.augment and self.aug_cfg is not None:
            aug = self.aug_cfg
            rng = self._rng

            # 1. Pseudo-anchor crop
            if rng.random() < aug.pseudo_anchor_prob:
                pseudo = apply_pseudo_anchor(
                    sample, rng,
                    lo=aug.pseudo_anchor_lo,
                    hi=aug.pseudo_anchor_hi,
                    k_seg=self.k_seg,
                    rows_per_step=self.rows_per_step,
                )
                if pseudo is not None:
                    active = pseudo
                    feats = pseudo.features.copy()
                    region_ids = pseudo.region_ids.copy()
                    hidden_mask = pseudo.hidden_mask.copy()
                    s_star = pseudo.s_star.copy()

            # 2. GR dropout for some hidden steps
            if rng.random() < aug.gr_dropout_prob:
                hidden_step_idxs = np.where(hidden_mask)[0]
                if len(hidden_step_idxs) > 0:
                    n_drop = max(1, int(len(hidden_step_idxs) * aug.gr_dropout_frac))
                    drop_steps = rng.choice(hidden_step_idxs, size=n_drop, replace=False)
                    feats[drop_steps, 22:29] = 0.0   # zero GR features [22-28]
                    feats[drop_steps, 26] = 0.0       # valid_frac = 0

            # 3. Feature noise (no prior dropout — no external priors in no-prior build)
            if aug.feature_noise_std > 0:
                noise = rng.normal(0, aug.feature_noise_std, feats.shape).astype(np.float32)
                # Don't noise the anchor/broadcast scalars (31-41 are anchor-derived)
                noise[:, 31:42] = 0.0
                feats = feats + noise

        T = active.seq_len
        L = self.max_seq_len
        pad = L - T
        assert pad >= 0, f"seq_len {T} > max_seq_len {L}"

        pad_mask = np.zeros(L, dtype=bool)
        if pad > 0:
            feats = _pad_or_trim(feats, L, 0.0)
            region_ids_padded = np.concatenate([region_ids, np.full(pad, REGION_PAD, dtype=np.int32)])
            hidden_mask_padded = np.concatenate([hidden_mask, np.zeros(pad, dtype=bool)])
            top_state_padded = np.concatenate([
                active.top_state_step,
                np.full(pad, TOP_STATE_IGNORE, dtype=np.int64),
            ])
            top_event_padded = np.concatenate([active.top_event_step, np.zeros(pad, dtype=np.float32)])
            top_mask_padded = np.concatenate([active.top_teacher_mask, np.zeros(pad, dtype=bool)])
            pad_mask[T:] = True
        else:
            feats = feats[:L]
            region_ids_padded = region_ids[:L]
            hidden_mask_padded = hidden_mask[:L]
            top_state_padded = active.top_state_step[:L]
            top_event_padded = active.top_event_step[:L]
            top_mask_padded = active.top_teacher_mask[:L]

        # ---- Hidden-row tensors (built from `active` to stay consistent with features) ----
        # MAX_H is the theoretical upper bound (max_seq_len * rows_per_step).
        MAX_H = self.max_seq_len * self.rows_per_step
        H = active.n_hidden_rows
        ar = active.anchor_row

        if H > MAX_H:
            raise ValueError(
                f"{active.well_id}: n_hidden_rows={H} exceeds MAX_H={MAX_H} "
                f"(max_seq_len={self.max_seq_len}, rows_per_step={self.rows_per_step}). "
                f"Bump max_seq_len in config to at least {(H + self.rows_per_step - 1) // self.rows_per_step}, "
                f"or use a larger rows_per_step."
            )

        def _pad_h(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
            """Pad 1-D float32 array of length H to MAX_H."""
            out = np.full(MAX_H, fill, dtype=np.float32)
            n = min(len(arr), MAX_H)
            out[:n] = arr[:n]
            return out

        def _pad_h_long(arr: np.ndarray, fill: int = 0) -> np.ndarray:
            out = np.full(MAX_H, fill, dtype=np.int64)
            n = min(len(arr), MAX_H)
            out[:n] = arr[:n]
            return out

        # Rows immediately after anchor: indices [ar+1 .. ar+H]
        _base_h = _pad_h(active.base_tvt_rows[ar + 1: ar + 1 + H])
        _tvt_h = _pad_h(active.tvt_rows[ar + 1: ar + 1 + H])
        _z_h = _pad_h(active.z_rows[ar + 1: ar + 1 + H])
        _dC_h = _pad_h(active.dC_forward) if active is sample else _pad_h(np.zeros(H, np.float32))

        # hidden_row_to_step: critical for materializer correctness
        # Each hidden row's absolute step index (after bin_shift + step_offset).
        hidden_steps_raw = active.row_to_step[ar + 1: ar + 1 + H]
        _hidden_step = _pad_h_long(hidden_steps_raw, fill=0)

        return {
            "features": torch.from_numpy(feats).float(),               # (L, N_FEATURES)
            "region_ids": torch.from_numpy(region_ids_padded).long(),  # (L,)
            "hidden_mask": torch.from_numpy(hidden_mask_padded),       # (L,) bool
            "pad_mask": torch.from_numpy(pad_mask),                    # (L,) bool
            "top_state_step": torch.from_numpy(top_state_padded).long(),  # (L,)
            "top_event_step": torch.from_numpy(top_event_padded).float(),  # (L,)
            "top_teacher_mask": torch.from_numpy(top_mask_padded),      # (L,) bool
            "s_star": torch.from_numpy(s_star).float(),                # (K_SEG,)
            "seq_len": torch.tensor(T, dtype=torch.long),
            "well_id": active.well_id,
            "anchor_step": torch.tensor(active.anchor_step, dtype=torch.long),
            "anchor_tvt": torch.tensor(active.last_known_tvt, dtype=torch.float32),
            "anchor_z": torch.tensor(active.last_known_z, dtype=torch.float32),
            "c0": torch.tensor(active.c0, dtype=torch.float32),
            # Hidden-row tensors (padded to MAX_H = max_seq_len * rows_per_step)
            "n_hidden_rows": torch.tensor(H, dtype=torch.long),
            "base_tvt_hidden": torch.from_numpy(_base_h).float(),       # (MAX_H,)
            "tvt_hidden": torch.from_numpy(_tvt_h).float(),             # (MAX_H,)
            "z_hidden": torch.from_numpy(_z_h).float(),                 # (MAX_H,)
            "dC_hidden": torch.from_numpy(_dC_h).float(),               # (MAX_H,)
            "hidden_row_to_step": torch.from_numpy(_hidden_step).long(),  # (MAX_H,)
        }
