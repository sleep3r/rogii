"""Oracle ceiling computation for the offset-state decoder.

Computes the best possible TVT RMSE for a given well under:
  - global single offset (grid search)
  - K-segment constant offsets (least squares, exact optimum)

These oracle numbers establish the achievable ceiling. The oracle values
MUST match the established benchmarks:
  K=1  (global): ~7.64 RMSE
  K=3:            ~3.02 RMSE
  K=5:            ~1.82 RMSE
  K=15:           ~0.65 RMSE

Usage:
    from mtpnet.oracle import compute_oracle_report, find_global_offset_oracle

    samples = load_offset_samples("data/train")
    report  = compute_oracle_report(samples, ks=[1, 3, 5, 15])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .offsets import (
    make_offset_grid,
    segment_assignment,
    tvt_from_global_offset,
    tvt_from_ksegment_offsets,
    tvt_from_offset_per_row,
    expand_segment_offsets,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class GlobalOracleResult(NamedTuple):
    offset: float       # best offset value (ft/row)
    bin_idx: int        # index in the offset grid
    rmse: float         # oracle RMSE (ft)


class KSegOracleResult(NamedTuple):
    K: int
    offsets: np.ndarray  # (K,) oracle per-segment offsets
    rmse: float          # oracle RMSE (ft) — should match known ceilings


# ---------------------------------------------------------------------------
# Row-RMSE helper
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(t)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))


# ---------------------------------------------------------------------------
# 1. Global offset oracle (grid search)
# ---------------------------------------------------------------------------

def find_global_offset_oracle(
    sample,  # OffsetSample
    grid: np.ndarray | None = None,
) -> GlobalOracleResult:
    """Find the best single global offset by exhaustive grid search.

    Args:
        sample : OffsetSample with has_true = True
        grid   : offset grid; defaults to make_offset_grid()

    Returns:
        GlobalOracleResult(offset, bin_idx, rmse)
    """
    if grid is None:
        grid = make_offset_grid()

    tvt_true = sample.tvt_hidden_true
    if tvt_true is None:
        raise ValueError(f"Well {sample.well_id} has no true TVT")

    z = sample.z
    anchor_row = sample.anchor_row
    anchor_tvt = sample.anchor_tvt
    hidden_rows = sample.hidden_rows

    best_rmse = float("inf")
    best_i = 0

    for i, c in enumerate(grid):
        pred = tvt_from_global_offset(z, anchor_row, anchor_tvt, hidden_rows, c)
        r = _rmse(pred, tvt_true)
        if r < best_rmse:
            best_rmse = r
            best_i = i

    return GlobalOracleResult(
        offset=float(grid[best_i]),
        bin_idx=best_i,
        rmse=best_rmse,
    )


# ---------------------------------------------------------------------------
# 2. K-segment oracle (least squares — exact optimum)
# ---------------------------------------------------------------------------

def _build_segment_design_matrix(
    n_hidden: int,
    K: int,
) -> np.ndarray:
    """Build the cumulative design matrix M (n_hidden x K).

    M[i, k] = number of hidden rows in segment k up to and including row i.

    This satisfies: (M @ c)[i] = cumulative offset = sum_{j<=i} c_seg(j)
    where c_seg(j) is the constant offset of segment k for row j.
    """
    seg = segment_assignment(n_hidden, K)   # (n_hidden,) in [0, K-1]
    one_hot = np.zeros((n_hidden, K), dtype=np.float64)
    one_hot[np.arange(n_hidden), seg] = 1.0
    M = np.cumsum(one_hot, axis=0)          # cumulative count
    return M


def find_kseg_oracle_ls(
    sample,  # OffsetSample
    K: int,
) -> KSegOracleResult:
    """Find K-segment oracle offsets by least squares (exact optimum).

    Solves:
        min_c || M @ c - y ||^2
    where:
        y[i] = TVT_true[h[i]] - base_TVT[h[i]]
             = TVT_true[h[i]] - (anchor_tvt - (z[h[i]] - z_anchor))
             = C_true[h[i]] - C_anchor
        M    = cumulative segment design matrix

    This is the exact minimiser of TVT RMSE over the K-segment family.

    Args:
        sample : OffsetSample with has_true = True
        K      : number of equal-spaced segments

    Returns:
        KSegOracleResult(K, offsets, rmse)
    """
    tvt_true = sample.tvt_hidden_true
    if tvt_true is None:
        raise ValueError(f"Well {sample.well_id} has no true TVT")

    H = sample.n_hidden
    # If fewer hidden rows than segments, clamp K to H
    K_eff = max(1, min(K, H))

    z = sample.z.astype(np.float64)
    anchor_row = sample.anchor_row
    anchor_tvt = float(sample.anchor_tvt)
    anchor_z = float(sample.anchor_z)
    hidden_rows = sample.hidden_rows

    # Residual to integrate: y[i] = C_true[h[i]] - C_anchor
    #  = (tvt_true[h[i]] + z[h[i]]) - (anchor_tvt + anchor_z)
    tvt_t = np.asarray(tvt_true, dtype=np.float64)
    y = (tvt_t + z[hidden_rows]) - (anchor_tvt + anchor_z)

    M = _build_segment_design_matrix(H, K_eff)

    # Ridge-smoothed least squares (same as RACFormer's compute_segment_oracle)
    D = np.zeros((K_eff - 1, K_eff), dtype=np.float64)
    for j in range(K_eff - 1):
        D[j, j] = -1.0
        D[j, j + 1] = 1.0
    lhs = M.T @ M + 1e-3 * D.T @ D + 1e-6 * np.eye(K_eff)
    rhs = M.T @ y
    c = np.linalg.solve(lhs, rhs)

    pred = tvt_from_ksegment_offsets(
        z=np.asarray(z, dtype=np.float32),
        anchor_row=anchor_row,
        anchor_tvt=anchor_tvt,
        hidden_rows=hidden_rows,
        offsets=c,
    )
    rmse = _rmse(pred, tvt_t)

    return KSegOracleResult(K=K, offsets=c.astype(np.float64), rmse=rmse)


# ---------------------------------------------------------------------------
# 3. dC-mean oracle (fast approximation — used for sanity checks)
# ---------------------------------------------------------------------------

def find_kseg_oracle_dc_mean(
    sample,  # OffsetSample
    K: int,
) -> KSegOracleResult:
    """K-segment oracle via mean dC per segment (fast approximation).

    This is NOT the exact optimum (unlike the LS approach), but it's
    simpler and gives a sanity check. The LS approach should give equal
    or better RMSE.

    true_offset_per_row[k] = d(TVT_true + Z)[k] = dC_true[k]
    oracle_offset_k        = mean(dC_true[rows in segment k])
    """
    tvt_true = sample.tvt_hidden_true
    if tvt_true is None:
        raise ValueError(f"Well {sample.well_id} has no true TVT")

    H = sample.n_hidden
    if H == 0:
        return KSegOracleResult(K=K, offsets=np.zeros(K), rmse=float("nan"))

    z = sample.z.astype(np.float64)
    tvt_t = np.asarray(tvt_true, dtype=np.float64)
    hidden_rows = sample.hidden_rows
    anchor_row = sample.anchor_row

    # Per-row true offset: dC = d(TVT + Z)
    prev_rows = np.concatenate([[anchor_row], hidden_rows[:-1]])
    prev_tvt = np.where(
        prev_rows == anchor_row,
        float(sample.anchor_tvt),
        tvt_t[np.searchsorted(hidden_rows, prev_rows)],
    )
    # Safer: build prev TVT array explicitly
    prev_tvt_arr = np.empty(H, dtype=np.float64)
    prev_tvt_arr[0] = float(sample.anchor_tvt)
    prev_tvt_arr[1:] = tvt_t[:-1]

    dtvt = tvt_t - prev_tvt_arr
    dz = z[hidden_rows] - z[np.concatenate([[anchor_row], hidden_rows[:-1]])]
    true_offset_per_row = dtvt + dz   # = dC per row

    # Segment mean
    seg = segment_assignment(H, K)
    c = np.zeros(K, dtype=np.float64)
    for k in range(K):
        mask = seg == k
        if np.any(mask):
            vals = true_offset_per_row[mask]
            finite = vals[np.isfinite(vals)]
            c[k] = float(np.mean(finite)) if len(finite) > 0 else 0.0

    pred = tvt_from_ksegment_offsets(
        z=np.asarray(sample.z, dtype=np.float32),
        anchor_row=anchor_row,
        anchor_tvt=float(sample.anchor_tvt),
        hidden_rows=hidden_rows,
        offsets=c,
    )
    rmse = _rmse(pred, tvt_t)
    return KSegOracleResult(K=K, offsets=c, rmse=rmse)


# ---------------------------------------------------------------------------
# 4. Populate oracle targets into samples
# ---------------------------------------------------------------------------

def compute_and_attach_oracles(
    samples: list,  # list[OffsetSample]
    ks: list[int] | None = None,
    grid: np.ndarray | None = None,
    verbose: bool = True,
) -> None:
    """Compute and store oracle offsets into samples in-place.

    After calling this, each sample will have:
        sample.global_offset_star  : float
        sample.kseg_offset_star    : {K: np.ndarray}

    Only run on training samples (sample.has_true = True).

    Args:
        samples : list of OffsetSample
        ks      : which K values to compute (default [1, 3, 5, 15])
        grid    : offset grid for global oracle
        verbose : print progress
    """
    if ks is None:
        ks = [1, 3, 5, 15]
    if grid is None:
        grid = make_offset_grid()

    train_samples = [s for s in samples if s.has_true]
    if verbose:
        print(f"Computing oracles for {len(train_samples)} training wells, K={ks}")

    for i, s in enumerate(train_samples):
        if verbose and (i % 100 == 0 or i == len(train_samples) - 1):
            print(f"  oracle {i + 1}/{len(train_samples)} — {s.well_id}")

        # Global oracle
        result = find_global_offset_oracle(s, grid)
        s.global_offset_star = result.offset

        # K-segment oracles
        s.kseg_offset_star = {}
        for K in ks:
            kseg_result = find_kseg_oracle_ls(s, K)
            s.kseg_offset_star[K] = kseg_result.offsets


# ---------------------------------------------------------------------------
# 5. Oracle report: pooled RMSE across all training wells
# ---------------------------------------------------------------------------

def compute_oracle_report(
    samples: list,  # list[OffsetSample]
    ks: list[int] | None = None,
    grid: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Compute pooled oracle RMSE for each K across all training wells.

    Returns a dict with keys like "global_grid_oracle", "K1_oracle", etc.,
    and also per-well breakdowns.

    Expected values (from audit):
        global_grid_oracle : ~7.64 ft
        K1_oracle          : ~7.59 ft  (same as global with LS)
        K3_oracle          : ~3.02 ft
        K5_oracle          : ~1.82 ft
        K15_oracle         : ~0.65 ft
    """
    if ks is None:
        ks = [1, 3, 5, 15]
    if grid is None:
        grid = make_offset_grid()

    train_samples = [s for s in samples if s.has_true]
    if not train_samples:
        return {"error": "no training samples"}

    if verbose:
        print(f"Oracle report: {len(train_samples)} wells, K={ks}")

    # Collect all predictions
    all_true: list[float] = []
    all_base: list[float] = []
    all_global: list[float] = []
    kseg_preds: dict[int, list[float]] = {K: [] for K in ks}
    kseg_true_grouped: dict[int, list[float]] = {K: [] for K in ks}

    per_well = []

    for s in train_samples:
        tvt_true = s.tvt_hidden_true
        if tvt_true is None:
            continue
        H = s.n_hidden

        # Base TVT (c0 drift)
        from .offsets import tvt_base
        base = tvt_base(s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, c0=s.c0)

        # Global grid oracle
        g_result = find_global_offset_oracle(s, grid)

        # K-segment oracles
        kseg_results: dict[int, KSegOracleResult] = {}
        for K in ks:
            kseg_results[K] = find_kseg_oracle_ls(s, K)

        all_true.extend(tvt_true.tolist())
        all_base.extend(base.tolist())
        all_global.extend(
            tvt_from_global_offset(
                s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, g_result.offset
            ).tolist()
        )

        well_row = {
            "well_id": s.well_id,
            "n_hidden": H,
            "base_rmse": _rmse(base, tvt_true),
            "global_grid_oracle_rmse": g_result.rmse,
            "global_grid_oracle_offset": g_result.offset,
        }
        for K in ks:
            r = kseg_results[K]
            well_row[f"K{K}_oracle_rmse"] = r.rmse
            pred_k = tvt_from_ksegment_offsets(
                s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, r.offsets
            )
            kseg_preds[K].extend(pred_k.tolist())
            kseg_true_grouped[K].extend(tvt_true.tolist())
        per_well.append(well_row)

    def _pooled_rmse(preds: list[float], trues: list[float]) -> float:
        p = np.array(preds, dtype=np.float64)
        t = np.array(trues, dtype=np.float64)
        valid = np.isfinite(p) & np.isfinite(t)
        if not np.any(valid):
            return float("nan")
        return float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))

    all_true_arr = np.array(all_true, dtype=np.float64)

    report: dict = {
        "n_wells": len(train_samples),
        "base_c0_rmse": _pooled_rmse(all_base, all_true),
        "global_grid_oracle_rmse": _pooled_rmse(all_global, all_true),
    }
    for K in ks:
        report[f"K{K}_oracle_rmse"] = _pooled_rmse(kseg_preds[K], kseg_true_grouped[K])

    report["per_well"] = per_well

    if verbose:
        print("\n=== Oracle Report ===")
        print(f"  n_wells              : {report['n_wells']}")
        print(f"  base_c0_rmse         : {report['base_c0_rmse']:.4f}")
        print(f"  global_grid_oracle   : {report['global_grid_oracle_rmse']:.4f}  (expect ~7.64)")
        for K in ks:
            expect = {1: "~7.59", 3: "~3.02", 5: "~1.82", 15: "~0.65"}.get(K, "?")
            print(f"  K{K:2d}_oracle          : {report[f'K{K}_oracle_rmse']:.4f}  (expect {expect})")

    return report
