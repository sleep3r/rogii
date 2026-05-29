"""
eval_gr_path_scorer.py — OOF evaluation of the GR-guided local path scorer.

Reports TWO metrics to allow direct comparison:
  (A) per-well RMSE mean   — matches hengck23's "12.18" style number
  (B) pooled row-RMSE      — primary MTPNet metric (matches leaderboard)

Hyperparameter sweep over:
  lookahead    : [50, 75, 100, 150, 200]
  commit_frac  : [0.20, 0.30, 0.50, 1.00]

Usage:
    python eval_gr_path_scorer.py
    python eval_gr_path_scorer.py --lookahead 100 --commit-frac 0.20  # single run
    python eval_gr_path_scorer.py --sweep                              # full sweep
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from mtpnet.data         import load_offset_samples
from mtpnet.local_search import (gr_path_score_brute, load_typewell,
                                  make_local_offset_grid)
from mtpnet.metrics      import row_rmse

DATA_DIR   = "/Users/alexander/Desktop/rogii/MTPNet/data/train"
CACHE_PATH = "artifacts/mtpnet_cache/samples.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def per_well_mean_rmse(preds: list[np.ndarray], trues: list[np.ndarray]) -> float:
    """Mean of per-well RMSE — matches hengck23's metric."""
    rmses = []
    for p, t in zip(preds, trues):
        if len(p) > 0 and len(t) > 0:
            n = min(len(p), len(t))
            rmses.append(float(np.sqrt(np.mean((p[:n] - t[:n]) ** 2))))
    return float(np.mean(rmses)) if rmses else float("nan")


def run_one(
    train: list,
    offset_grid: np.ndarray,
    *,
    lookahead: int,
    commit_frac: float,
    max_wells: int = 0,
    verbose: bool = True,
) -> dict:
    samples = train[:max_wells] if max_wells else train
    all_pred, all_true, well_rmse_list = [], [], []

    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            tw_tvt, tw_gr = load_typewell(s.well_id, DATA_DIR)
        except FileNotFoundError:
            if verbose:
                print(f"  [SKIP] no typewell for {s.well_id}")
            continue

        pred = gr_path_score_brute(
            s, tw_tvt, tw_gr, offset_grid,
            lookahead=lookahead,
            commit_frac=commit_frac,
        )
        true = s.tvt_hidden_true

        n = min(len(pred), len(true))
        if n == 0:
            continue

        all_pred.append(pred[:n])
        all_true.append(true[:n])
        wr = float(np.sqrt(np.mean((pred[:n] - true[:n]) ** 2)))
        well_rmse_list.append(wr)

        if verbose and (i % 50 == 0 or i == len(samples) - 1):
            elapsed = time.time() - t0
            print(f"  [{i+1:4d}/{len(samples)}]  well={s.well_id}  "
                  f"rmse={wr:.2f} ft  elapsed={elapsed:.0f}s")

    pooled = row_rmse(np.concatenate(all_pred), np.concatenate(all_true))
    perwell = per_well_mean_rmse(all_pred, all_true)

    return dict(
        lookahead=lookahead,
        commit_frac=commit_frac,
        n_wells=len(all_pred),
        pooled_row_rmse=pooled,
        perwell_mean_rmse=perwell,
        elapsed_s=time.time() - t0,
        well_rmse_p50=float(np.median(well_rmse_list)),
        well_rmse_p90=float(np.percentile(well_rmse_list, 90)),
    )


# ---------------------------------------------------------------------------
# Sign-convention sanity check (first 5 known-section rows)
# ---------------------------------------------------------------------------

def sign_check(train: list, n: int = 5) -> None:
    """Verify dtvt = -dZ + offset reproduces known TVT on a few rows."""
    s = train[0]
    z  = s.z.astype(np.float64)
    from mtpnet.data import load_offset_samples
    # load raw csv to get ground truth TVT for known rows
    csv_path = os.path.join(DATA_DIR, f"{s.well_id}__horizontal_well.csv")
    df = pd.read_csv(csv_path, usecols=["TVT", "TVT_input", "Z"])
    tvt_raw = df["TVT"].values
    tvt_inp = df["TVT_input"].values

    known_idx = np.flatnonzero(np.isfinite(tvt_inp))
    if len(known_idx) < n + 2:
        print("  sign_check: not enough known rows")
        return

    last_row = known_idx[-1]
    anchor_tvt = float(tvt_inp[last_row])

    # Project forward 5 rows using c0 (expected to be close to truth)
    c0 = s.c0
    dz = np.diff(z[last_row: last_row + n + 1])
    tvt_pred = anchor_tvt + np.cumsum(-dz + c0)
    tvt_true = tvt_raw[last_row + 1: last_row + 1 + n]

    print(f"\n  Sign-convention check (well={s.well_id}, c0={c0:.5f} ft/row):")
    print(f"  {'row':>4}  {'pred':>10}  {'true':>10}  {'err':>8}")
    for i in range(min(n, len(tvt_true))):
        err = tvt_pred[i] - tvt_true[i] if i < len(tvt_true) else float("nan")
        print(f"  {last_row+1+i:4d}  {tvt_pred[i]:10.3f}  "
              f"{tvt_true[i]:10.3f}  {err:+8.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookahead",    type=int,   default=100)
    ap.add_argument("--commit-frac",  type=float, default=0.20)
    ap.add_argument("--sweep",        action="store_true",
                    help="Run full lookahead × commit_frac grid")
    ap.add_argument("--max-wells",    type=int,   default=0,
                    help="Limit to N wells for quick tests (0=all)")
    ap.add_argument("--offset-lo",    type=float, default=-0.30)
    ap.add_argument("--offset-hi",    type=float, default= 0.30)
    ap.add_argument("--offset-bins",  type=int,   default=121)
    args = ap.parse_args()

    print("Loading samples …")
    samples = load_offset_samples(
        DATA_DIR, k_wells=0,
        cache_path=CACHE_PATH, verbose=False,
    )
    train = [s for s in samples if s.has_true]
    print(f"{len(train)} training wells")

    # Sign convention sanity check
    sign_check(train)

    offset_grid = make_local_offset_grid(args.offset_lo, args.offset_hi, args.offset_bins)
    print(f"Offset grid: {len(offset_grid)} bins  "
          f"[{offset_grid[0]:.3f}, {offset_grid[-1]:.3f}] ft/row")

    if args.sweep:
        lookaheads    = [50, 75, 100, 150, 200]
        commit_fracs  = [0.20, 0.30, 0.50, 1.00]
        combos = list(product(lookaheads, commit_fracs))
    else:
        combos = [(args.lookahead, args.commit_frac)]

    results = []
    for la, cf in combos:
        print(f"\n{'='*60}")
        print(f"lookahead={la}  commit_frac={cf:.2f}")
        print(f"{'='*60}")
        res = run_one(
            train, offset_grid,
            lookahead=la, commit_frac=cf,
            max_wells=args.max_wells,
            verbose=(len(combos) == 1),
        )
        results.append(res)
        print(f"  pooled row-RMSE    = {res['pooled_row_rmse']:.4f} ft")
        print(f"  per-well mean-RMSE = {res['perwell_mean_rmse']:.4f} ft  "
              f"(p50={res['well_rmse_p50']:.2f}, p90={res['well_rmse_p90']:.2f})")
        print(f"  elapsed            = {res['elapsed_s']:.0f}s  "
              f"({res['n_wells']} wells)")

    print(f"\n{'='*60}")
    print("SCOREBOARD  (reference)")
    print(f"{'='*60}")
    print(f"  {'c0_baseline':40s}: 39.03 ft")
    print(f"  {'global_regression (K=1)':40s}: 15.19 ft  pooled")
    print(f"  {'K=3_regression_ensemble5':40s}: 13.73 ft  pooled")
    print(f"  {'oracle_K=1':40s}:  7.59 ft  pooled")
    print(f"  {'oracle_K=3':40s}:  3.02 ft  pooled")
    print()
    if len(results) > 1:
        df = pd.DataFrame(results)
        print(df[["lookahead","commit_frac","pooled_row_rmse","perwell_mean_rmse",
                  "elapsed_s"]].to_string(index=False))

    # Save results
    os.makedirs("artifacts/results", exist_ok=True)
    out = "artifacts/results/gr_path_scorer.pkl"
    import pickle
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
