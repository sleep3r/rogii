"""
Experiment R — Regression-based offset prediction with residual learning.

Key differences from the earlier classifier approach:
  - Regression (not 321-class multiclass) eliminates hard-bin discretisation errors
  - Residual target: predict (oracle_offset - c0) instead of raw oracle_offset
    · Reduces target std from 0.0344 → 0.0121 ft/row (3x tighter signal)
    · Keeps model focused on *corrections* to the c0 heuristic
  - Stronger regularisation (min_data_in_leaf=50, lambda_l2=1.0)
    · Addresses 1.43x train/val overfit seen in baseline regression
  - Extended features (dC short-window slopes, autocorr, GR regime change)

Models trained:
  R_global  — single global offset per well
  R_K3      — K=3 segments, residual from per-segment c0
  R_K5      — K=5 segments, residual from per-segment c0

Results saved to artifacts/results/
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import lightgbm as lgb
import numpy as np

# -- project imports --
import sys
sys.path.insert(0, os.path.dirname(__file__))

from mtpnet.data   import load_offset_samples, make_group_kfold
from mtpnet.eval   import run_c0_baseline
from mtpnet.features import make_well_features, make_segment_features
from mtpnet.metrics  import row_rmse, per_well_rmse, print_scoreboard
from mtpnet.offsets  import (make_offset_grid, segment_assignment,
                              tvt_from_global_offset, tvt_from_ksegment_offsets)
from mtpnet.oracle   import compute_and_attach_oracles


# ---------------------------------------------------------------------------
# LightGBM regression wrapper
# ---------------------------------------------------------------------------

def _train_lgbm_regressor(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    num_leaves: int = 63,
    min_data_in_leaf: int = 50,
    lambda_l2: float = 1.0,
    n_estimators: int = 1000,
    lr: float = 0.05,
    early_stop: int = 50,
) -> lgb.Booster:
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    params = dict(
        objective        = "regression",
        n_jobs           = -1,
        verbosity        = -1,
        num_leaves       = num_leaves,
        min_data_in_leaf = min_data_in_leaf,
        lambda_l2        = lambda_l2,
        learning_rate    = lr,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        feature_fraction_bynode = 0.8,
    )
    cb = [lgb.early_stopping(early_stop, verbose=False)]
    bst = lgb.train(params, dtrain, n_estimators, valid_sets=[dval], callbacks=cb)
    return bst


# ---------------------------------------------------------------------------
# Global regression (K=1)
# ---------------------------------------------------------------------------

def run_global_regression(train: list, folds) -> dict:
    """OOF global regression with residual learning."""
    print("\n=== Experiment R_global — regression + residual learning ===")
    t0 = time.time()

    X_all    = make_well_features(train)
    oracle   = np.array([s.global_offset_star for s in train], dtype=np.float64)
    c0_arr   = np.array([s.c0                 for s in train], dtype=np.float64)
    y_resid  = (oracle - c0_arr).astype(np.float32)   # residual target

    print(f"  features: {X_all.shape[1]}  |  target std: {y_resid.std():.5f} ft/row "
          f"(oracle std: {oracle.std():.5f})")

    oof_preds_raw = np.zeros(len(train), dtype=np.float64)   # predicted residual
    fi = np.zeros(X_all.shape[1], dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tr_idx, va_idx = list(tr_idx), list(va_idx)
        bst = _train_lgbm_regressor(
            X_all[tr_idx], y_resid[tr_idx],
            X_all[va_idx], y_resid[va_idx],
        )
        oof_preds_raw[va_idx] = bst.predict(X_all[va_idx])
        fi += bst.feature_importance(importance_type="gain")
        print(f"    fold {fold_i}: best_iter={bst.best_iteration}")

    # Final prediction = c0 + predicted_residual
    oof_offset = c0_arr + oof_preds_raw

    # Decode to TVT and compute RMSE
    all_pred, all_true = [], []
    for i, s in enumerate(train):
        pred = tvt_from_global_offset(
            s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, float(oof_offset[i])
        )
        all_pred.append(pred.astype(np.float64))
        all_true.append(s.tvt_hidden_true.astype(np.float64))

    rmse = row_rmse(np.concatenate(all_pred), np.concatenate(all_true))
    elapsed = time.time() - t0
    print(f"\n  R_global OOF RMSE = {rmse:.4f} ft  ({elapsed:.0f}s)")

    # Feature importance (top 15)
    df_fi = sorted(zip(make_well_features([train[0]], return_df=True).columns,
                       fi / fi.sum()), key=lambda x: -x[1])
    print("  Top-15 features by gain:")
    for name, imp in df_fi[:15]:
        print(f"    {name:35s}: {imp:.4f}")

    return dict(
        experiment   = "R_global",
        oof_rmse     = float(rmse),
        oof_offset   = oof_offset.tolist(),
        elapsed_s    = elapsed,
        well_rmse    = per_well_rmse(all_pred, all_true, [s.well_id for s in train]),
    )


# ---------------------------------------------------------------------------
# K-segment regression
# ---------------------------------------------------------------------------

def _seg_c0(sample, k: int, K: int) -> float:
    """Approximate c0 for segment k: use c0 of the well (conservative fallback)."""
    # For early segments, the best local estimate is still c0 of the full prefix,
    # since we cannot observe the hidden section's drift.  A future improvement
    # would use depth-correlated sub-windows, but that requires knowing segment
    # depth range relative to the known suffix.
    return float(sample.c0)


def run_kseg_regression(train: list, folds, K: int) -> dict:
    """OOF K-segment regression with residual learning."""
    print(f"\n=== Experiment R_K{K} — K={K} segment regression + residual ===")
    t0 = time.time()

    X_seg = make_segment_features(train, K)

    # Per-segment oracle and c0
    oracle_flat = np.zeros(len(train) * K, dtype=np.float64)
    c0_flat     = np.zeros(len(train) * K, dtype=np.float64)
    for i, s in enumerate(train):
        offsets_k = s.kseg_offset_star.get(K, np.zeros(K))
        for k in range(K):
            oracle_flat[i * K + k] = offsets_k[k] if k < len(offsets_k) else offsets_k[-1]
            c0_flat[i * K + k]     = _seg_c0(s, k, K)

    y_resid = (oracle_flat - c0_flat).astype(np.float32)
    print(f"  features: {X_seg.shape[1]}  |  target std: {y_resid.std():.5f} ft/row")

    oof_resid = np.zeros(len(train) * K, dtype=np.float64)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tr_idx, va_idx = list(tr_idx), list(va_idx)
        tr_seg = np.concatenate([np.arange(i * K, i * K + K) for i in tr_idx])
        va_seg = np.concatenate([np.arange(i * K, i * K + K) for i in va_idx])
        bst = _train_lgbm_regressor(
            X_seg[tr_seg], y_resid[tr_seg],
            X_seg[va_seg], y_resid[va_seg],
        )
        oof_resid[va_seg] = bst.predict(X_seg[va_seg])
        print(f"    fold {fold_i}: best_iter={bst.best_iteration}")

    oof_offset = c0_flat + oof_resid   # final per-segment offset

    all_pred, all_true = [], []
    for i, s in enumerate(train):
        offsets = oof_offset[i * K: i * K + K]
        pred = tvt_from_ksegment_offsets(
            s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, offsets
        )
        all_pred.append(pred.astype(np.float64))
        all_true.append(s.tvt_hidden_true.astype(np.float64))

    rmse = row_rmse(np.concatenate(all_pred), np.concatenate(all_true))
    elapsed = time.time() - t0
    print(f"\n  R_K{K} OOF RMSE = {rmse:.4f} ft  ({elapsed:.0f}s)")

    return dict(
        experiment = f"R_K{K}",
        oof_rmse   = float(rmse),
        elapsed_s  = elapsed,
        well_rmse  = per_well_rmse(all_pred, all_true, [s.well_id for s in train]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default="/Users/alexander/Desktop/rogii/MTPNet/data/train")
    ap.add_argument("--cache-path", default="artifacts/mtpnet_cache/samples.pkl")
    ap.add_argument("--out-dir",    default="artifacts/results")
    ap.add_argument("--k-wells",    type=int, default=0, help="0=all wells")
    ap.add_argument("--ks",         nargs="+", type=int, default=[1, 3, 5],
                    help="Which K values to run (1=global, 3=K3, 5=K5)")
    ap.add_argument("--n-folds",    type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    samples = load_offset_samples(
        args.data_dir, k_wells=args.k_wells,
        cache_path=args.cache_path, verbose=True,
    )
    train = [s for s in samples if s.has_true]
    print(f"\n{len(train)} training wells loaded")

    # Oracle labels
    grid = make_offset_grid()
    ks_oracle = sorted(set(args.ks) | {1})   # always need K=1 for global
    compute_and_attach_oracles(train, ks=ks_oracle, grid=grid, verbose=False)

    folds = make_group_kfold(train, n_folds=args.n_folds)

    # Baselines
    c0_rmse = run_c0_baseline(train, verbose=False)
    print(f"\nBaseline c0 RMSE: {c0_rmse:.4f} ft")
    print(f"Oracle K=1: 7.5875  K=3: 3.0241  K=5: 1.8234")

    results = []

    # --- Global (K=1) ---
    if 1 in args.ks:
        res = run_global_regression(train, folds)
        results.append(res)
        out = os.path.join(args.out_dir, "R_global_regression.pkl")
        with open(out, "wb") as f:
            pickle.dump(res, f)
        print(f"  Saved → {out}")

    # --- K-segment ---
    for K in args.ks:
        if K == 1:
            continue
        if K not in [s.kseg_offset_star for s in train[:1]][0]:
            compute_and_attach_oracles(train, ks=[K], grid=grid, verbose=False)
        res = run_kseg_regression(train, folds, K)
        results.append(res)
        out = os.path.join(args.out_dir, f"R_K{K}_regression.pkl")
        with open(out, "wb") as f:
            pickle.dump(res, f)
        print(f"  Saved → {out}")

    # --- Scoreboard ---
    print("\n" + "=" * 55)
    print("SCOREBOARD")
    print("=" * 55)
    print(f"  {'c0_baseline':40s}: {c0_rmse:.4f} ft")
    for res in results:
        print(f"  {res['experiment']:40s}: {res['oof_rmse']:.4f} ft")
    print(f"  {'oracle_K1':40s}: 7.5875 ft")
    print(f"  {'oracle_K3':40s}: 3.0241 ft")
    print(f"  {'oracle_K5':40s}: 1.8234 ft")


if __name__ == "__main__":
    main()
