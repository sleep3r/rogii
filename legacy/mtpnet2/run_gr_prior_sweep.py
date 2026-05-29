"""
Шаг 4: GR path scorer + MTPNet prior penalty.

score = gr_rmse_z
      + λ_offset * |offset - k3_pred_for_s0|
      + λ_smooth * |offset - prev_offset|
      + λ_range  * out_of_typewell_range_frac

gr_rmse_z = (gr_rmse - median_candidates) / iqr_candidates   (per step)

K=3 OOF offsets re-computed here with correct hyperparams
  (min_data_in_leaf=5, lambda_l2=0, num_leaves=63).
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import lightgbm as lgb
from scipy.stats import iqr as scipy_iqr

from mtpnet.data      import load_offset_samples, make_group_kfold
from mtpnet.features  import make_segment_features
from mtpnet.offsets   import (make_offset_grid, expand_segment_offsets,
                               tvt_from_ksegment_offsets)
from mtpnet.oracle    import compute_and_attach_oracles
from mtpnet.local_search import load_typewell, smooth_gr
from mtpnet.metrics   import row_rmse

DATA_DIR   = "/Users/alexander/Desktop/rogii/MTPNet/data/train"
CACHE_PATH = "artifacts/mtpnet_cache/samples.pkl"
K = 3

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load data & oracle labels
# ──────────────────────────────────────────────────────────────────────────────
print("Loading samples …")
samples = load_offset_samples('', k_wells=0, cache_path=CACHE_PATH, verbose=False)
train   = [s for s in samples if s.has_true]
grid    = make_offset_grid()
compute_and_attach_oracles(train, ks=[1, K], grid=grid, verbose=False)
print(f"  {len(train)} training wells")

folds = make_group_kfold(train, n_folds=5)

# ──────────────────────────────────────────────────────────────────────────────
# 2. K=3 OOF per-segment offset predictions (correct hyperparams)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nTraining K={K} LGB regression (min_data_in_leaf=5, lambda_l2=0) …")

X_seg = make_segment_features(train, K)
# raw oracle target (confirmed better than residual in ablation)
oracle_flat = np.array([
    (s.kseg_offset_star[K][k] if k < len(s.kseg_offset_star[K]) else s.kseg_offset_star[K][-1])
    for s in train for k in range(K)
], dtype=np.float32)

oof_raw = np.zeros(len(train) * K, dtype=np.float64)
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    tr_idx, va_idx = list(tr_idx), list(va_idx)
    tr_seg = np.concatenate([np.arange(i*K, i*K+K) for i in tr_idx])
    va_seg = np.concatenate([np.arange(i*K, i*K+K) for i in va_idx])
    dtrain = lgb.Dataset(X_seg[tr_seg], label=oracle_flat[tr_seg])
    dval   = lgb.Dataset(X_seg[va_seg], label=oracle_flat[va_seg], reference=dtrain)
    params = dict(objective='regression', n_jobs=-1, verbosity=-1,
                  num_leaves=63, min_data_in_leaf=5, lambda_l2=0.0,
                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    bst = lgb.train(params, dtrain, 2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_raw[va_seg] = bst.predict(X_seg[va_seg])
    print(f"  fold {fold_i}: best_iter={bst.best_iteration}")

# per-well K-segment OOF offsets: shape (n_wells, K)
oof_offsets_k3 = oof_raw.reshape(len(train), K)

# quick sanity – regression OOF pooled RMSE
all_pred_r, all_true_r = [], []
for i, s in enumerate(train):
    pred = tvt_from_ksegment_offsets(
        s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, oof_offsets_k3[i])
    all_pred_r.append(pred.astype(np.float64))
    all_true_r.append(s.tvt_hidden_true.astype(np.float64))
lgb_k3_rmse = row_rmse(np.concatenate(all_pred_r), np.concatenate(all_true_r))
print(f"\nK=3 regression OOF pooled RMSE: {lgb_k3_rmse:.4f} ft  (target ≈ 13.73)")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Prior-augmented GR path scorer
# ──────────────────────────────────────────────────────────────────────────────

def gr_prior_score(sample, tw_tvt, tw_gr, offset_grid,
                   k3_segs,          # (K,) predicted offsets for this well
                   *,
                   lookahead=200, commit_frac=0.20,
                   lam_offset=0.0, lam_smooth=0.0, lam_range=0.0,
                   gr_smooth_window=101):
    """GR path scorer with MTPNet prior penalty."""
    hidden_rows = sample.hidden_rows
    n_hidden    = len(hidden_rows)
    if n_hidden == 0:
        return np.array([], dtype=np.float64)

    z         = sample.z.astype(np.float64)
    gr_smooth = smooth_gr(sample.gr.astype(np.float64), window=gr_smooth_window)

    tw_min, tw_max = float(tw_tvt.min()), float(tw_tvt.max())
    last_tvt   = float(sample.anchor_tvt)
    prev_offset = float(k3_segs[0])   # initialise smooth penalty anchor

    n_off   = len(offset_grid)
    predict = np.empty(n_hidden, dtype=np.float64)

    pos = 0
    while pos < n_hidden:
        end_pos = min(pos + lookahead, n_hidden)
        seg_hi  = hidden_rows[pos:end_pos]
        n_seg   = len(seg_hi)

        # which K=3 segment covers pos?
        seg_idx = int(np.clip(pos * K // n_hidden, 0, K - 1))
        mtpnet_prior = float(k3_segs[seg_idx])

        # dZ for segment
        prev_row = hidden_rows[pos - 1] if pos > 0 else sample.anchor_row
        dz_seg   = np.diff(np.concatenate([[z[prev_row]], z[seg_hi]]))

        # integrate all candidates:  (n_off, n_seg)
        dtvt    = -dz_seg[None, :] + offset_grid[:, None]
        tvt_can = last_tvt + np.cumsum(dtvt, axis=1)

        # GR score (RMSE)
        gr_can  = np.interp(tvt_can.ravel(), tw_tvt, tw_gr).reshape(n_off, n_seg)
        gr_rmse = np.sqrt(np.mean((gr_can - gr_smooth[seg_hi][None, :]) ** 2, axis=1))

        # normalise gr_rmse → z-score within this step
        med = float(np.median(gr_rmse))
        iq  = float(scipy_iqr(gr_rmse))
        if iq < 1e-9:
            gr_rmse_z = np.zeros(n_off)
        else:
            gr_rmse_z = (gr_rmse - med) / iq

        # out-of-typewell-range penalty
        if lam_range > 0:
            out_frac = np.mean(
                (tvt_can < tw_min) | (tvt_can > tw_max), axis=1
            ).astype(np.float64)
        else:
            out_frac = np.zeros(n_off)

        # combined score
        offset_penalty = np.abs(offset_grid - mtpnet_prior)
        smooth_penalty = np.abs(offset_grid - prev_offset)
        score = (gr_rmse_z
                 + lam_offset * offset_penalty
                 + lam_smooth * smooth_penalty
                 + lam_range  * out_frac)

        best_j = int(np.argmin(score))
        n_commit = max(1, int(commit_frac * n_seg))
        predict[pos: pos + n_commit] = tvt_can[best_j, :n_commit]
        last_tvt    = float(tvt_can[best_j, n_commit - 1])
        prev_offset = float(offset_grid[best_j])
        pos += n_commit

    return predict


def eval_config(lam_offset, lam_smooth, lam_range, la=200, cf=0.20,
                lo=-0.16, hi=0.16, n_bins=65):
    off_grid = np.linspace(lo, hi, n_bins)
    all_pred, all_true, pw = [], [], []
    for i, s in enumerate(train):
        try:
            tw_tvt, tw_gr = load_typewell(s.well_id, DATA_DIR)
        except FileNotFoundError:
            continue
        pred = gr_prior_score(s, tw_tvt, tw_gr, off_grid, oof_offsets_k3[i],
                              lookahead=la, commit_frac=cf,
                              lam_offset=lam_offset,
                              lam_smooth=lam_smooth,
                              lam_range=lam_range)
        true = s.tvt_hidden_true
        n = min(len(pred), len(true))
        all_pred.append(pred[:n]); all_true.append(true[:n])
        pw.append(float(np.sqrt(np.mean((pred[:n]-true[:n])**2))))
    pooled = row_rmse(np.concatenate(all_pred), np.concatenate(all_true))
    return pooled, float(np.mean(pw))

# ──────────────────────────────────────────────────────────────────────────────
# 4. λ sweep
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== λ_offset sweep  (la=200, cf=0.20, ±0.16 65-bin grid) ===")
print(f"{'λ_off':>8} {'λ_smo':>7} {'λ_rng':>7} {'pooled':>9} {'pw_mean':>9}")
print("-" * 46)

results = []
for lam_o in [0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    t0 = time.time()
    pooled, pw_mean = eval_config(lam_o, 0.0, 0.0)
    results.append((pooled, lam_o, 0.0, 0.0))
    print(f"  {lam_o:6.1f}  {'0.0':>6}  {'0.0':>6}  {pooled:8.4f}  {pw_mean:8.4f}  ({time.time()-t0:.0f}s)")

best_lam_o = sorted(results)[0][1]
print(f"\nBest λ_offset so far: {best_lam_o}")

print(f"\n=== λ_smooth sweep  (λ_offset={best_lam_o}) ===")
print(f"{'λ_off':>8} {'λ_smo':>7} {'λ_rng':>7} {'pooled':>9} {'pw_mean':>9}")
print("-" * 46)
results2 = []
for lam_s in [0, 0.5, 1.0, 2.0]:
    t0 = time.time()
    pooled, pw_mean = eval_config(best_lam_o, lam_s, 0.0)
    results2.append((pooled, best_lam_o, lam_s, 0.0))
    print(f"  {best_lam_o:6.1f}  {lam_s:6.1f}  {'0.0':>6}  {pooled:8.4f}  {pw_mean:8.4f}  ({time.time()-t0:.0f}s)")

best_lam_s = sorted(results2)[0][2]

print(f"\n=== λ_range sweep  (λ_offset={best_lam_o}, λ_smooth={best_lam_s}) ===")
results3 = []
for lam_r in [0, 10.0, 100.0]:
    t0 = time.time()
    pooled, pw_mean = eval_config(best_lam_o, best_lam_s, lam_r)
    results3.append((pooled, best_lam_o, best_lam_s, lam_r))
    print(f"  {best_lam_o:6.1f}  {best_lam_s:6.1f}  {lam_r:6.0f}  {pooled:8.4f}  {pw_mean:8.4f}  ({time.time()-t0:.0f}s)")

all_results = sorted(results + results2 + results3)

print("\n=== SCOREBOARD ===")
print(f"  {'c0_baseline':40s}:  39.03 ft pooled")
print(f"  {'K=3 LGB regression (re-run)':40s}:  {lgb_k3_rmse:.4f} ft pooled")
print(f"  {'K=3 regression ensemble(5) [prev best]':40s}:  13.73 ft pooled")
print(f"  {'GR scorer c0±0.06 la=200 (baseline)':40s}:  14.81 ft pooled")
print(f"  {'oracle K=1':40s}:   7.59 ft pooled")
print()
print("  Best GR+prior configs:")
for pooled, lo, ls, lr in all_results[:5]:
    print(f"    λ_off={lo}  λ_smo={ls}  λ_rng={lr}  →  {pooled:.4f} ft pooled")
