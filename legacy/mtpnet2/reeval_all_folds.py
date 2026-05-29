"""
reeval_all_folds.py  —  Re-evaluate all Run 1 fold checkpoints with fixed dp_decode.

Root cause of 95 ft bug:
  - Forward DP shift direction was inverted (src[j]=prev[j+delta] instead of prev[j-delta])
  - Traceback pointers were consistent with the wrong direction → path mirrored to edge
  - Fix: swap the array assignment branches in dp_decode.py

Usage:
    python3 reeval_all_folds.py [--run run1] [--folds 5]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from mtpnet.data      import load_offset_samples
from mtpnet.metrics   import row_rmse
from dlmtp.dataset    import build_inference_heatmap, compute_prior_tvt
from dlmtp.heatmap    import _smooth_gr
from dlmtp.unet       import HeatmapUNet
from dlmtp.dp_decode  import dp_decode, bins_to_tvt
from mtpnet.local_search import load_typewell as _load_tw_raw

DATA_DIR   = "/Users/alexander/Desktop/rogii/MTPNet/data/train"
CACHE_PATH = "artifacts/mtpnet_cache/samples.pkl"
OOF_K3_PATH= "artifacts/results/oof_k3.pkl"

CFG = dict(
    crop_len         = 256,
    tvt_bins         = 96,
    bin_ft           = 2.0,
    gr_smooth_window = 101,
    n_crops          = 6,
)
MODEL_CFG = dict(in_ch=8, base=16, depth=3)
DP_CFG    = dict(lambda_smooth=0.05, max_trans=8)
N_FOLDS   = 5
SEED      = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",   default="run1")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(f"artifacts/dlmtp/{args.run}")
    device  = (torch.device("mps") if torch.backends.mps.is_available()
               else torch.device("cpu"))
    pad_multiple = 2 ** MODEL_CFG["depth"]

    print(f"Re-evaluating {args.run}  ({args.folds} folds)  device={device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading samples…")
    samples = load_offset_samples("", k_wells=0, cache_path=CACHE_PATH, verbose=False)
    with open(OOF_K3_PATH, "rb") as f:
        oof_k3 = pickle.load(f)["oof_k3"]

    print("Loading typewell cache…")
    tw_cache = {}
    for s in samples:
        try:
            tw_cache[s.well_id] = _load_tw_raw(s.well_id, DATA_DIR)
        except Exception:
            pass
    print(f"  {len(tw_cache)} typewells loaded")

    # ── Reproduce fold split ────────────────────────────────────────────────
    valid_idx = [i for i, s in enumerate(samples)
                 if s.well_id in tw_cache and s.has_true]
    n_valid   = len(valid_idx)
    print(f"  Valid wells: {n_valid}")

    rng      = np.random.default_rng(SEED)
    fold_ids = rng.integers(0, N_FOLDS, size=n_valid)

    # ── Per-fold eval ───────────────────────────────────────────────────────
    oof_preds: dict[int, np.ndarray] = {}
    oof_trues: dict[int, np.ndarray] = {}

    for fold in range(args.folds):
        ckpt = run_dir / f"fold{fold}_model.pt"
        if not ckpt.exists():
            print(f"  [skip] fold {fold}: checkpoint not found ({ckpt})")
            continue

        val_local   = [i for i in range(n_valid) if fold_ids[i] == fold]
        val_samples = [samples[valid_idx[i]] for i in val_local]
        val_k3      = oof_k3[[valid_idx[i] for i in val_local]]
        val_gidx    = [valid_idx[i] for i in val_local]   # global sample indices

        print(f"\n── Fold {fold+1}/{args.folds}  (val={len(val_samples)}) ──")
        model = HeatmapUNet(**MODEL_CFG).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f"  Loaded {ckpt}")

        ap, at = [], []
        for vi, (s, k3) in enumerate(zip(val_samples, val_k3)):
            if s.well_id not in tw_cache:
                continue
            tw_tvt, tw_gr = tw_cache[s.well_id]
            prior_tvt     = compute_prior_tvt(s, k3)
            nh            = len(s.hidden_rows)

            gr_smooth = _smooth_gr(s.gr.astype(float), CFG.get("gr_smooth_window", 101))
            heatmap, _ = build_inference_heatmap(
                s, prior_tvt, tw_tvt, tw_gr, CFG, pad_to=pad_multiple,
                gr_smooth_precomputed=gr_smooth,
            )
            x = torch.from_numpy(heatmap).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x).squeeze(0).cpu().numpy()[:nh]

            path = dp_decode(logits, **DP_CFG)
            pred = bins_to_tvt(path, prior_tvt[:nh], CFG["tvt_bins"], CFG["bin_ft"])
            true = s.tvt_true[s.hidden_rows]
            n2   = min(len(pred), len(true))

            ap.append(pred[:n2]); at.append(true[:n2])
            oof_preds[val_gidx[vi]] = pred[:n2]
            oof_trues[val_gidx[vi]] = true[:n2]

            if (vi + 1) % 20 == 0 or vi == len(val_samples) - 1:
                cur = row_rmse(np.concatenate(ap), np.concatenate(at))
                print(f"    {vi+1}/{len(val_samples)}  pool_val={cur:.4f}", flush=True)

        fold_rmse = row_rmse(np.concatenate(ap), np.concatenate(at))
        print(f"  ✓ Fold {fold+1} RMSE = {fold_rmse:.4f} ft")

    if not oof_preds:
        print("No completed folds found.")
        return

    # ── Partial or full OOF result ──────────────────────────────────────────
    all_pred = np.concatenate([oof_preds[i] for i in sorted(oof_preds)])
    all_true = np.concatenate([oof_trues[i] for i in sorted(oof_trues)])
    oof_rmse = row_rmse(all_pred, all_true)
    n_wells  = len(oof_preds)

    buckets = {"short": [], "medium": [], "long": [], "xlong": []}
    per_well = []
    for gi in sorted(oof_preds):
        s  = samples[gi]
        p  = oof_preds[gi]; t = oof_trues[gi]
        pw = float(np.sqrt(np.mean((p - t) ** 2)))
        per_well.append(pw)
        nh = len(s.hidden_rows)
        bk = ("xlong" if nh >= 8000 else "long" if nh >= 5000
              else "medium" if nh >= 2000 else "short")
        buckets[bk].append((p, t))

    print(f"\n{'='*50}")
    print(f"Partial OOF ({n_wells}/{n_valid} wells)  pooled RMSE = {oof_rmse:.4f} ft")
    print(f"Per-well mean RMSE = {np.mean(per_well):.4f} ft")
    for bk, pairs in buckets.items():
        if pairs:
            bp = np.concatenate([p for p, _ in pairs])
            bt = np.concatenate([t for _, t in pairs])
            print(f"  {bk:7s} ({len(pairs):3d} wells): {row_rmse(bp, bt):.3f} ft")
    print(f"{'='*50}\n")

    # Save
    out_path = run_dir / "oof_reeval.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"oof_preds": oof_preds, "oof_trues": oof_trues,
                     "oof_rmse": oof_rmse, "n_wells": n_wells}, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
