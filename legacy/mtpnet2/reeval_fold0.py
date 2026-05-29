"""
reeval_fold0.py  —  Re-evaluate fold 0 checkpoint with fixed dp_decode (lam=5e-3).

Reproduces the exact fold-0 validation split from train_dlmtp.py (seed=42)
and re-runs inference on fold0_model.pt.
"""
from __future__ import annotations

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
CKPT_PATH  = "artifacts/dlmtp/run1/fold0_model.pt"

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
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    pad_multiple = 2 ** MODEL_CFG["depth"]

    print("Loading samples…")
    samples = load_offset_samples("", k_wells=0, cache_path=CACHE_PATH, verbose=False)

    print("Loading oof_k3…")
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

    # Reproduce exact valid_idx and fold split from train_dlmtp.py
    valid_idx = [i for i, s in enumerate(samples)
                 if s.well_id in tw_cache and s.has_true]
    n_valid   = len(valid_idx)
    print(f"  Valid wells: {n_valid}")

    rng      = np.random.default_rng(SEED)
    fold_ids = rng.integers(0, N_FOLDS, size=n_valid)

    val_mask    = fold_ids == 0
    val_local   = [i for i in range(n_valid) if val_mask[i]]
    val_samples = [samples[valid_idx[i]] for i in val_local]
    val_k3      = oof_k3[[valid_idx[i] for i in val_local]]
    print(f"  Fold 0 val wells: {len(val_samples)}")

    # Load model
    model = HeatmapUNet(**MODEL_CFG).to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  Loaded {CKPT_PATH}")
    print(f"  dp_decode lam = 5e-3  (fixed)")

    # Inference
    all_pred, all_true = [], []
    for vi, (s, k3) in enumerate(zip(val_samples, val_k3)):
        if s.well_id not in tw_cache:
            continue
        tw_tvt, tw_gr = tw_cache[s.well_id]
        prior_tvt     = compute_prior_tvt(s, k3)

        gr_smooth = _smooth_gr(s.gr.astype(float), CFG.get("gr_smooth_window", 101))
        heatmap, _ = build_inference_heatmap(
            s, prior_tvt, tw_tvt, tw_gr, CFG, pad_to=pad_multiple,
            gr_smooth_precomputed=gr_smooth,
        )
        x      = torch.from_numpy(heatmap).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x).squeeze(0).cpu().numpy()
        nh = len(s.hidden_rows)
        logits = logits[:nh]

        path = dp_decode(logits, **DP_CFG)
        pred = bins_to_tvt(path, prior_tvt[:nh], CFG["tvt_bins"], CFG["bin_ft"])
        true = s.tvt_true[s.hidden_rows]
        n2   = min(len(pred), len(true))
        all_pred.append(pred[:n2])
        all_true.append(true[:n2])

        if (vi + 1) % 20 == 0 or vi == len(val_samples) - 1:
            cur = row_rmse(np.concatenate(all_pred), np.concatenate(all_true))
            print(f"  {vi+1}/{len(val_samples)}  pool_rmse={cur:.4f} ft")

    final = row_rmse(np.concatenate(all_pred), np.concatenate(all_true))
    print(f"\n>>> Fold 0 RMSE (lam=5e-3) = {final:.4f} ft")
    print(f"    (was 95.1752 ft with broken decoder)")


if __name__ == "__main__":
    main()
