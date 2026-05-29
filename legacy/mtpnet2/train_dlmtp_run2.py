"""
train_dlmtp_run2.py  —  Run 2: +greedy-GR prior channel (9 channels).

Differences from Run 1:
  - Loads artifacts/results/oof_gr_tvt.pkl  →  gr_tvt_cache
  - Passes gr_tvt_cache to HeatmapDataset and predict_well
  - U-Net in_ch=9 instead of 8
  - Saves to artifacts/dlmtp/run2/

Usage:
    python train_dlmtp_run2.py [--quick] [--folds 5] [--epochs 50]
"""
from __future__ import annotations
import sys, argparse, pickle
sys.path.insert(0, '.')
import numpy as np

# Monkey-patch MODEL_CFG before importing main
import train_dlmtp as _base

# Re-use everything from train_dlmtp, override specific parts

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mtpnet.data       import load_offset_samples
from mtpnet.metrics    import row_rmse
from dlmtp.dataset     import HeatmapDataset, build_inference_heatmap, compute_prior_tvt
from dlmtp.heatmap     import _smooth_gr
from dlmtp.unet        import HeatmapUNet
from dlmtp.losses      import gaussian_ce_loss, path_smooth_loss
from dlmtp.dp_decode   import dp_decode, bins_to_tvt
from train_dlmtp       import (
    DATA_DIR, CACHE_PATH, OOF_K3_PATH,
    DEFAULT_CFG, TRAIN_CFG, DP_CFG,
    pad_to, train_one_epoch,
    _load_typewell as _load_tw_raw,
)
import time

GR_TVT_PATH = "artifacts/results/oof_gr_tvt.pkl"
OUT_DIR     = Path("artifacts/dlmtp/run2")

MODEL_CFG_R2 = dict(in_ch=9, base=16, depth=3)


@torch.no_grad()
def predict_well_r2(model, sample, prior_tvt, tw_tvt, tw_gr, cfg, dp_cfg,
                    device, pad_multiple, gr_prior_tvt=None):
    model.eval()
    nh = len(sample.hidden_rows)
    gr_smooth = _smooth_gr(sample.gr.astype(float), cfg.get("gr_smooth_window", 101))
    heatmap, _ = build_inference_heatmap(
        sample, prior_tvt, tw_tvt, tw_gr, cfg, pad_to=pad_multiple,
        gr_smooth_precomputed=gr_smooth, gr_prior_tvt=gr_prior_tvt,
    )
    x      = torch.from_numpy(heatmap).unsqueeze(0).to(device)
    logits = model(x).squeeze(0).cpu().numpy()[:nh]
    path   = dp_decode(logits, **dp_cfg)
    return bins_to_tvt(path, prior_tvt[:nh], cfg["tvt_bins"], cfg["bin_ft"])


def run_oof_r2(samples, oof_k3, tw_cache, gr_tvt_cache, cfg,
               model_cfg, train_cfg, dp_cfg, n_folds=5, seed=42,
               save_dir=OUT_DIR):
    save_dir.mkdir(parents=True, exist_ok=True)
    device       = (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu"))
    pad_multiple = 2 ** model_cfg["depth"]

    print(f"\n{'='*60}")
    print(f"DL-MTP Run 2  |  device={device}  |  {n_folds}-fold OOF")
    print(f"  U-Net in_ch={model_cfg['in_ch']} base={model_cfg['base']}")
    print(f"  +greedy-GR prior channel")
    print(f"  crop={cfg['crop_len']}×{cfg['tvt_bins']}  epochs={train_cfg['epochs']}")
    print(f"{'='*60}\n")

    valid_idx = [i for i, s in enumerate(samples)
                 if s.well_id in tw_cache and s.has_true]
    n_valid   = len(valid_idx)
    print(f"  Valid wells: {n_valid}")
    batches_per_epoch = int(n_valid * 0.8 * cfg["n_crops"] / train_cfg["batch_size"])
    est_min = batches_per_epoch * 0.045 * train_cfg["epochs"] * n_folds / 60
    print(f"  Batches/epoch ≈ {batches_per_epoch}  →  est. {est_min:.0f} min")

    rng      = np.random.default_rng(seed)
    fold_ids = rng.integers(0, n_folds, size=n_valid)

    oof_preds, oof_trues = {}, {}
    t_start = time.time()

    for fold in range(n_folds):
        fold_t0 = time.time()
        val_mask = fold_ids == fold
        tr_mask  = ~val_mask
        tr_idxs  = [valid_idx[i] for i in range(n_valid) if tr_mask[i]]
        val_idxs = [valid_idx[i] for i in range(n_valid) if val_mask[i]]
        tr_samples  = [samples[i] for i in tr_idxs]
        val_samples = [samples[i] for i in val_idxs]
        tr_k3       = oof_k3[tr_idxs]
        val_k3      = oof_k3[val_idxs]

        print(f"\n── Fold {fold+1}/{n_folds}  (train={len(tr_idxs)} val={len(val_idxs)}) ──")

        tr_ds = HeatmapDataset(tr_samples, tr_k3, tw_cache, cfg,
                               n_crops=cfg["n_crops"], train=True,
                               gr_tvt_cache=gr_tvt_cache)
        tr_dl = DataLoader(tr_ds, batch_size=train_cfg["batch_size"],
                           shuffle=True, num_workers=2, drop_last=True,
                           persistent_workers=True)

        model = HeatmapUNet(**model_cfg).to(device)
        if fold == 0:
            print(f"  Model params: {model.n_params():,}")

        opt   = torch.optim.AdamW(model.parameters(),
                                  lr=train_cfg["lr"], weight_decay=train_cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=train_cfg["epochs"], eta_min=train_cfg["lr"] * 0.1)

        for ep in range(1, train_cfg["epochs"] + 1):
            ep_loss = train_one_epoch(model, tr_dl, opt, device, train_cfg,
                                      pad_multiple)
            sched.step()
            if ep % train_cfg["log_every"] == 0 or ep == train_cfg["epochs"]:
                elapsed = time.time() - fold_t0
                eta     = elapsed / ep * (train_cfg["epochs"] - ep)
                print(f"    ep {ep:3d}/{train_cfg['epochs']}  loss={ep_loss:.4f}"
                      f"  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        print(f"  Evaluating {len(val_idxs)} val wells…", flush=True)
        ap, at = [], []
        for vi, (s, k3) in enumerate(zip(val_samples, val_k3)):
            if s.well_id not in tw_cache:
                continue
            tw_tvt, tw_gr = tw_cache[s.well_id]
            prior_tvt     = compute_prior_tvt(s, k3)
            gr_prior      = gr_tvt_cache.get(s.well_id) if gr_tvt_cache else None
            pred = predict_well_r2(model, s, prior_tvt, tw_tvt, tw_gr,
                                   cfg, dp_cfg, device, pad_multiple, gr_prior)
            true = s.tvt_true[s.hidden_rows]
            n2   = min(len(pred), len(true))
            ap.append(pred[:n2]); at.append(true[:n2])
            oof_preds[val_idxs[vi]] = pred[:n2]
            oof_trues[val_idxs[vi]] = true[:n2]
            if (vi + 1) % 20 == 0 or vi == len(val_idxs) - 1:
                cur = row_rmse(np.concatenate(ap), np.concatenate(at))
                print(f"    val {vi+1}/{len(val_idxs)}  pool_val={cur:.4f}", flush=True)

        fold_rmse = row_rmse(np.concatenate(ap), np.concatenate(at))
        print(f"  ✓ Fold {fold+1} RMSE = {fold_rmse:.4f} ft  ({time.time()-fold_t0:.0f}s)")
        torch.save(model.state_dict(), save_dir / f"fold{fold}_model.pt")

    all_pred = np.concatenate([oof_preds[i] for i in sorted(oof_preds)])
    all_true = np.concatenate([oof_trues[i] for i in sorted(oof_trues)])
    oof_rmse = row_rmse(all_pred, all_true)

    buckets = {"short": [], "medium": [], "long": [], "xlong": []}
    per_well = []
    for i in sorted(oof_preds):
        s  = samples[i]; p = oof_preds[i]; t = oof_trues[i]
        pw = float(np.sqrt(np.mean((p - t) ** 2)))
        per_well.append(pw)
        bk = ("xlong" if len(s.hidden_rows) >= 8000 else
              "long"  if len(s.hidden_rows) >= 5000 else
              "medium" if len(s.hidden_rows) >= 2000 else "short")
        buckets[bk].append((p, t))

    print(f"\n{'='*60}")
    print(f"OOF pooled RMSE : {oof_rmse:.4f} ft")
    print(f"Per-well mean   : {np.mean(per_well):.4f} ft")
    for bk, pairs in buckets.items():
        if pairs:
            bp = np.concatenate([p for p, _ in pairs])
            bt = np.concatenate([t for _, t in pairs])
            print(f"  {bk:7s} ({len(pairs):3d} wells): {row_rmse(bp, bt):.3f} ft")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}\n")

    with open(save_dir / "oof_predictions.pkl", "wb") as f:
        pickle.dump({"oof_preds": oof_preds, "oof_trues": oof_trues,
                     "oof_rmse": oof_rmse}, f)
    print(f"Saved → {save_dir / 'oof_predictions.pkl'}")
    return oof_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true")
    parser.add_argument("--folds",  type=int, default=5)
    parser.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    args = parser.parse_args()

    cfg  = DEFAULT_CFG.copy()
    mcfg = MODEL_CFG_R2.copy()
    tcfg = TRAIN_CFG.copy()
    dcfg = DP_CFG.copy()
    tcfg["epochs"] = args.epochs

    print("Loading samples…")
    samples = load_offset_samples("", k_wells=0, cache_path=CACHE_PATH, verbose=False)
    train   = [s for s in samples if s.has_true]

    with open(OOF_K3_PATH, "rb") as f:
        oof_k3 = pickle.load(f)["oof_k3"]

    print("Loading typewell cache…")
    tw_cache = {}
    for s in samples:
        try: tw_cache[s.well_id] = _load_tw_raw(s.well_id, DATA_DIR)
        except: pass
    print(f"  {len(tw_cache)} typewells loaded")

    print("Loading greedy-GR TVT cache…")
    with open(GR_TVT_PATH, "rb") as f:
        gr_data = pickle.load(f)
    gr_tvt_cache = {wid: v for wid, v in gr_data["gr_tvt"].items() if v is not None}
    print(f"  {len(gr_tvt_cache)} wells with greedy-GR TVT  "
          f"(precomputed RMSE={gr_data['oof_rmse']:.4f} ft)")

    if args.quick:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(train), 50, replace=False).tolist()
        train_sub = [train[i] for i in keep]
        k3_sub    = oof_k3[keep]
        tcfg["epochs"] = 5
        run_oof_r2(train_sub, k3_sub, tw_cache, gr_tvt_cache, cfg,
                   mcfg, tcfg, dcfg, n_folds=2,
                   save_dir=Path("artifacts/dlmtp/run2_quick"))
    else:
        run_oof_r2(train, oof_k3, tw_cache, gr_tvt_cache, cfg,
                   mcfg, tcfg, dcfg, n_folds=args.folds,
                   save_dir=Path("artifacts/dlmtp/run2"))


if __name__ == "__main__":
    main()
