"""
train_dlmtp.py  —  Run 1: Heatmap U-Net band predictor.

Architecture:
    input  : 8-channel 2D heatmap (horizontal × typewell GR alignment)
    model  : HeatmapUNet — (B, 8, L, J) → (B, L, J) logits
    loss   : Gaussian CE  +  path-smoothness penalty
    decode : band-limited Viterbi DP

OOF evaluation: 5-fold CV on the 773 training wells.
    Each fold:  train on 4 folds (50 epochs),  eval on held-out fold
    Final metric: pooled row-RMSE across all 773 wells.

Usage:
    python train_dlmtp.py [--quick] [--folds 5] [--epochs 50] [--base 16]
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mtpnet.data    import load_offset_samples
from mtpnet.metrics import row_rmse
from dlmtp.dataset  import HeatmapDataset, build_inference_heatmap, compute_prior_tvt
from dlmtp.heatmap  import _smooth_gr
from dlmtp.unet     import HeatmapUNet
from dlmtp.losses   import gaussian_ce_loss, path_smooth_loss
from dlmtp.dp_decode import dp_decode, argmax_decode, bins_to_tvt

# ── Try to load typewell (reuse helper from local_search) ──────────────────
from mtpnet.local_search import load_typewell as _load_tw_raw

def _load_typewell(well_id: str, data_dir: str):
    return _load_tw_raw(well_id, data_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "/Users/alexander/Desktop/rogii/MTPNet/data/train"
CACHE_PATH = "artifacts/mtpnet_cache/samples.pkl"
OOF_K3_PATH= "artifacts/results/oof_k3.pkl"
OUT_DIR    = Path("artifacts/dlmtp")

DEFAULT_CFG = dict(
    crop_len         = 256,    # rows per training crop  (256 = ~15 min/fold on MPS)
    tvt_bins         = 96,     # J — TVT bins  (±96 ft at 2 ft/bin; covers p99 of K3 residuals)
    bin_ft           = 2.0,    # ft per bin
    gr_smooth_window = 101,
    n_crops          = 6,      # crops per well per epoch
)

MODEL_CFG = dict(
    in_ch  = 8,
    base   = 16,   # 16 ≈ 483 K params (MPS: ~20 ms/batch);  32 ≈ 1.9 M params (slower)
    depth  = 3,    # 3 pool levels → pad_multiple = 8
)

TRAIN_CFG = dict(
    epochs     = 50,
    batch_size = 4,
    lr         = 2e-4,
    wd         = 1e-4,
    sigma_ce   = 3.0,     # Gaussian CE width (bins)
    w_smooth   = 0.05,    # path-smoothness loss weight
    log_every  = 5,       # print epoch stats every N epochs
)

DP_CFG = dict(
    lambda_smooth = 0.05,
    max_trans     = 8,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def pad_to(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Pad (B, C, L, J) along L so L is divisible by `multiple`. Returns (padded, L, extra)."""
    L = x.shape[2]
    extra = (multiple - L % multiple) % multiple
    if extra > 0:
        x = torch.nn.functional.pad(x, [0, 0, 0, extra])
    return x, L, extra


def train_one_epoch(
    model: HeatmapUNet,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    tcfg: dict,
    pad_multiple: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for heatmap, true_bins, valid_mask, _ in loader:
        heatmap    = heatmap.to(device)
        true_bins  = true_bins.to(device)
        valid_mask = valid_mask.to(device)

        # Pad spatial dims to multiple of 2^depth
        heatmap, L, extra = pad_to(heatmap, pad_multiple)

        logits = model(heatmap)                          # (B, L+extra, J)
        logits = logits[:, :logits.shape[1] - extra, :] if extra > 0 else logits
        # trim back to L after model (skip connections preserve padded rows anyway)
        logits = logits[:, :L, :]

        loss = (
            gaussian_ce_loss(logits, true_bins, valid_mask, tcfg["sigma_ce"])
            + path_smooth_loss(logits, valid_mask, tcfg["w_smooth"])
        )

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_well(
    model: HeatmapUNet,
    sample,
    prior_tvt: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cfg: dict,
    dp_cfg: dict,
    device: torch.device,
    pad_multiple: int,
) -> np.ndarray:
    """Return (nh,) predicted TVT for all hidden rows of one well."""
    model.eval()
    nh = len(sample.hidden_rows)
    J  = cfg["tvt_bins"]

    gr_smooth = _smooth_gr(sample.gr.astype(float), cfg.get("gr_smooth_window", 101))
    heatmap, padded_len = build_inference_heatmap(
        sample, prior_tvt, tw_tvt, tw_gr, cfg, pad_to=pad_multiple,
        gr_smooth_precomputed=gr_smooth,
    )

    x = torch.from_numpy(heatmap).unsqueeze(0).to(device)  # (1, C, padded, J)
    logits = model(x).squeeze(0).cpu().numpy()              # (padded, J)
    logits = logits[:nh]                                    # (nh, J)

    path = dp_decode(logits, **dp_cfg)                      # (nh,)
    pred = bins_to_tvt(path, prior_tvt[:nh], J, cfg["bin_ft"])
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# OOF evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_oof(
    samples: list,
    oof_k3: np.ndarray,
    tw_cache: dict,
    cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    dp_cfg: dict,
    n_folds: int = 5,
    seed: int = 42,
    save_dir: Path = OUT_DIR,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    device      = (torch.device("mps")  if torch.backends.mps.is_available()
                   else torch.device("cpu"))
    pad_multiple = 2 ** model_cfg["depth"]

    print(f"\n{'='*60}")
    print(f"DL-MTP Run 1  |  device={device}  |  {n_folds}-fold OOF")
    print(f"  U-Net base={model_cfg['base']}, depth={model_cfg['depth']}")
    print(f"  crop={cfg['crop_len']}×{cfg['tvt_bins']}  bin_ft={cfg['bin_ft']}")
    print(f"  epochs={train_cfg['epochs']}  batch={train_cfg['batch_size']}")
    print(f"{'='*60}\n")

    # Filter to wells with typewell + true labels
    valid_idx   = [i for i, s in enumerate(samples)
                   if s.well_id in tw_cache and s.has_true]
    n_valid     = len(valid_idx)
    print(f"  Valid wells (typewell + label): {n_valid}")

    # Estimate wall time (45 ms/batch observed on MPS with num_workers=2)
    batches_per_epoch = int(n_valid * 0.8 * cfg["n_crops"] / train_cfg["batch_size"])
    est_min = (batches_per_epoch * 0.045 * train_cfg["epochs"] * n_folds) / 60
    print(f"  Batches/epoch ≈ {batches_per_epoch}  →  estimated total: {est_min:.0f} min"
          f"  ({est_min/60:.1f} h)")

    # Random fold assignment
    rng         = np.random.default_rng(seed)
    fold_ids    = rng.integers(0, n_folds, size=n_valid)

    # Storage for OOF predictions
    oof_preds   = {}   # well_idx → pred array
    oof_trues   = {}   # well_idx → true array

    t_start = time.time()

    for fold in range(n_folds):
        fold_t0   = time.time()
        val_mask  = fold_ids == fold
        tr_mask   = ~val_mask

        tr_idxs   = [valid_idx[i] for i in range(n_valid) if tr_mask[i]]
        val_idxs  = [valid_idx[i] for i in range(n_valid) if val_mask[i]]

        tr_samples = [samples[i] for i in tr_idxs]
        val_samples= [samples[i] for i in val_idxs]
        tr_k3      = oof_k3[tr_idxs]
        val_k3     = oof_k3[val_idxs]

        print(f"\n── Fold {fold+1}/{n_folds}  "
              f"(train={len(tr_idxs)} val={len(val_idxs)}) ──")

        # Build dataset + loader
        tr_ds = HeatmapDataset(tr_samples, tr_k3, tw_cache, cfg,
                               n_crops=cfg["n_crops"], train=True)
        tr_dl = DataLoader(tr_ds, batch_size=train_cfg["batch_size"],
                           shuffle=True, num_workers=2, drop_last=True,
                           persistent_workers=True)

        # Init model
        model = HeatmapUNet(**model_cfg).to(device)
        if fold == 0:
            print(f"  Model params: {model.n_params():,}")

        opt   = torch.optim.AdamW(model.parameters(),
                                  lr=train_cfg["lr"], weight_decay=train_cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=train_cfg["epochs"], eta_min=train_cfg["lr"] * 0.1)

        # ── Train ──────────────────────────────────────────────────────────
        for ep in range(1, train_cfg["epochs"] + 1):
            ep_loss = train_one_epoch(model, tr_dl, opt, device, train_cfg, pad_multiple)
            sched.step()
            if ep % train_cfg["log_every"] == 0 or ep == train_cfg["epochs"]:
                elapsed = time.time() - fold_t0
                eta_ep  = elapsed / ep * (train_cfg["epochs"] - ep)
                print(f"    ep {ep:3d}/{train_cfg['epochs']}  "
                      f"loss={ep_loss:.4f}  "
                      f"elapsed={elapsed:.0f}s  eta={eta_ep:.0f}s",
                      flush=True)

        # ── Validate ───────────────────────────────────────────────────────
        print(f"  Evaluating {len(val_idxs)} val wells…", flush=True)
        ap, at = [], []
        for vi, (s, k3) in enumerate(zip(val_samples, val_k3)):
            if s.well_id not in tw_cache:
                continue
            tw_tvt, tw_gr = tw_cache[s.well_id]
            prior_tvt     = compute_prior_tvt(s, k3)
            pred = predict_well(model, s, prior_tvt, tw_tvt, tw_gr,
                                cfg, dp_cfg, device, pad_multiple)
            true = s.tvt_true[s.hidden_rows]
            n2   = min(len(pred), len(true))
            ap.append(pred[:n2]); at.append(true[:n2])
            oof_preds[val_idxs[vi]] = pred[:n2]
            oof_trues[val_idxs[vi]] = true[:n2]

            if (vi + 1) % 20 == 0 or vi == len(val_idxs) - 1:
                cur = row_rmse(np.concatenate(ap), np.concatenate(at))
                print(f"    val {vi+1}/{len(val_idxs)}  pool_val={cur:.4f}", flush=True)

        fold_rmse = row_rmse(np.concatenate(ap), np.concatenate(at))
        print(f"  ✓ Fold {fold+1} val RMSE = {fold_rmse:.4f} ft  "
              f"({time.time()-fold_t0:.0f}s)", flush=True)

        # Save fold checkpoint
        ckpt_path = save_dir / f"fold{fold}_model.pt"
        torch.save(model.state_dict(), ckpt_path)

    # ── OOF pooled result ──────────────────────────────────────────────────
    all_pred = np.concatenate([oof_preds[i] for i in sorted(oof_preds)])
    all_true = np.concatenate([oof_trues[i] for i in sorted(oof_trues)])
    oof_rmse = row_rmse(all_pred, all_true)

    # Per-bucket breakdown
    buckets = {"short": [], "medium": [], "long": [], "xlong": []}
    per_well = []
    for i in sorted(oof_preds):
        s   = samples[i]
        nh  = len(s.hidden_rows)
        p   = oof_preds[i]; t = oof_trues[i]
        pw  = float(np.sqrt(np.mean((p - t) ** 2)))
        per_well.append(pw)
        bk  = ("xlong" if nh >= 8000 else "long" if nh >= 5000
               else "medium" if nh >= 2000 else "short")
        buckets[bk].append((p, t))

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"OOF pooled RMSE : {oof_rmse:.4f} ft")
    print(f"Per-well mean   : {np.mean(per_well):.4f} ft")
    for bk, pairs in buckets.items():
        if pairs:
            bp = np.concatenate([p for p, _ in pairs])
            bt = np.concatenate([t for _, t in pairs])
            print(f"  {bk:7s} ({len(pairs):3d} wells): {row_rmse(bp, bt):.3f} ft")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"{'='*60}\n")

    # Save OOF predictions
    with open(save_dir / "oof_predictions.pkl", "wb") as f:
        pickle.dump({"oof_preds": oof_preds, "oof_trues": oof_trues,
                     "oof_rmse": oof_rmse}, f)
    print(f"Saved OOF predictions → {save_dir / 'oof_predictions.pkl'}")

    return oof_rmse


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true",
                        help="Quick test: 2 folds, 5 epochs, 50 wells")
    parser.add_argument("--folds",  type=int, default=5)
    parser.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    parser.add_argument("--base",   type=int, default=MODEL_CFG["base"],
                        help="U-Net base channels (16=fast CPU, 32=better)")
    parser.add_argument("--bins",   type=int, default=DEFAULT_CFG["tvt_bins"])
    args = parser.parse_args()

    cfg       = DEFAULT_CFG.copy()
    mcfg      = MODEL_CFG.copy()
    tcfg      = TRAIN_CFG.copy()
    dcfg      = DP_CFG.copy()

    mcfg["base"] = args.base
    tcfg["epochs"] = args.epochs
    cfg["tvt_bins"] = args.bins

    # Limit J to be divisible by 2^depth
    pad_j = 2 ** mcfg["depth"]
    cfg["tvt_bins"] = int(np.ceil(cfg["tvt_bins"] / pad_j) * pad_j)

    print("Loading samples…", flush=True)
    samples = load_offset_samples("", k_wells=0, cache_path=CACHE_PATH, verbose=False)
    train   = [s for s in samples if s.has_true]

    with open(OOF_K3_PATH, "rb") as f:
        oof_k3 = pickle.load(f)["oof_k3"]

    print("Loading typewell cache…", flush=True)
    tw_cache = {}
    for s in samples:
        try:
            tw_cache[s.well_id] = _load_tw_raw(s.well_id, DATA_DIR)
        except (FileNotFoundError, Exception):
            pass
    print(f"  {len(tw_cache)} typewells loaded")

    if args.quick:
        n_folds = 2
        tcfg["epochs"] = 5
        rng = np.random.default_rng(0)
        keep = rng.choice(len(train), 50, replace=False).tolist()
        train_sub = [train[i] for i in keep]
        k3_sub    = oof_k3[keep]
        run_oof(train_sub, k3_sub, tw_cache, cfg, mcfg, tcfg, dcfg,
                n_folds=n_folds, save_dir=OUT_DIR / "quick")
    else:
        run_oof(train, oof_k3, tw_cache, cfg, mcfg, tcfg, dcfg,
                n_folds=args.folds, save_dir=OUT_DIR / "run1")


if __name__ == "__main__":
    main()
