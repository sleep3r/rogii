"""
pathpolicy/train.py — PolicyFormer training loop.

Two modes, driven by cfg["head_type"]:

  "classifier" (default)
    Loss = CE(logits, oracle_label) + lambda_v * MSE(value, oracle_rmse)
    Val metric: val_acc = fraction where argmax(logits) == oracle_label

  "rmse_regressor"
    Loss = Huber(pred_log_rmse, log1p(oracle_rmse_per_action))   [masked]
           + lambda_rank * ranking_margin_loss                    [optional]
           + lambda_v    * MSE(value, oracle_rmse)
    Val metric: val_mean_regret = mean(selected_rmse - oracle_best_rmse) per chunk  [ft]
    Checkpoint by: val_mean_regret (lower = better)

Usage:
    uv run --extra dev python -m pathpolicy.train --config configs/pathpolicy_v5.yml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pathpolicy.actions import N_ACTION_TYPES, ACTION_VOCAB
from pathpolicy.dataset import make_datasets, D_FEAT
from pathpolicy.model import PolicyFormer


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    "head_type": "classifier",   # "classifier" or "rmse_regressor"
    "d_model": 128,
    "n_layers": 2,
    "nhead": 4,
    "dim_ff": 256,
    "dropout": 0.1,
    "batch_size": 64,
    "n_epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "lambda_v": 0.1,
    # --- classifier-only ---
    "label_smoothing": 0.0,
    "soft_kl_alpha": 0.0,
    "kl_tau": 2.0,
    # --- regressor-only ---
    "huber_delta": 2.0,          # Huber loss delta (in log1p-RMSE space)
    "lambda_rank": 0.0,          # weight for ranking margin loss (0 = disabled)
    "rank_margin": 0.2,          # margin in log1p-RMSE space
    # --- shared ---
    "early_stop_patience": 0,
    "tail_oversample": 1,
    "val_frac": 0.2,
    "seed": 42,
    "output_dir": "artifacts/pathpolicy_v0",
    # data paths (relative to repo root)
    "oracle_chunks": "artifacts/oracle_v0/oracle_chunks.parquet",
    "data_dir": "data/train",
    "base_path": "../old/artifacts/oof_baseline/schema10_oof.parquet",
    "b2_path": "../old/artifacts/formation_b2_danger_guard_a2_full_schema10/guarded_predictions.parquet",
    "a_path": "../old/artifacts/formation_plane_knn/oof_candidates.parquet",
    "mtp_path": "artifacts/mtp_v3_gr_forced/track_row_predictions.parquet",
}


def _load_cfg(config_path: str | None) -> dict:
    cfg = dict(DEFAULT_CFG)
    if config_path:
        import yaml
        with open(config_path) as f:
            override = yaml.safe_load(f)
        cfg.update(override)
    return cfg


# ---------------------------------------------------------------------------
# Collate (handles variable context length from short wells)
# ---------------------------------------------------------------------------

def masked_smooth_ce(
    logits: torch.Tensor,    # [B, K]  — already has -inf for unavailable actions
    labels: torch.Tensor,    # [B]
    mask: torch.Tensor,      # [B, K]  bool, True = available
    epsilon: float = 0.0,
) -> torch.Tensor:
    """CE loss with label smoothing restricted to *available* actions.

    Standard PyTorch label_smoothing distributes mass to ALL K classes
    including -inf-masked ones → -inf * 0 = NaN.  This version:
    1. Zeroes out masked positions explicitly before summing (avoids -inf * 0).
    2. Distributes smoothing mass only over available (unmasked) actions.
    """
    log_probs = F.log_softmax(logits, dim=-1)               # -inf for masked slots
    # Zero out masked positions explicitly to avoid -inf * 0 = NaN
    log_probs_safe = log_probs.masked_fill(~mask, 0.0)

    nll = -log_probs.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
    if epsilon <= 0.0:
        return nll.mean()
    n_valid = mask.float().sum(dim=-1).clamp(min=1.0)          # [B]
    smooth  = -log_probs_safe.sum(dim=-1) / n_valid            # avg log-prob over valid actions
    loss = (1.0 - epsilon) * nll + epsilon * smooth
    return loss.mean()


def soft_kl_loss(
    logits: torch.Tensor,        # [B, K]  — -inf for unavailable actions
    action_rmses: torch.Tensor,  # [B, K]  — per-action RMSE from oracle (NaN = unavailable)
    mask: torch.Tensor,          # [B, K]  bool, True = available
    tau: float = 2.0,
) -> torch.Tensor:
    """KL( softmax(-rmse/tau) || softmax(logits) ) over available actions."""
    rmse_filled = action_rmses.clone()
    rmse_filled[~mask] = 1e6
    nan_mask = torch.isnan(rmse_filled)
    rmse_filled = rmse_filled.masked_fill(nan_mask, 1e6)
    soft_targets = torch.softmax(-rmse_filled / tau, dim=-1)
    log_preds = F.log_softmax(logits, dim=-1)
    log_preds_safe = log_preds.masked_fill(~mask, 0.0)
    kl = -(soft_targets * log_preds_safe).sum(dim=-1)
    return kl.mean()


# ---------------------------------------------------------------------------
# Regressor losses
# ---------------------------------------------------------------------------

def masked_huber_loss(
    pred: torch.Tensor,   # [B, K] predicted log1p-RMSE
    target: torch.Tensor, # [B, K] target log1p-RMSE (NaN where unavailable)
    valid: torch.Tensor,  # [B, K] bool — True where both mask=True and target is finite
    delta: float = 2.0,
) -> torch.Tensor:
    """Huber loss over valid (available + non-NaN) positions.

    Using log1p-RMSE space keeps high-RMSE actions from dominating the gradient.
    delta=2.0 ≈ 7.3 ft in RMSE space (log1p(7.3) ≈ 2.1).
    """
    if not valid.any():
        return pred.sum() * 0.0   # zero grad, avoids leaf-tensor detach issues
    return F.huber_loss(pred[valid], target[valid], delta=delta, reduction="mean")


def ranking_margin_loss(
    pred: torch.Tensor,   # [B, K] predicted log1p-RMSE
    target: torch.Tensor, # [B, K] target log1p-RMSE (NaN where unavailable)
    valid: torch.Tensor,  # [B, K] bool
    margin: float = 0.2,
) -> torch.Tensor:
    """Encourage argmin(pred) == argmin(target) via pairwise margin.

    For each sample b:
      oracle_idx = argmin over valid targets
      For each other valid action j: penalise if pred[oracle_idx] > pred[j] - margin
                                     (oracle should be lower than j by at least margin)

    Normalised by number of competitors so loss is O(1) regardless of K.
    """
    B, K = pred.shape
    target_filled = target.clone()
    target_filled[~valid] = 1e9
    target_filled[torch.isnan(target_filled)] = 1e9

    oracle_idx = target_filled.argmin(dim=-1)             # [B]
    oracle_pred = pred.gather(1, oracle_idx.unsqueeze(1)) # [B, 1]

    # oracle_pred should be at least `margin` lower than every competitor
    violations = F.relu(oracle_pred - pred + margin)      # [B, K]

    # zero out: oracle itself and invalid actions
    oracle_oh = torch.zeros(B, K, dtype=torch.bool, device=pred.device)
    oracle_oh.scatter_(1, oracle_idx.unsqueeze(1), True)
    violations = violations.masked_fill(~valid | oracle_oh, 0.0)

    n_comp = (valid & ~oracle_oh).float().sum(dim=-1).clamp(min=1.0)  # [B]
    return (violations.sum(dim=-1) / n_comp).mean()


# ---------------------------------------------------------------------------
# Collate (handles variable context length from short wells)
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    # Context: pad to max T in batch
    max_t = max(b["context"].shape[0] for b in batch)
    contexts, action_segs_list, action_masks, labels, values, type_ids, action_rmses_list = (
        [], [], [], [], [], [], []
    )

    for b in batch:
        ctx = b["context"]   # [T, D_FEAT]
        T = ctx.shape[0]
        if T < max_t:
            pad = torch.zeros(max_t - T, ctx.shape[1])
            ctx = torch.cat([pad, ctx], dim=0)
        contexts.append(ctx)
        action_segs_list.append(b["action_segs"])   # [K, CHUNK_LEN]
        action_masks.append(b["action_mask"])        # [K]
        labels.append(b["oracle_label"])
        values.append(b["value_target"])
        action_rmses_list.append(b["action_rmses"])  # [K]
        # type_ids: just 0..K-1 as type indices (already embedded in model via vocab)
        type_ids.append(torch.arange(N_ACTION_TYPES, dtype=torch.long))

    return {
        "context":      torch.stack(contexts),            # [B, T, D_FEAT]
        "action_segs":  torch.stack(action_segs_list),   # [B, K, CHUNK_LEN]
        "action_mask":  torch.stack(action_masks),        # [B, K]
        "oracle_label": torch.stack(labels),              # [B]
        "value_target": torch.stack(values),              # [B]
        "type_ids":     torch.stack(type_ids),            # [B, K]
        "action_rmses": torch.stack(action_rmses_list),  # [B, K]
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: PolicyFormer,
    loader: DataLoader,
    device: torch.device,
    lambda_v: float,
    head_type: str = "classifier",
    huber_delta: float = 2.0,
) -> dict:
    model.eval()
    total_primary = total_v = total_acc = total_n = 0.0
    total_regret = total_regret_n = 0.0   # regressor-only

    for batch in loader:
        ctx   = batch["context"].to(device)
        segs  = batch["action_segs"].to(device)
        mask  = batch["action_mask"].to(device)
        label = batch["oracle_label"].to(device)
        vtgt  = batch["value_target"].to(device)
        tids  = batch["type_ids"].to(device)
        armse = batch["action_rmses"].to(device)   # [B, K]

        scores, value = model(ctx, segs, tids, mask)
        v_loss = F.mse_loss(value, vtgt)
        total_v += v_loss.item() * len(label)

        if head_type == "rmse_regressor":
            # Primary loss: masked Huber on log1p-regret (same target as training loop)
            valid = mask & ~torch.isnan(armse)
            armse_for_min = armse.clone()
            armse_for_min[~valid] = 1e9
            min_rmse_eval = armse_for_min.min(dim=1, keepdim=True).values  # [B, 1]
            regret = (armse - min_rmse_eval).clamp(min=0)
            log_target = torch.log1p(regret)
            primary = masked_huber_loss(scores, log_target, valid, delta=huber_delta)

            # argmin inference
            scores_safe = scores.masked_fill(~mask, 1e9)
            preds = scores_safe.argmin(dim=-1)   # [B]

            # Mean regret: actual RMSE of selected action minus oracle-best RMSE
            armse_filled = armse.clone()
            armse_filled[~mask] = 1e9
            armse_filled[torch.isnan(armse_filled)] = 1e9
            min_rmse  = armse_filled.min(dim=1).values             # [B] oracle best
            sel_rmse  = armse.gather(1, preds.unsqueeze(1)).squeeze(1)  # [B]
            valid_sel = mask.any(dim=1) & ~torch.isnan(sel_rmse) & (min_rmse < 1e8)
            regret    = (sel_rmse - min_rmse).clamp(min=0)
            total_regret   += regret[valid_sel].sum().item()
            total_regret_n += valid_sel.float().sum().item()
        else:
            ce_fn = nn.CrossEntropyLoss(ignore_index=-1)
            primary = ce_fn(scores, label)
            preds = scores.argmax(dim=-1)

        total_primary += primary.item() * len(label)
        total_acc += (preds == label).float().sum().item()
        total_n   += len(label)

    mean_regret = total_regret / max(total_regret_n, 1)
    model.train()
    return {
        "primary":     total_primary / total_n,   # CE or Huber depending on mode
        "v":           total_v / total_n,
        "acc":         total_acc / total_n,        # argmax_acc or argmin_acc
        "mean_regret": mean_regret,                # ft; 0.0 for classifier mode
        "loss":        (total_primary + lambda_v * total_v) / total_n,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[train] device={device}", flush=True)

    # Data
    train_ds, val_ds, train_well_ids, val_well_ids = make_datasets(
        oracle_chunks_path=Path(cfg["oracle_chunks"]),
        data_dir=Path(cfg["data_dir"]),
        base_path=Path(cfg["base_path"]),
        b2_path=Path(cfg["b2_path"]),
        a_path=Path(cfg["a_path"]),
        mtp_path=Path(cfg["mtp_path"]),
        val_frac=cfg["val_frac"],
        seed=cfg["seed"],
        tail_oversample=cfg.get("tail_oversample", 1),
    )

    # Save well splits so evaluate.py can restrict to val-only wells
    with open(out_dir / "val_wells.json", "w") as f:
        json.dump(val_well_ids, f, indent=2)
    with open(out_dir / "train_wells.json", "w") as f:
        json.dump(train_well_ids, f, indent=2)
    print(f"[train] saved val_wells.json ({len(val_well_ids)} wells) and "
          f"train_wells.json ({len(train_well_ids)} wells)", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=_collate, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=_collate, num_workers=0, pin_memory=False,
    )

    # --- Config ---
    head_type   = cfg.get("head_type", "classifier")
    huber_delta = cfg.get("huber_delta", 2.0)
    lambda_rank = cfg.get("lambda_rank", 0.0)
    rank_margin = cfg.get("rank_margin", 0.2)
    lambda_v    = cfg["lambda_v"]
    epsilon     = cfg.get("label_smoothing", 0.0)
    soft_alpha  = cfg.get("soft_kl_alpha", 0.0)
    kl_tau      = cfg.get("kl_tau", 2.0)
    patience    = cfg.get("early_stop_patience", 0)

    # Model
    model = PolicyFormer(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        nhead=cfg["nhead"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        head_type=head_type,
    ).to(device)
    print(f"[train] model params: {model.n_params:,}  head_type={head_type}", flush=True)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"], eta_min=cfg["lr"] * 0.05)

    # Checkpointing state — regressor uses mean_regret (lower=better), classifier uses acc
    best_val_regret = float("inf")
    best_val_acc    = -1.0
    history: list[dict] = []
    epochs_no_improve = 0

    for epoch in range(1, cfg["n_epochs"] + 1):
        model.train()
        ep_loss = ep_v = ep_n = 0.0
        t0 = time.time()

        for batch in train_loader:
            ctx   = batch["context"].to(device)
            segs  = batch["action_segs"].to(device)
            mask  = batch["action_mask"].to(device)
            label = batch["oracle_label"].to(device)
            vtgt  = batch["value_target"].to(device)
            tids  = batch["type_ids"].to(device)
            armse = batch["action_rmses"].to(device)   # [B, K]

            scores, value = model(ctx, segs, tids, mask)
            v_loss = F.mse_loss(value, vtgt)

            if head_type == "rmse_regressor":
                # Primary loss: masked Huber on log1p-regret
                # regret_i = clip(rmse_i - min_chunk_rmse, 0) — chunk-difficulty-normalised target.
                valid = mask & ~torch.isnan(armse)
                armse_for_min = armse.clone()
                armse_for_min[~valid] = 1e9
                min_rmse_b5 = armse_for_min.min(dim=1, keepdim=True).values  # [B, 1]
                regret = (armse - min_rmse_b5).clamp(min=0)
                log_target = torch.log1p(regret)
                primary    = masked_huber_loss(scores, log_target, valid, delta=huber_delta)
                if lambda_rank > 0.0:
                    primary = primary + lambda_rank * ranking_margin_loss(
                        scores, log_target, valid, margin=rank_margin
                    )
            else:
                # Classifier mode: CE (hard) + optional soft KL
                ce_hard = masked_smooth_ce(scores, label, mask, epsilon)
                if soft_alpha > 0.0 and not armse.isnan().all():
                    ce_soft = soft_kl_loss(scores, armse, mask, kl_tau)
                    primary = (1.0 - soft_alpha) * ce_hard + soft_alpha * ce_soft
                else:
                    primary = ce_hard

            loss = primary + lambda_v * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += primary.item() * len(label)
            ep_v    += v_loss.item() * len(label)
            ep_n    += len(label)

        scheduler.step()

        val_metrics = evaluate(
            model, val_loader, device, lambda_v,
            head_type=head_type, huber_delta=huber_delta,
        )
        elapsed = time.time() - t0

        row = {
            "epoch":       epoch,
            "train_loss":  ep_loss / ep_n,
            "val_loss":    val_metrics["loss"],
            "val_acc":     val_metrics["acc"],
            "val_regret":  val_metrics["mean_regret"],
            "elapsed":     elapsed,
        }
        history.append(row)

        print(
            f"[epoch {epoch:02d}/{cfg['n_epochs']}] "
            f"train_loss={ep_loss/ep_n:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_regret={val_metrics['mean_regret']:.3f}ft "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        # Checkpoint: regressor → lower val_regret; classifier → higher val_acc
        if head_type == "rmse_regressor":
            improved = val_metrics["mean_regret"] < best_val_regret - 1e-6
            if improved:
                best_val_regret = val_metrics["mean_regret"]
        else:
            improved = val_metrics["acc"] > best_val_acc + 1e-6
            if improved:
                best_val_acc = val_metrics["acc"]

        if improved:
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            key = (f"val_regret={best_val_regret:.3f}ft" if head_type == "rmse_regressor"
                   else f"val_acc={best_val_acc:.4f}")
            print(f"  → saved best ({key})", flush=True)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"[train] early stop: no improvement for {patience} epochs.", flush=True)
                break

    # Save last checkpoint and history
    torch.save(model.state_dict(), out_dir / "last.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    best_str = (f"val_regret={best_val_regret:.3f}ft" if head_type == "rmse_regressor"
                else f"val_acc={best_val_acc:.4f}")
    print(f"[train] done. Best {best_str}. Outputs in {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    # Allow CLI overrides
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--n_epochs",    type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--d_model",     type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    for k in ["output_dir", "n_epochs", "batch_size", "d_model", "lr"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    train(cfg)


if __name__ == "__main__":
    main()
