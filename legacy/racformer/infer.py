"""RAC-Former v1 inference script.

TTA strategy: average predictions over 4 bin-shifts {0, 8, 16, 24}.
5-fold ensemble: average predictions from all fold checkpoints.
Outputs a submission CSV with columns [id, tvt].

Usage:
  python -m racformer.infer configs/racformer_v1.yml \\
      --output_dir artifacts/racformer_v1 \\
      --submission_path submission.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import RACFormerConfig, load_config
from .dataset import RACDataset, load_all_wells
from .model import RACFormer, load_state_dict_allowing_top_teacher_heads
from .oof_diagnostics import run_oof_diagnostics

# ---------------------------------------------------------------------------
# Batch to device
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Single-model, single bin-shift prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_one_pass(
    model: RACFormer,
    samples: list,
    cfg: RACFormerConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return {row_id: pred_tvt} for all hidden rows.

    batch_size=1 to handle variable hidden row counts cleanly.
    """
    model.eval()
    ds = RACDataset(
        samples,
        max_seq_len=cfg.data.max_seq_len,
        augment=False,
        k_seg=cfg.model.k_seg,
        rows_per_step=cfg.data.rows_per_step,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    predictions: dict[str, float] = {}
    for idx, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        out = model(batch)
        pred_tvt = model.materialize(batch, out)  # (1, H_max)

        sample = ds.samples[idx]
        n_hr = len(sample.hidden_row_ids)
        pred_vals = pred_tvt[0, :n_hr].cpu().numpy()

        for row_id, val in zip(sample.hidden_row_ids, pred_vals):
            predictions[row_id] = float(val)

    return predictions


# ---------------------------------------------------------------------------
# Multi-fold + TTA ensemble
# ---------------------------------------------------------------------------

def run_inference(
    cfg: RACFormerConfig,
    output_dir: Path,
    submission_path: Path,
    split: str = "test",
    bin_shifts: tuple[int, ...] = (0, 8, 16, 24),
) -> pd.DataFrame:
    """Ensemble across all folds × all TTA bin-shifts → submission CSV.

    Averaging strategy: simple mean over fold predictions × TTA passes.
    """
    device_str = cfg.train.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Inference device: {device}")

    # Locate fold checkpoints
    output_dir = Path(output_dir)
    fold_dirs = sorted(output_dir.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {output_dir}")
    print(f"Found {len(fold_dirs)} fold checkpoints")

    # Accumulator: {row_id: [list of predictions]}
    all_preds: dict[str, list[float]] = {}

    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / "best.pt"
        if not ckpt_path.exists():
            print(f"  [warn] missing {ckpt_path}, skipping", file=sys.stderr)
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model = RACFormer(cfg.model).to(device)

        # Honor explicit use_ema flag — only load EMA weights if the checkpoint
        # was saved after ema_start_epoch (issue #8).  Otherwise load raw weights.
        use_ema = bool(ckpt.get("use_ema", False)) and bool(ckpt.get("ema_shadow"))
        if use_ema:
            # Start from raw weights, then overwrite with EMA shadow where present
            load_state_dict_allowing_top_teacher_heads(model, ckpt["model_state"])
            ema_shadow = ckpt["ema_shadow"]
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in ema_shadow:
                        p.data.copy_(ema_shadow[name].to(p.device))
        else:
            load_state_dict_allowing_top_teacher_heads(model, ckpt["model_state"])

        model.eval()
        print(f"  Loaded {fold_dir.name} (val_rmse={ckpt.get('val_rmse', '?'):.4f})")

        for bin_shift in bin_shifts:
            samples = load_all_wells(cfg, split=split, bin_shift=bin_shift)
            if not samples:
                print(f"  [warn] no test samples found for split={split}", file=sys.stderr)
                break

            preds = predict_one_pass(model, samples, cfg, device)
            for row_id, val in preds.items():
                all_preds.setdefault(row_id, []).append(val)

    if not all_preds:
        raise RuntimeError("No predictions generated — check fold checkpoints and data paths")

    # Average over all passes
    row_ids = sorted(all_preds.keys())
    pred_vals = [float(np.mean(all_preds[rid])) for rid in row_ids]

    # Build DataFrame and merge with sample_submission for ordering
    pred_df = pd.DataFrame({"id": row_ids, "tvt": pred_vals})

    # Load sample_submission to get correct row ordering
    sample_sub_path = Path(cfg.data_dir) / "sample_submission.csv"
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        # Merge on id, keeping sample_submission order
        merged = sample_sub[["id"]].merge(pred_df, on="id", how="left")
        # Fill any missing predictions with prior (shouldn't happen)
        n_missing = merged["tvt"].isna().sum()
        if n_missing > 0:
            print(f"  [warn] {n_missing} missing predictions — filling with 0", file=sys.stderr)
            merged["tvt"] = merged["tvt"].fillna(0.0)
        submission = merged
    else:
        print(f"  [warn] sample_submission.csv not found at {sample_sub_path}", file=sys.stderr)
        submission = pred_df

    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission ({len(submission)} rows) → {submission_path}")

    return submission


# ---------------------------------------------------------------------------
# OOF prediction (for CV scoring)
# ---------------------------------------------------------------------------

def run_oof(
    cfg: RACFormerConfig,
    output_dir: Path,
    oof_path: Path,
    bin_shifts: tuple[int, ...] = (0, 8, 16, 24),
) -> pd.DataFrame:
    """Generate OOF predictions using each fold's val set and best.pt.

    Each fold's best checkpoint predicts its own held-out wells (TTA averaged).
    Returns a DataFrame with columns [id, tvt_pred, tvt_true].
    """
    device_str = cfg.train.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    output_dir = Path(output_dir)

    # Load all training wells
    all_samples = load_all_wells(cfg, split="train")
    from .train import _group_kfold_splits
    splits = _group_kfold_splits(all_samples, cfg.train.n_folds, cfg.train.seed)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_dir = output_dir / f"fold_{fold}"
        ckpt_path = fold_dir / "best.pt"
        if not ckpt_path.exists():
            print(f"  [warn] missing {ckpt_path}, skipping", file=sys.stderr)
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model = RACFormer(cfg.model).to(device)
        use_ema = bool(ckpt.get("use_ema", False)) and bool(ckpt.get("ema_shadow"))
        if use_ema:
            load_state_dict_allowing_top_teacher_heads(model, ckpt["model_state"])
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in ckpt["ema_shadow"]:
                        p.data.copy_(ckpt["ema_shadow"][name].to(p.device))
        else:
            load_state_dict_allowing_top_teacher_heads(model, ckpt["model_state"])
        model.eval()

        val_samples = [all_samples[i] for i in val_idx]

        # Aggregate predictions over TTA bin-shifts
        fold_preds: dict[str, list[float]] = {}
        fold_true: dict[str, float] = {}

        for bin_shift in bin_shifts:
            # Re-build samples with this bin_shift
            # We can't easily rebind bin_shift to existing samples, so reload
            # For OOF, reload only the val wells
            val_well_ids = {s.well_id for s in val_samples}
            samples_bs = load_all_wells(cfg, split="train", bin_shift=bin_shift)
            samples_bs_val = [s for s in samples_bs if s.well_id in val_well_ids]

            preds = predict_one_pass(model, samples_bs_val, cfg, device)
            for row_id, val in preds.items():
                fold_preds.setdefault(row_id, []).append(val)

        # Collect true values from original samples (bin_shift=0)
        for s in val_samples:
            ar = s.anchor_row
            for i, row_id in enumerate(s.hidden_row_ids):
                true_tvt = float(s.tvt_rows[ar + 1 + i])
                fold_true[row_id] = true_tvt

        for row_id in fold_preds:
            pred_mean = float(np.mean(fold_preds[row_id]))
            true = fold_true.get(row_id, float("nan"))
            rows.append({"id": row_id, "tvt_pred": pred_mean, "tvt_true": true})

    oof_df = pd.DataFrame(rows)
    if len(oof_df) > 0:
        valid = oof_df.dropna()
        rmse = float(np.sqrt(((valid["tvt_pred"] - valid["tvt_true"]) ** 2).mean()))
        print(f"OOF RMSE: {rmse:.4f} ({len(valid)} rows)")
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions → {oof_path}")
    return oof_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAC-Former v1 inference")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--output_dir", default=None, help="Directory with fold_* checkpoints (default: cfg.run.output_dir)")
    parser.add_argument("--submission_path", default="submission.csv", help="Output CSV path")
    parser.add_argument("--split", default="test", help="Data split to predict (default: test)")
    parser.add_argument("--oof", action="store_true", help="Generate OOF predictions instead of test")
    parser.add_argument("--oof_path", default="oof_predictions.csv", help="OOF output path")
    parser.add_argument("--oof_diagnostics", action="store_true", help="Generate OOF ablation diagnostics")
    parser.add_argument("--oof_diag_path", default="oof_diagnostics.csv", help="OOF diagnostics row CSV path")
    parser.add_argument(
        "--oof_summary_path",
        default="oof_diagnostics_summary.json",
        help="OOF diagnostics summary JSON path",
    )
    parser.add_argument("--oof_diag_bins", type=int, default=4, help="Number of buckets for OOF diagnostics")
    parser.add_argument("--bin_shifts", nargs="+", type=int, default=[0, 8, 16, 24], help="TTA bin shifts")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.run.output_dir)

    if args.oof_diagnostics:
        run_oof_diagnostics(
            cfg,
            out_dir,
            Path(args.oof_diag_path),
            Path(args.oof_summary_path),
            n_bins=args.oof_diag_bins,
        )
    elif args.oof:
        run_oof(cfg, out_dir, Path(args.oof_path), tuple(args.bin_shifts))
    else:
        run_inference(cfg, out_dir, Path(args.submission_path), args.split, tuple(args.bin_shifts))


if __name__ == "__main__":
    main()
