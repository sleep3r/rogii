#!/usr/bin/env python3
"""Experiment C: Top-state teacher model using ANCC formation labels.

ANCC (available train-only) correlates with dTVT + dZ (r≈0.9991).
We use it as a label to teach the model which formation state (offset bin)
applies to each row — this is a much stronger label than the LS oracle.

Model: LightGBM multiclass per-row classifier
Features: per-row (well-level stats + current row MD/Z/GR + prefix C-field)
Labels: ANCC-derived offset bin per hidden row
Decoder: Viterbi or beam over per-row probabilities → TVT

Since ANCC is train-only, this is a supervised teacher model:
  - At test time, the model receives per-row proba from features alone
  - Viterbi smooths the sequence

Note: top-state teacher is NOT a K-segment model — it produces per-row
predictions, making the Viterbi more powerful but also more expensive.

Run:
    python train_top_state.py [--data-dir DATA_DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from mtpnet.data import load_offset_samples, make_group_kfold
from mtpnet.offsets import (
    make_offset_grid, offset_to_bin, N_OFFSET_BINS,
    tvt_from_offset_per_row,
)
from mtpnet.eval import run_c0_baseline, save_oof_result
from mtpnet.models import train_offset_classifier, predict_offset_proba
from mtpnet.decode import viterbi_decode, beam_decode, make_log_transition
from mtpnet.metrics import row_rmse, print_scoreboard


# ---------------------------------------------------------------------------
# ANCC-derived per-row offset labels
# ---------------------------------------------------------------------------

def make_top_state_labels(
    samples: list,
    grid: np.ndarray,
) -> list[np.ndarray]:
    """Compute true per-row offset bin from ANCC + Z.

    true_offset_per_row[k] = d(TVT_true + Z)[k] = dC_true[k]

    The offset label for hidden row k is the nearest grid bin to dC_true[k].

    Args:
        samples : training OffsetSample objects (must have tvt_true and ANCC)
        grid    : offset grid

    Returns:
        List of (H_i,) int32 bin arrays, one per well
    """
    labels_per_well = []
    for s in samples:
        tvt_true = s.tvt_hidden_true
        if tvt_true is None:
            labels_per_well.append(np.zeros(s.n_hidden, dtype=np.int32))
            continue

        H = s.n_hidden
        hidden_rows = s.hidden_rows
        z = s.z.astype(np.float64)
        tvt_t = tvt_true.astype(np.float64)

        # Previous TVT: anchor for row 0, row k-1 for row k
        prev_tvt = np.empty(H, dtype=np.float64)
        prev_tvt[0] = float(s.anchor_tvt)
        prev_tvt[1:] = tvt_t[:-1]

        # Previous Z
        prev_z = np.empty(H, dtype=np.float64)
        prev_z[0] = float(s.anchor_z)
        prev_z[1:] = z[hidden_rows[:-1]]

        # dC = d(TVT + Z) = (dTVT + dZ)
        dtvt = tvt_t - prev_tvt
        dz   = z[hidden_rows] - prev_z
        dC   = dtvt + dz

        bins = np.array([offset_to_bin(float(v), grid) for v in dC], dtype=np.int32)
        labels_per_well.append(bins)

    return labels_per_well


# ---------------------------------------------------------------------------
# Per-row features for top-state model
# ---------------------------------------------------------------------------

def make_per_row_features(
    samples: list,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build a feature matrix with one row per hidden row.

    Returns:
        X         : (N_total_hidden_rows, F) float32 array
        well_segs : list of (start_idx, end_idx) into X for each well
    """
    from mtpnet.features import make_well_feature_dict

    rows = []
    well_segs = []
    offset = 0

    for s in samples:
        well_feats = make_well_feature_dict(s)
        H = s.n_hidden
        hidden_rows = s.hidden_rows
        z = s.z.astype(np.float64)
        md = s.md.astype(np.float64)
        gr = s.gr.astype(np.float64)

        for k, h in enumerate(hidden_rows):
            d = dict(well_feats)  # copy well-level features

            # Row-specific features
            d["row_z"]        = float(z[h])
            d["row_md"]       = float(md[h])
            d["row_gr"]       = float(gr[h]) if np.isfinite(gr[h]) else well_feats.get("gr_mean", 0.0)
            d["row_pos_norm"] = float(k) / max(H - 1, 1)   # 0 = first hidden, 1 = last
            d["row_dz"]       = float(z[h] - (z[hidden_rows[k-1]] if k > 0 else s.anchor_z))
            d["row_dmd"]      = float(md[h] - (md[hidden_rows[k-1]] if k > 0 else md[s.anchor_row]))
            d["row_z_rel"]    = float(z[h] - s.anchor_z)
            d["row_md_rel"]   = float(md[h] - md[s.anchor_row])

            rows.append(d)

        well_segs.append((offset, offset + H))
        offset += H

    df = pd.DataFrame(rows).fillna(0.0).astype(np.float32)
    return df.values, well_segs


# ---------------------------------------------------------------------------
# OOF for top-state model
# ---------------------------------------------------------------------------

def run_oof_top_state(
    samples: list,
    folds: list,
    grid: np.ndarray,
    backend: str = "lgbm",
    decoder: str = "viterbi",
    alpha: float = 20.0,
    beam_width: int = 32,
    verbose: bool = True,
) -> dict:
    """OOF evaluation for per-row top-state teacher model."""

    log_T = make_log_transition(len(grid), alpha, grid[1] - grid[0] if len(grid) > 1 else 0.001)

    # Build full features and labels
    X_all, well_segs = make_per_row_features(samples)
    labels_per_well  = make_top_state_labels(samples, grid)

    # Flatten labels
    y_all = np.concatenate(labels_per_well).astype(np.int32)

    # Row-index groups (each group = one well's rows)
    all_predictions = [None] * len(samples)
    fold_rmses: list[float] = []

    def _well_row_indices(well_idx_list):
        idxs = []
        for i in well_idx_list:
            start, end = well_segs[i]
            idxs.extend(range(start, end))
        return np.array(idxs, dtype=np.int64)

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tr_row_idx = _well_row_indices(tr_idx)
        va_row_idx = _well_row_indices(va_idx)

        X_tr = X_all[tr_row_idx]
        y_tr = y_all[tr_row_idx]
        X_va = X_all[va_row_idx]
        y_va = y_all[va_row_idx]

        model = train_offset_classifier(
            X_tr, y_tr, X_va, y_va,
            backend=backend,
            early_stopping_rounds=30,
            verbose=False,
        )

        proba_va = predict_offset_proba(model, X_va)  # (N_va_rows, N_BINS)

        fold_preds_flat: list[np.ndarray] = []
        fold_true_flat:  list[np.ndarray] = []

        row_cursor = 0
        for i in va_idx:
            s = samples[i]
            H = s.n_hidden
            tvt_true = s.tvt_hidden_true
            if tvt_true is None or H == 0:
                row_cursor += H
                continue

            proba_well = proba_va[row_cursor: row_cursor + H]  # (H, N_BINS)
            row_cursor += H

            # Decode
            if decoder == "viterbi":
                bins = viterbi_decode(proba_well, alpha=alpha, log_transition=log_T)
            elif decoder == "beam":
                bins = beam_decode(proba_well, alpha=alpha, beam_width=beam_width, log_transition=log_T)
            else:
                bins = np.argmax(proba_well, axis=1).astype(np.int32)

            offsets = grid[bins]   # (H,)
            pred = tvt_from_offset_per_row(
                s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, offsets
            )
            all_predictions[i] = pred
            fold_preds_flat.append(pred.astype(np.float64))
            fold_true_flat.append(tvt_true.astype(np.float64))

        fold_rmse = row_rmse(
            np.concatenate(fold_preds_flat),
            np.concatenate(fold_true_flat),
        )
        fold_rmses.append(fold_rmse)
        if verbose:
            print(f"  fold {fold_i + 1}/{len(folds)}  val_rmse={fold_rmse:.4f}")

    # Pooled RMSE
    all_p = np.concatenate([p.astype(np.float64) for p in all_predictions if p is not None])
    all_t = np.concatenate([
        s.tvt_hidden_true.astype(np.float64)
        for s in samples if s.tvt_hidden_true is not None
    ])
    pooled = row_rmse(all_p, all_t)
    if verbose:
        print(f"  [top_state]  pooled={pooled:.4f}  folds={[f'{r:.4f}' for r in fold_rmses]}")

    return {
        "pooled_rmse": pooled,
        "fold_rmses":  fold_rmses,
        "all_predictions": all_predictions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment C: Top-state teacher")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/Users/alexander/Desktop/rogii/MTPNet/data/train"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/mtpnet_cache"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument("--backend", choices=["lgbm", "catboost"], default="lgbm")
    parser.add_argument("--decoder", choices=["viterbi", "beam", "greedy"], default="viterbi")
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.cache_dir / "samples.pkl" if args.k_wells < 0 else None
    print(f"\nLoading wells …")
    samples = load_offset_samples(
        args.data_dir, k_wells=args.k_wells, cache_path=cache_path, verbose=True
    )
    train_samples = [s for s in samples if s.has_true]
    print(f"Training wells: {len(train_samples)}")

    folds = make_group_kfold(train_samples, n_folds=args.n_folds)
    grid  = make_offset_grid()

    print("\n--- c0 baseline ---")
    c0_rmse = run_c0_baseline(train_samples, verbose=True)

    exp_name = f"top_state_{args.backend}_{args.decoder}_a{args.alpha:.0f}"
    print(f"\n--- Experiment C: {exp_name} ---")
    result = run_oof_top_state(
        train_samples, folds, grid,
        backend=args.backend,
        decoder=args.decoder,
        alpha=args.alpha,
        beam_width=args.beam_width,
        verbose=True,
    )

    scoreboard = {
        "c0_baseline": c0_rmse,
        exp_name:      result["pooled_rmse"],
    }
    print_scoreboard(scoreboard, title="Experiment C Results")

    return 0


if __name__ == "__main__":
    sys.exit(main())
