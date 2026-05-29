"""Out-of-fold evaluation pipeline for offset-state decoder experiments.

run_oof() is the core evaluation loop:
  - Takes a list of OffsetSample, fold splits, and a model-building function
  - For each fold: builds features, trains model, predicts on val, decodes TVT
  - Returns pooled RMSE + per-fold breakdown

Usage:
    from mtpnet.eval import run_oof, OOFResult

    def build_model_fn(X_tr, y_tr, X_va, y_va):
        return train_offset_classifier(X_tr, y_tr, X_va, y_va)

    result = run_oof(samples, folds, build_model_fn, mode="global")
    print(result.pooled_rmse)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

from .offsets import make_offset_grid, tvt_from_global_offset, tvt_from_ksegment_offsets
from .metrics import row_rmse, collect_predictions_and_targets
from .features import (
    make_well_features,
    make_segment_features,
    make_global_offset_labels,
    make_kseg_offset_labels,
)
from .models import predict_offset_proba
from .decode import decode_global_predictions, decode_kseg_predictions, viterbi_decode


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OOFResult:
    """Results from a single OOF evaluation run."""
    experiment: str
    mode: str               # "global" | "kseg"
    K: int                  # 1 for global
    decoder: str
    alpha: float

    pooled_rmse: float
    fold_rmses: list[float] = field(default_factory=list)

    # Per-well predictions (aligned with samples order)
    all_predictions: list[np.ndarray] = field(default_factory=list)
    all_true: list[np.ndarray] = field(default_factory=list)
    well_ids: list[str] = field(default_factory=list)

    # Optional soft probabilities (for diagnostics)
    all_proba: list[np.ndarray] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"OOFResult(experiment={self.experiment!r}, "
            f"mode={self.mode!r}, K={self.K}, "
            f"pooled_rmse={self.pooled_rmse:.4f}, "
            f"n_folds={len(self.fold_rmses)})"
        )


# ---------------------------------------------------------------------------
# OOF for global offset classifier
# ---------------------------------------------------------------------------

def run_oof_global(
    samples: list,
    folds: list[tuple[list[int], list[int]]],
    build_model_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any],
    experiment: str = "global_offset",
    decode_mode: str = "argmax",
    grid: np.ndarray | None = None,
    verbose: bool = True,
) -> OOFResult:
    """OOF evaluation for global (single-offset) classifier.

    Args:
        samples        : all training OffsetSample objects (with oracle labels)
        folds          : list of (train_idx, val_idx) splits
        build_model_fn : fn(X_tr, y_tr, X_va, y_va) → fitted model
        experiment     : name for this run
        decode_mode    : "argmax" or "mean"
        grid           : offset grid; defaults to make_offset_grid()
        verbose        : print fold progress

    Returns:
        OOFResult with pooled_rmse and per-fold details
    """
    if grid is None:
        grid = make_offset_grid()

    # Build full feature matrix and labels once
    X_all = make_well_features(samples)
    y_all = make_global_offset_labels(samples, grid)

    all_predictions = [None] * len(samples)
    all_proba       = [None] * len(samples)
    fold_rmses      = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = list(tr_idx)
        va_idx = list(va_idx)

        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]

        model = build_model_fn(X_tr, y_tr, X_va, y_va)

        # Predict probabilities for val set
        proba_va = predict_offset_proba(model, X_va)  # (|va|, N_BINS)

        val_samples = [samples[i] for i in va_idx]
        proba_list  = [proba_va[j] for j in range(len(va_idx))]

        preds = decode_global_predictions(val_samples, proba_list, grid, mode=decode_mode)
        trues = [s.tvt_hidden_true for s in val_samples]

        fold_preds_flat = np.concatenate([p.astype(np.float64) for p in preds])
        fold_true_flat  = np.concatenate([t.astype(np.float64) for t in trues])
        fold_rmse = row_rmse(fold_preds_flat, fold_true_flat)
        fold_rmses.append(fold_rmse)

        for j, i in enumerate(va_idx):
            all_predictions[i] = preds[j]
            all_proba[i]       = proba_list[j]

        if verbose:
            print(f"  fold {fold_i + 1}/{len(folds)}  val_rmse={fold_rmse:.4f}")

    # Pooled RMSE
    all_preds_flat = np.concatenate([p.astype(np.float64) for p in all_predictions])
    all_true_flat  = np.concatenate([
        s.tvt_hidden_true.astype(np.float64) for s in samples
    ])
    pooled = row_rmse(all_preds_flat, all_true_flat)

    if verbose:
        print(f"  [global OOF]  pooled={pooled:.4f}  folds={[f'{r:.4f}' for r in fold_rmses]}")

    return OOFResult(
        experiment=experiment,
        mode="global",
        K=1,
        decoder=decode_mode,
        alpha=0.0,
        pooled_rmse=pooled,
        fold_rmses=fold_rmses,
        all_predictions=all_predictions,
        all_true=[s.tvt_hidden_true for s in samples],
        well_ids=[s.well_id for s in samples],
        all_proba=all_proba,
    )


# ---------------------------------------------------------------------------
# OOF for K-segment classifier
# ---------------------------------------------------------------------------

def run_oof_kseg(
    samples: list,
    folds: list[tuple[list[int], list[int]]],
    build_model_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any],
    K: int,
    experiment: str | None = None,
    decoder: str = "viterbi",
    alpha: float = 10.0,
    beam_width: int = 32,
    grid: np.ndarray | None = None,
    verbose: bool = True,
) -> OOFResult:
    """OOF evaluation for K-segment classifier.

    For each well, the model predicts K soft distributions (one per segment),
    then Viterbi/beam decode finds the best-scoring segment sequence.

    Args:
        samples        : all training OffsetSample objects (with oracle labels)
        folds          : list of (train_idx, val_idx) splits
        build_model_fn : fn(X_tr, y_tr, X_va, y_va) → fitted model
        K              : number of equal-spaced segments
        experiment     : run name (defaults to "kseg_K{K}")
        decoder        : "viterbi", "beam", or "greedy"
        alpha          : Viterbi/beam transition penalty
        beam_width     : beam width for beam decoder
        grid           : offset grid
        verbose        : print fold progress

    Returns:
        OOFResult with pooled_rmse and per-fold details
    """
    if grid is None:
        grid = make_offset_grid()
    if experiment is None:
        experiment = f"kseg_K{K}_{decoder}"

    # Build per-segment features and labels
    # X_all shape: (N_wells * K, F)
    # y_all shape: (N_wells * K,)
    X_all = make_segment_features(samples, K)
    y_all = make_kseg_offset_labels(samples, K, grid)

    all_predictions = [None] * len(samples)
    all_proba       = [None] * len(samples)  # list of (K, N_BINS)
    fold_rmses      = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = list(tr_idx)
        va_idx = list(va_idx)

        # Select segment rows for this fold
        # Well i → rows [i*K, i*K+1, ..., i*K+K-1]
        tr_seg_rows = np.concatenate([np.arange(i * K, i * K + K) for i in tr_idx])
        va_seg_rows = np.concatenate([np.arange(i * K, i * K + K) for i in va_idx])

        X_tr = X_all[tr_seg_rows]
        y_tr = y_all[tr_seg_rows]
        X_va = X_all[va_seg_rows]
        y_va = y_all[va_seg_rows]

        model = build_model_fn(X_tr, y_tr, X_va, y_va)

        # Predict probabilities for val segments
        proba_va = predict_offset_proba(model, X_va)  # (|va| * K, N_BINS)

        # Reshape to (|va|, K, N_BINS)
        n_va = len(va_idx)
        proba_va_3d = proba_va.reshape(n_va, K, -1)  # (n_va, K, N_BINS)

        val_samples = [samples[i] for i in va_idx]
        proba_list  = [proba_va_3d[j] for j in range(n_va)]  # list of (K, N_BINS)

        preds = decode_kseg_predictions(
            val_samples, proba_list, K,
            grid=grid, decoder=decoder, alpha=alpha, beam_width=beam_width,
        )
        trues = [s.tvt_hidden_true for s in val_samples]

        fold_preds_flat = np.concatenate([p.astype(np.float64) for p in preds])
        fold_true_flat  = np.concatenate([t.astype(np.float64) for t in trues])
        fold_rmse = row_rmse(fold_preds_flat, fold_true_flat)
        fold_rmses.append(fold_rmse)

        for j, i in enumerate(va_idx):
            all_predictions[i] = preds[j]
            all_proba[i]       = proba_list[j]

        if verbose:
            print(f"  fold {fold_i + 1}/{len(folds)}  val_rmse={fold_rmse:.4f}")

    # Pooled RMSE
    all_preds_flat = np.concatenate([p.astype(np.float64) for p in all_predictions])
    all_true_flat  = np.concatenate([
        s.tvt_hidden_true.astype(np.float64) for s in samples
    ])
    pooled = row_rmse(all_preds_flat, all_true_flat)

    if verbose:
        print(f"  [{experiment}]  pooled={pooled:.4f}  folds={[f'{r:.4f}' for r in fold_rmses]}")

    return OOFResult(
        experiment=experiment,
        mode="kseg",
        K=K,
        decoder=decoder,
        alpha=alpha,
        pooled_rmse=pooled,
        fold_rmses=fold_rmses,
        all_predictions=all_predictions,
        all_true=[s.tvt_hidden_true for s in samples],
        well_ids=[s.well_id for s in samples],
        all_proba=all_proba,
    )


# ---------------------------------------------------------------------------
# Convenience: dispatch by mode
# ---------------------------------------------------------------------------

def run_oof(
    samples: list,
    folds: list[tuple[list[int], list[int]]],
    build_model_fn: Callable,
    mode: str = "global",
    K: int = 3,
    experiment: str | None = None,
    decoder: str = "viterbi",
    alpha: float = 10.0,
    beam_width: int = 32,
    grid: np.ndarray | None = None,
    verbose: bool = True,
) -> OOFResult:
    """Dispatch to run_oof_global or run_oof_kseg based on mode.

    Args:
        mode : "global" or "kseg"
        (all other args as in run_oof_global / run_oof_kseg)
    """
    if mode == "global":
        return run_oof_global(
            samples, folds, build_model_fn,
            experiment=experiment or "global_offset",
            decode_mode=decoder if decoder in ("argmax", "mean") else "argmax",
            grid=grid, verbose=verbose,
        )
    elif mode == "kseg":
        return run_oof_kseg(
            samples, folds, build_model_fn,
            K=K, experiment=experiment,
            decoder=decoder, alpha=alpha, beam_width=beam_width,
            grid=grid, verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'global' or 'kseg'.")


# ---------------------------------------------------------------------------
# Baseline (c0 drift, no model)
# ---------------------------------------------------------------------------

def run_c0_baseline(samples: list, verbose: bool = True) -> float:
    """Compute pooled RMSE for the c0 drift baseline (no model).

    Equivalent to predicting: TVT[h] = anchor - (Z[h] - Z_anchor) + c0 * row_delta

    Returns:
        Pooled row RMSE (ft)
    """
    from .offsets import tvt_base
    all_preds = []
    all_trues = []
    for s in samples:
        if s.tvt_hidden_true is None:
            continue
        pred = tvt_base(s.z, s.anchor_row, s.anchor_tvt, s.hidden_rows, c0=s.c0)
        all_preds.append(pred.astype(np.float64))
        all_trues.append(s.tvt_hidden_true.astype(np.float64))

    if not all_preds:
        return float("nan")

    rmse = row_rmse(np.concatenate(all_preds), np.concatenate(all_trues))
    if verbose:
        print(f"  [c0_baseline]  pooled_rmse={rmse:.4f}")
    return rmse


# ---------------------------------------------------------------------------
# Save / load OOF results
# ---------------------------------------------------------------------------

def save_oof_result(result: OOFResult, path: str) -> None:
    """Persist an OOFResult to disk (pickle)."""
    import pickle
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_oof_result(path: str) -> OOFResult:
    """Load an OOFResult from disk."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
