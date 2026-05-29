#!/usr/bin/env python3
"""Experiment A: Global offset classifier (single offset per well).

Model: LightGBM multiclass over N_OFFSET_BINS = 321 bins
Features: well-level (C-drift statistics, geometry, GR)
Labels: oracle global offset bin (grid-search best offset)
Evaluation: OOF row-RMSE after physical integration

Expected output:
    c0_baseline_rmse    : ~14.8 ft  (no model, just known c0 drift)
    global_offset_rmse  : ??? ft    (should be better than c0 baseline)

Run:
    python train_global_offset.py [--data-dir DATA_DIR] [--n-folds 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from mtpnet.data import load_offset_samples, make_group_kfold
from mtpnet.offsets import make_offset_grid
from mtpnet.oracle import compute_and_attach_oracles
from mtpnet.eval import run_oof_global, run_c0_baseline, save_oof_result
from mtpnet.models import train_offset_classifier
from mtpnet.metrics import print_scoreboard


def build_model_fn(backend="lgbm", params=None):
    """Return a model-building closure for OOF."""
    def fn(X_tr, y_tr, X_va, y_va):
        return train_offset_classifier(
            X_tr, y_tr,
            X_val=X_va, y_val=y_va,
            backend=backend,
            params=params,
            early_stopping_rounds=50,
            verbose=False,
        )
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment A: Global offset classifier")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/Users/alexander/Desktop/rogii/MTPNet/data/train"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/mtpnet_cache"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-wells", type=int, default=-1,
                        help="Smoke-test: use only first k wells")
    parser.add_argument("--backend", choices=["lgbm", "catboost"], default="lgbm")
    parser.add_argument("--decode-mode", choices=["argmax", "mean"], default="argmax")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/results"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    cache_path = args.cache_dir / "samples.pkl" if args.k_wells < 0 else None
    print(f"\nLoading wells …")
    samples = load_offset_samples(
        args.data_dir, k_wells=args.k_wells, cache_path=cache_path, verbose=True
    )
    train_samples = [s for s in samples if s.has_true]
    print(f"Training wells: {len(train_samples)}")

    # ---- Compute oracles (if not cached) ----
    needs_oracle = any(s.global_offset_star is None for s in train_samples)
    if needs_oracle:
        print("\nComputing global oracles …")
        grid = make_offset_grid()
        compute_and_attach_oracles(train_samples, ks=[1], grid=grid, verbose=True)
    else:
        print("Oracle labels already present.")

    # ---- Folds ----
    folds = make_group_kfold(train_samples, n_folds=args.n_folds)
    print(f"Folds: {args.n_folds}")

    # ---- c0 baseline ----
    print("\n--- c0 baseline ---")
    c0_rmse = run_c0_baseline(train_samples, verbose=True)

    # ---- Global offset OOF ----
    print(f"\n--- Experiment A: global offset [{args.backend}, decode={args.decode_mode}] ---")
    result = run_oof_global(
        train_samples,
        folds,
        build_model_fn(args.backend),
        experiment=f"global_offset_{args.backend}_{args.decode_mode}",
        decode_mode=args.decode_mode,
        verbose=True,
    )

    # ---- Scoreboard ----
    scoreboard = {
        "c0_baseline":       c0_rmse,
        result.experiment:   result.pooled_rmse,
    }
    print_scoreboard(scoreboard, title="Experiment A Results")

    # ---- Save ----
    out_path = args.output_dir / f"{result.experiment}.pkl"
    save_oof_result(result, str(out_path))
    print(f"Saved → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
