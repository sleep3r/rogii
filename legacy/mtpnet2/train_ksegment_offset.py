#!/usr/bin/env python3
"""Experiments B: K-segment offset classifier (K=3 and K=5).

Model: LightGBM multiclass over N_OFFSET_BINS = 321 bins
Features: per-segment (well-level + segment position/geometry)
Labels: oracle K-segment LS offsets per segment
Decoder: Viterbi DP over K steps
Evaluation: OOF row-RMSE after physical integration

Expected output:
    K3_kseg_viterbi_rmse : < 3.02 ft  (K3 oracle ceiling)
    K5_kseg_viterbi_rmse : < 1.82 ft  (K5 oracle ceiling)

Run:
    python train_ksegment_offset.py [--ks 3 5] [--data-dir DATA_DIR]
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
from mtpnet.eval import run_oof_kseg, run_c0_baseline, save_oof_result
from mtpnet.models import train_offset_classifier
from mtpnet.metrics import print_scoreboard


def build_model_fn(backend="lgbm", params=None):
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
    parser = argparse.ArgumentParser(description="Experiments B: K-segment classifiers")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/Users/alexander/Desktop/rogii/MTPNet/data/train"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/mtpnet_cache"))
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 5],
                        help="Which K values to run (e.g. --ks 3 5 15)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument("--backend", choices=["lgbm", "catboost"], default="lgbm")
    parser.add_argument("--decoder", choices=["viterbi", "beam", "greedy"], default="viterbi")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Viterbi/beam transition smoothness penalty")
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
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

    # ---- Compute oracles ----
    all_ks_needed = [k for k in args.ks if any(k not in s.kseg_offset_star for s in train_samples)]
    if all_ks_needed:
        print(f"\nComputing K-segment oracles for K={all_ks_needed} …")
        compute_and_attach_oracles(train_samples, ks=all_ks_needed, verbose=True)
    else:
        print("Oracle labels already present.")

    # ---- Folds ----
    folds = make_group_kfold(train_samples, n_folds=args.n_folds)
    print(f"Folds: {args.n_folds}")

    # ---- c0 baseline ----
    print("\n--- c0 baseline ---")
    c0_rmse = run_c0_baseline(train_samples, verbose=True)
    scoreboard = {"c0_baseline": c0_rmse}

    # ---- Run each K ----
    for K in args.ks:
        exp_name = f"kseg_K{K}_{args.backend}_{args.decoder}"
        print(f"\n--- K={K}: [{args.backend}, decoder={args.decoder}, alpha={args.alpha}] ---")

        result = run_oof_kseg(
            train_samples,
            folds,
            build_model_fn(args.backend),
            K=K,
            experiment=exp_name,
            decoder=args.decoder,
            alpha=args.alpha,
            beam_width=args.beam_width,
            verbose=True,
        )

        scoreboard[exp_name] = result.pooled_rmse

        out_path = args.output_dir / f"{exp_name}.pkl"
        save_oof_result(result, str(out_path))
        print(f"Saved → {out_path}")

    # ---- Final scoreboard ----
    print_scoreboard(scoreboard, title=f"Experiments B Results  [decoder={args.decoder}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
