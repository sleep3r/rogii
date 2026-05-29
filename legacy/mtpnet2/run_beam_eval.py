#!/usr/bin/env python3
"""Experiments D/E: Beam-search decoder comparison on K-segment results.

Loads saved OOF results and re-decodes with different alpha values and beam
widths to find the best decoder configuration.

Also performs a sweep over alpha to find the sweet spot for Viterbi.

Run:
    python run_beam_eval.py --result-dir artifacts/results [--ks 3 5]
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
from mtpnet.eval import run_oof_kseg, run_c0_baseline, save_oof_result, load_oof_result
from mtpnet.models import train_offset_classifier
from mtpnet.decode import decode_kseg_predictions, make_log_transition
from mtpnet.metrics import row_rmse, print_scoreboard


# ---------------------------------------------------------------------------
# Re-decode from saved proba arrays
# ---------------------------------------------------------------------------

def redecode_oof_result(
    result,           # OOFResult (loaded from disk, has all_proba + samples)
    samples: list,
    K: int,
    grid: np.ndarray,
    decoder: str = "viterbi",
    alpha: float = 10.0,
    beam_width: int = 32,
) -> float:
    """Re-run decoding on saved OOF proba arrays with new decoder settings.

    Returns pooled row RMSE.
    """
    from mtpnet.decode import decode_kseg_predictions
    preds = decode_kseg_predictions(
        samples,
        result.all_proba,   # list of (K, N_BINS) arrays
        K=K,
        grid=grid,
        decoder=decoder,
        alpha=alpha,
        beam_width=beam_width,
    )
    trues = [s.tvt_hidden_true for s in samples]
    all_p = np.concatenate([p.astype(np.float64) for p in preds])
    all_t = np.concatenate([t.astype(np.float64) for t in trues])
    return row_rmse(all_p, all_t)


# ---------------------------------------------------------------------------
# Alpha sweep
# ---------------------------------------------------------------------------

def alpha_sweep(
    result,
    samples: list,
    K: int,
    grid: np.ndarray,
    decoder: str = "viterbi",
    alphas: list[float] | None = None,
) -> dict[float, float]:
    """Sweep alpha values and return {alpha: rmse} dict."""
    if alphas is None:
        alphas = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    results = {}
    for a in alphas:
        rmse = redecode_oof_result(result, samples, K, grid, decoder=decoder, alpha=a)
        results[a] = rmse
        print(f"  alpha={a:6.1f}  {decoder}_rmse={rmse:.4f}")

    best_alpha = min(results, key=results.get)
    print(f"  → best alpha={best_alpha}  rmse={results[best_alpha]:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Experiments D/E: Beam decoder sweep")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/Users/alexander/Desktop/rogii/MTPNet/data/train"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/mtpnet_cache"))
    parser.add_argument("--result-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument("--backend", choices=["lgbm", "catboost"], default="lgbm")
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.0, 1.0, 5.0, 10.0, 20.0, 50.0],
    )
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()

    cache_path = args.cache_dir / "samples.pkl" if args.k_wells < 0 else None
    print(f"\nLoading wells …")
    samples = load_offset_samples(
        args.data_dir, k_wells=args.k_wells, cache_path=cache_path, verbose=True
    )
    train_samples = [s for s in samples if s.has_true]
    print(f"Training wells: {len(train_samples)}")

    grid = make_offset_grid()
    folds = make_group_kfold(train_samples, n_folds=args.n_folds)
    scoreboard: dict[str, float] = {}

    # c0 baseline
    c0_rmse = run_c0_baseline(train_samples, verbose=True)
    scoreboard["c0_baseline"] = c0_rmse

    for K in args.ks:
        # Try to load saved result (avoid re-training)
        result_path = args.result_dir / f"kseg_K{K}_{args.backend}_viterbi.pkl"
        if result_path.exists():
            print(f"\nLoading saved result: {result_path}")
            result = load_oof_result(str(result_path))
        else:
            print(f"\nNo saved result for K={K}; training fresh …")
            # Compute oracles if needed
            if any(K not in s.kseg_offset_star for s in train_samples):
                compute_and_attach_oracles(train_samples, ks=[K], verbose=True)

            def _build(X_tr, y_tr, X_va, y_va):
                return train_offset_classifier(
                    X_tr, y_tr, X_va, y_va,
                    backend=args.backend,
                    early_stopping_rounds=50,
                    verbose=False,
                )

            result = run_oof_kseg(
                train_samples, folds, _build,
                K=K, decoder="viterbi", alpha=10.0, verbose=True,
            )
            save_oof_result(result, str(result_path))

        # Check that proba arrays are saved
        if not result.all_proba or result.all_proba[0] is None:
            print(f"  [warn] No proba arrays in saved result for K={K}; skipping sweep")
            continue

        # Alpha sweep — Viterbi
        print(f"\n  === K={K}: Viterbi alpha sweep ===")
        vit_sweep = alpha_sweep(result, train_samples, K, grid, decoder="viterbi", alphas=args.alphas)
        best_alpha = min(vit_sweep, key=vit_sweep.get)
        scoreboard[f"K{K}_viterbi_best_alpha={best_alpha}"] = vit_sweep[best_alpha]

        # Beam width sweep with best alpha
        print(f"\n  === K={K}: Beam width sweep (alpha={best_alpha}) ===")
        for bw in args.beam_widths:
            rmse = redecode_oof_result(
                result, train_samples, K, grid,
                decoder="beam", alpha=best_alpha, beam_width=bw,
            )
            print(f"  beam_width={bw:3d}  rmse={rmse:.4f}")
            scoreboard[f"K{K}_beam_bw{bw}_a{best_alpha}"] = rmse

    print_scoreboard(scoreboard, title="Experiments D/E Results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
