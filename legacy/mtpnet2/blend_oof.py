"""
blend_oof.py  —  Ensemble / blend OOF predictions from multiple runs.

Loads two or more oof_predictions.pkl files, blends predictions per-well
(weighted average in TVT space), and reports pooled RMSE for each run
and all blend weightings.

Usage:
    # Equal blend of Run 1 and Run 2:
    python blend_oof.py \\
        artifacts/dlmtp/run1/oof_predictions.pkl \\
        artifacts/dlmtp/run2/oof_predictions.pkl

    # Named runs with custom weights:
    python blend_oof.py \\
        artifacts/dlmtp/run1/oof_predictions.pkl \\
        artifacts/dlmtp/run2/oof_predictions.pkl \\
        artifacts/dlmtp/run3/oof_predictions.pkl \\
        --tags R1 R2 R3 \\
        --weights 0.3 0.4 0.3
"""
from __future__ import annotations
import argparse, pickle
import numpy as np
import sys
sys.path.insert(0, '.')
from mtpnet.metrics import row_rmse


def load_oof(path: str) -> tuple[dict, dict, float]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["oof_preds"], d["oof_trues"], d.get("oof_rmse", float("nan"))


def blend(preds_list: list[dict], weights: list[float]) -> dict:
    """Weighted average of per-well predictions. Handles different-length wells."""
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    well_ids = sorted(set.intersection(*[set(p.keys()) for p in preds_list]))
    blended = {}
    for wid in well_ids:
        arrs = [p[wid] for p in preds_list]
        min_len = min(len(a) for a in arrs)
        blended[wid] = sum(w * a[:min_len] for w, a in zip(weights, arrs))
    return blended


def pooled_rmse(preds: dict, trues: dict) -> float:
    common = sorted(set(preds) & set(trues))
    ap = np.concatenate([preds[i] for i in common])
    at = np.concatenate([trues[i][:len(preds[i])] for i in common])
    return row_rmse(ap, at)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths",    nargs="+", help="oof_predictions.pkl files")
    parser.add_argument("--tags",   nargs="*", help="Labels for each run")
    parser.add_argument("--weights",nargs="*", type=float,
                        help="Optional blend weights (default: equal)")
    args = parser.parse_args()

    runs = [load_oof(p) for p in args.paths]
    tags = args.tags or [f"R{i+1}" for i in range(len(runs))]
    trues = runs[0][1]   # use trues from first run (should be identical)

    print(f"\n{'─'*55}")
    print(f"{'Run':<12}  OOF RMSE")
    print(f"{'─'*55}")
    for (preds, _, stored_rmse), tag in zip(runs, tags):
        r = pooled_rmse(preds, trues)
        print(f"  {tag:<10}  {r:.4f} ft  (stored={stored_rmse:.4f})")

    # Grid-search over blend ratios for 2-run case
    if len(runs) == 2:
        print(f"\n{'─'*55}")
        print(f"Blend grid (R1 weight → R2 weight):")
        preds0, preds1 = runs[0][0], runs[1][0]
        best_w, best_r = 0.5, 999.
        for w in np.arange(0.0, 1.01, 0.1):
            b = blend([preds0, preds1], [w, 1 - w])
            r = pooled_rmse(b, trues)
            marker = " ←" if r < best_r else ""
            if r < best_r:
                best_w, best_r = w, r
            print(f"  w={w:.1f}/{1-w:.1f}  RMSE={r:.4f} ft{marker}")
        print(f"\nBest blend: {tags[0]}×{best_w:.1f} + {tags[1]}×{1-best_w:.1f}"
              f"  →  {best_r:.4f} ft")

    # Custom weights
    if args.weights:
        w = args.weights
        if len(w) != len(runs):
            print(f"Error: {len(w)} weights but {len(runs)} runs"); return
        b = blend([r[0] for r in runs], w)
        r = pooled_rmse(b, trues)
        tag = "+".join(f"{t}×{wi:.2f}" for t, wi in zip(tags, w))
        print(f"\nCustom blend [{tag}]: {r:.4f} ft")

    # Equal blend (always print)
    if len(runs) > 1:
        eq = blend([r[0] for r in runs], [1.0] * len(runs))
        r  = pooled_rmse(eq, trues)
        print(f"\nEqual blend ({'/'.join(tags)}): {r:.4f} ft")

    print()


if __name__ == "__main__":
    main()
