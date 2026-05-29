#!/usr/bin/env python3
"""Full OOF scoreboard: aggregate all experiment results into one table.

Loads all saved OOF result .pkl files from artifacts/results/ and produces:
  - artifacts/oof_scoreboard.csv  — full table
  - Printed scoreboard

Run:
    python make_oof_table.py [--result-dir artifacts/results]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from mtpnet.eval import load_oof_result
from mtpnet.metrics import print_scoreboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full OOF scoreboard")
    parser.add_argument("--result-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/oof_scoreboard.csv"))
    parser.add_argument("--oracle-report", type=Path, default=Path("artifacts/oracle_report.json"))
    args = parser.parse_args()

    rows = []

    # Oracle ceilings from report
    if args.oracle_report.exists():
        import json
        with open(args.oracle_report) as f:
            oracle = json.load(f)
        for key, val in oracle.items():
            if key.endswith("_rmse") and isinstance(val, (int, float)):
                rows.append({
                    "experiment": f"[oracle] {key}",
                    "pooled_rmse": float(val),
                    "type": "oracle",
                    "n_folds": None,
                    "fold_rmses": None,
                })

    # OOF results
    pkl_files = sorted(args.result_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} OOF result files in {args.result_dir}")

    for path in pkl_files:
        try:
            result = load_oof_result(str(path))
            rows.append({
                "experiment":  result.experiment,
                "pooled_rmse": result.pooled_rmse,
                "type":        f"{result.mode}_K{result.K}",
                "decoder":     result.decoder,
                "alpha":       result.alpha,
                "n_folds":     len(result.fold_rmses),
                "fold_rmses":  str([f"{r:.4f}" for r in result.fold_rmses]),
            })
            print(f"  {result.experiment:<50s}  rmse={result.pooled_rmse:.4f}")
        except Exception as e:
            print(f"  [warn] Could not load {path.name}: {e}")

    if not rows:
        print("No results found. Run experiments first.")
        return 1

    df = pd.DataFrame(rows).sort_values("pooled_rmse").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved scoreboard → {args.output}")

    # Print scoreboard
    scoreboard = {
        row["experiment"]: row["pooled_rmse"]
        for _, row in df.iterrows()
    }
    print_scoreboard(scoreboard, title="Full OOF Scoreboard")

    return 0


if __name__ == "__main__":
    sys.exit(main())
