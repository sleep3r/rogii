#!/usr/bin/env python3
"""Validate oracle numbers against established benchmarks.

Expected output:
    global_grid_oracle  : ~7.64 ft
    K1_oracle           : ~7.59 ft
    K3_oracle           : ~3.02 ft
    K5_oracle           : ~1.82 ft
    K15_oracle          : ~0.65 ft

Run:
    python make_oracle_report.py [--data-dir DATA_DIR] [--cache-dir CACHE_DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---- project root on sys.path ----
sys.path.insert(0, str(Path(__file__).parent))

from mtpnet.data import load_offset_samples
from mtpnet.oracle import compute_oracle_report


# Expected benchmark values (from audit of rogii/MTPNet/mtpnet/k_segment_offset.py)
EXPECTED: dict[str, tuple[float, float]] = {
    "global_grid_oracle_rmse": (7.0,  8.5),   # expect ~7.64
    "K1_oracle_rmse":          (7.0,  8.5),   # expect ~7.59
    "K3_oracle_rmse":          (2.5,  3.5),   # expect ~3.02
    "K5_oracle_rmse":          (1.5,  2.2),   # expect ~1.82
    "K15_oracle_rmse":         (0.4,  0.9),   # expect ~0.65
}

TOLERANCE_REL = 0.15   # ±15% tolerance around expected midpoint


def check_value(key: str, value: float) -> bool:
    if key not in EXPECTED:
        return True
    lo, hi = EXPECTED[key]
    if lo <= value <= hi:
        print(f"  ✓  {key:<35s} = {value:.4f}  [expected {lo:.2f}–{hi:.2f}]")
        return True
    else:
        mid = (lo + hi) / 2
        print(f"  ✗  {key:<35s} = {value:.4f}  [expected {lo:.2f}–{hi:.2f}]  ← OUT OF RANGE")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate oracle benchmark numbers")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/alexander/Desktop/rogii/MTPNet/data/train"),
        help="Directory with *__horizontal_well.csv files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/mtpnet_cache"),
        help="Cache directory for loaded samples",
    )
    parser.add_argument(
        "--k-wells",
        type=int,
        default=-1,
        help="Load only first k wells (for quick smoke test; -1 = all)",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=Path("artifacts/oracle_report.json"),
        help="Where to save the JSON report",
    )
    args = parser.parse_args()

    cache_path = args.cache_dir / "samples.pkl" if args.k_wells < 0 else None
    print(f"Loading wells from {args.data_dir} …")

    samples = load_offset_samples(
        args.data_dir,
        k_wells=args.k_wells,
        cache_path=cache_path,
        verbose=True,
    )

    train_samples = [s for s in samples if s.has_true]
    print(f"Training wells: {len(train_samples)}")

    report = compute_oracle_report(
        train_samples,
        ks=[1, 3, 5, 15],
        verbose=True,
    )

    print("\n=== Benchmark Validation ===")
    all_pass = True
    for key in [
        "global_grid_oracle_rmse",
        "K1_oracle_rmse",
        "K3_oracle_rmse",
        "K5_oracle_rmse",
        "K15_oracle_rmse",
    ]:
        val = report.get(key, float("nan"))
        ok = check_value(key, val)
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("All oracle benchmarks PASSED.")
    else:
        print("Some oracle benchmarks FAILED — check data or oracle implementation.")

    # Save report (without per_well detail for brevity)
    save_report = {k: v for k, v in report.items() if k != "per_well"}
    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_report, "w") as f:
        json.dump(save_report, f, indent=2)
    print(f"Report saved → {args.save_report}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
