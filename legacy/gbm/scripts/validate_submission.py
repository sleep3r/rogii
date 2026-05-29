#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--expected-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.submission.exists():
        raise FileNotFoundError(args.submission)

    with args.submission.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["id", "tvt"]:
            raise ValueError(f"Expected columns ['id', 'tvt'], got {reader.fieldnames}")

        rows = 0
        min_tvt = math.inf
        max_tvt = -math.inf
        for row in reader:
            rows += 1
            if not row["id"]:
                raise ValueError(f"Empty id at row {rows}")
            try:
                tvt = float(row["tvt"])
            except ValueError as exc:
                raise ValueError(f"Non-numeric tvt at row {rows}: {row['tvt']!r}") from exc
            if not math.isfinite(tvt):
                raise ValueError(f"Non-finite tvt at row {rows}: {row['tvt']!r}")
            min_tvt = min(min_tvt, tvt)
            max_tvt = max(max_tvt, tvt)

    if args.expected_rows and rows != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, got {rows}")
    if rows == 0:
        raise ValueError("Submission has no rows")

    print(
        f"OK submission rows={rows} min_tvt={min_tvt:.6f} max_tvt={max_tvt:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
