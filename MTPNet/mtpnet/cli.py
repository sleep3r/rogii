from __future__ import annotations

import argparse
from pathlib import Path

from .io import copy_data_tree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTPNet research prototype CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    copy_parser = sub.add_parser("copy-data", help="Copy Kaggle-style data into this project")
    copy_parser.add_argument("--source", type=Path, required=True)
    copy_parser.add_argument("--target", type=Path, required=True)

    train_parser = sub.add_parser("train", help="Train an MTPNet run")
    train_parser.add_argument("--config", type=Path, required=True)

    eval_parser = sub.add_parser("eval", help="Evaluate a trained MTPNet run")
    eval_parser.add_argument("--run-dir", type=Path, required=True)

    stitch_parser = sub.add_parser(
        "stitch", help="Stitch window predictions into row-level OOF candidates"
    )
    stitch_parser.add_argument("--run-dir", type=Path, required=True)

    rank_parser = sub.add_parser(
        "rank", help="Train CatBoost MTP mode ranker on stitched window modes"
    )
    rank_parser.add_argument("--run-dir", type=Path, required=True)
    rank_parser.add_argument("--seed", type=int, default=42)
    rank_parser.add_argument("--valid-fraction", type=float, default=0.35)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "copy-data":
        copy_data_tree(args.source, args.target)
        print(f"Copied data from {args.source} to {args.target}", flush=True)
        return
    if args.command == "train":
        from .train import train_from_config

        train_from_config(args.config)
        return
    if args.command == "eval":
        from .eval import evaluate_run

        evaluate_run(args.run_dir)
        return
    if args.command == "stitch":
        from .stitch import run_stitch

        run_stitch(args.run_dir)
        return
    if args.command == "rank":
        from .ranker import run_ranker

        run_ranker(
            args.run_dir,
            seed=args.seed,
            valid_fraction=args.valid_fraction,
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
