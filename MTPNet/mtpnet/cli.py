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

    rank_crossfit_parser = sub.add_parser(
        "rank-crossfit", help="Train OOF CatBoost MTP ranker folds on current MTP windows"
    )
    rank_crossfit_parser.add_argument("--run-dir", type=Path, required=True)
    rank_crossfit_parser.add_argument("--output-dir", type=Path)
    rank_crossfit_parser.add_argument("--n-folds", type=int, default=5)
    rank_crossfit_parser.add_argument("--seed", type=int, default=42)
    rank_crossfit_parser.add_argument(
        "--ranker-variant",
        choices=("conservative_regression", "pairwise"),
        default="conservative_regression",
    )

    track_parser = sub.add_parser(
        "track", help="Sequentially track MTP modes as particle realizations"
    )
    track_parser.add_argument("--run-dir", type=Path, required=True)
    track_parser.add_argument("--n-realizations", type=int, default=32)
    track_parser.add_argument("--keep-top", type=int, default=32)
    track_parser.add_argument("--merge-tolerance-ft", type=float, default=3.0)
    track_parser.add_argument("--overlap-penalty", type=float, default=0.10)
    track_parser.add_argument("--max-modes-per-window", type=int, default=8)
    track_parser.add_argument(
        "--logit-source", choices=("corr", "ranker", "ranker_oof", "nn"), default="ranker"
    )
    track_parser.add_argument("--ranker-logits", type=Path)
    track_parser.add_argument("--tau-ft", type=float, default=5.0)
    track_parser.add_argument("--ranker-beta", type=float, default=0.5)
    track_parser.add_argument("--corr-beta", type=float)

    track_audit_parser = sub.add_parser(
        "track-audit", help="Audit MTP tracker on ranker train/valid well splits"
    )
    track_audit_parser.add_argument("--run-dir", type=Path, required=True)
    track_audit_parser.add_argument("--n-realizations", type=int, default=32)
    track_audit_parser.add_argument("--keep-top", type=int, default=32)
    track_audit_parser.add_argument("--merge-tolerance-ft", type=float, default=3.0)
    track_audit_parser.add_argument("--overlap-penalty", type=float, default=0.10)
    track_audit_parser.add_argument("--max-modes-per-window", type=int, default=8)
    track_audit_parser.add_argument("--tau-ft", type=float, default=5.0)
    track_audit_parser.add_argument("--ranker-beta", type=float, default=0.5)

    oof_parser = sub.add_parser(
        "oof", help="Train fold-safe OOF MTP runs and track held-out wells"
    )
    oof_parser.add_argument("--config", type=Path, required=True)
    oof_parser.add_argument("--output-dir", type=Path)
    oof_parser.add_argument("--n-folds", type=int, default=5)
    oof_parser.add_argument("--max-folds", type=int)
    oof_parser.add_argument("--seed", type=int, default=42)
    oof_parser.add_argument(
        "--logit-source", choices=("corr", "ranker", "ranker_oof", "nn"), default="nn"
    )
    oof_parser.add_argument("--n-realizations", type=int, default=32)
    oof_parser.add_argument("--keep-top", type=int, default=32)
    oof_parser.add_argument("--merge-tolerance-ft", type=float, default=3.0)
    oof_parser.add_argument("--overlap-penalty", type=float, default=0.10)
    oof_parser.add_argument("--max-modes-per-window", type=int, default=8)
    oof_parser.add_argument("--corr-beta", type=float)
    oof_parser.add_argument(
        "--full-stitch",
        action="store_true",
        help="Run full stitch diagnostics instead of fast mode-window export",
    )
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
    if args.command == "rank-crossfit":
        from .ranker import run_ranker_crossfit

        run_ranker_crossfit(
            args.run_dir,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            seed=args.seed,
            ranker_variant=args.ranker_variant,
        )
        return
    if args.command == "track":
        from .track import run_tracker

        run_tracker(
            args.run_dir,
            n_realizations=args.n_realizations,
            keep_top=args.keep_top,
            merge_tolerance_ft=args.merge_tolerance_ft,
            overlap_penalty=args.overlap_penalty,
            max_modes_per_window=args.max_modes_per_window,
            logit_source=args.logit_source,
            ranker_logits=args.ranker_logits,
            tau_ft=args.tau_ft,
            ranker_beta=args.ranker_beta,
            corr_beta=args.corr_beta,
        )
        return
    if args.command == "track-audit":
        from .track import run_track_split_audit

        run_track_split_audit(
            args.run_dir,
            n_realizations=args.n_realizations,
            keep_top=args.keep_top,
            merge_tolerance_ft=args.merge_tolerance_ft,
            overlap_penalty=args.overlap_penalty,
            max_modes_per_window=args.max_modes_per_window,
            tau_ft=args.tau_ft,
            ranker_beta=args.ranker_beta,
        )
        return
    if args.command == "oof":
        from .oof import run_oof

        run_oof(
            args.config,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            max_folds=args.max_folds,
            seed=args.seed,
            logit_source=args.logit_source,
            n_realizations=args.n_realizations,
            keep_top=args.keep_top,
            merge_tolerance_ft=args.merge_tolerance_ft,
            overlap_penalty=args.overlap_penalty,
            max_modes_per_window=args.max_modes_per_window,
            corr_beta=args.corr_beta,
            full_stitch=args.full_stitch,
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
