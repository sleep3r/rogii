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

    tail_audit_parser = sub.add_parser(
        "tail-audit", help="Build well-level tail/worst-well audit from row candidates"
    )
    tail_audit_parser.add_argument("--run-dir", type=Path)
    tail_audit_parser.add_argument("--candidates", type=Path)
    tail_audit_parser.add_argument("--output-dir", type=Path)
    tail_audit_parser.add_argument("--primary-candidate")
    tail_audit_parser.add_argument("--top-n", type=int, default=30)

    residual_parser = sub.add_parser(
        "residual-stack",
        help="Train fold-safe schema-safe residual stack over compressed hidden steps",
    )
    residual_parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    residual_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/residual_stack_v0")
    )
    residual_parser.add_argument("--rows-per-step", type=int, default=32)
    residual_parser.add_argument("--n-folds", type=int, default=5)
    residual_parser.add_argument("--seed", type=int, default=42)
    residual_parser.add_argument("--k-wells", type=int, default=-1)
    residual_parser.add_argument("--iterations", type=int, default=700)
    residual_parser.add_argument("--learning-rate", type=float, default=0.04)
    residual_parser.add_argument("--depth", type=int, default=6)

    candidate_bank_parser = sub.add_parser(
        "candidate-bank",
        help="Build expanded schema-safe candidate bank and strict tail oracle audit",
    )
    candidate_bank_parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    candidate_bank_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/candidate_bank_v2")
    )
    candidate_bank_parser.add_argument("--residual-predictions", type=Path)
    candidate_bank_parser.add_argument("--primary-candidate", default="residual_stack_v0")
    candidate_bank_parser.add_argument("--k-wells", type=int, default=-1)
    candidate_bank_parser.add_argument(
        "--streaming-oracle",
        action="store_true",
        help="Evaluate full candidate-bank oracle without writing long candidate_bank.parquet",
    )

    candidate_selector_parser = sub.add_parser(
        "candidate-selector",
        help="Train fold-safe deployable selector over expanded candidate-bank paths",
    )
    candidate_selector_parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    candidate_selector_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/candidate_selector_v0")
    )
    candidate_selector_parser.add_argument("--residual-predictions", type=Path)
    candidate_selector_parser.add_argument("--n-folds", type=int, default=5)
    candidate_selector_parser.add_argument("--seed", type=int, default=42)
    candidate_selector_parser.add_argument("--k-wells", type=int, default=-1)
    candidate_selector_parser.add_argument("--iterations", type=int, default=900)
    candidate_selector_parser.add_argument("--learning-rate", type=float, default=0.035)
    candidate_selector_parser.add_argument("--depth", type=int, default=5)
    candidate_selector_parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    candidate_selector_parser.add_argument("--progress-every", type=int, default=50)
    candidate_selector_parser.add_argument(
        "--mode", choices=["regressor", "ranker_guard"], default="regressor"
    )
    candidate_selector_parser.add_argument("--ranker-loss", default="YetiRankPairwise")
    candidate_selector_parser.add_argument("--guard-gain-threshold", type=float, default=0.0)
    candidate_selector_parser.add_argument(
        "--dangerous-guard-gain-threshold", type=float, default=5.0
    )

    corr_panel_parser = sub.add_parser(
        "corr-panel",
        help="Build webinar-style GR/typewell correlation panel diagnostics",
    )
    corr_panel_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    corr_panel_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corr_panel_v0"))
    corr_panel_parser.add_argument("--rows-per-step", type=int, default=32)
    corr_panel_parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    corr_panel_parser.add_argument("--patch-radius", type=int, default=3)
    corr_panel_parser.add_argument(
        "--patch-radii",
        default="",
        help="Comma-separated multiscale patch radii, e.g. 2,4,8,16",
    )
    corr_panel_parser.add_argument(
        "--stretch-factors",
        default="1.0",
        help="Comma-separated vertical stretch factors, e.g. 0.5,0.75,1.0,1.25,1.5",
    )
    corr_panel_parser.add_argument(
        "--anchor-source",
        choices=("none", "known_tail_linear"),
        default="none",
    )
    corr_panel_parser.add_argument("--search-radius-ft", type=float)
    corr_panel_parser.add_argument("--k-wells", type=int, default=-1)
    corr_panel_parser.add_argument("--seed", type=int, default=42)
    corr_panel_parser.add_argument(
        "--no-shuffled",
        action="store_true",
        help="Skip shuffled-GR sanity baseline",
    )

    mismatch_parser = sub.add_parser(
        "typewell-mismatch",
        help="Audit whether the true TVT path itself matches the provided typewell GR",
    )
    mismatch_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    mismatch_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/typewell_mismatch_v0")
    )
    mismatch_parser.add_argument("--rows-per-step", type=int, default=32)
    mismatch_parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    mismatch_parser.add_argument("--patch-radius", type=int, default=3)
    mismatch_parser.add_argument(
        "--patch-radii",
        default="",
        help="Comma-separated multiscale patch radii, e.g. 2,4,8,16",
    )
    mismatch_parser.add_argument(
        "--stretch-factors",
        default="1.0",
        help="Comma-separated vertical stretch factors, e.g. 0.5,0.75,1.0,1.25,1.5",
    )
    mismatch_parser.add_argument("--k-wells", type=int, default=-1)
    mismatch_parser.add_argument("--seed", type=int, default=42)

    true_path_gr_parser = sub.add_parser(
        "true-path-gr-audit",
        help="Audit whether GR/typewell score is strong at the true TVT path",
    )
    true_path_gr_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    true_path_gr_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/true_path_gr_audit_v0")
    )
    true_path_gr_parser.add_argument("--rows-per-step", type=int, default=32)
    true_path_gr_parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    true_path_gr_parser.add_argument("--patch-radius", type=int, default=3)
    true_path_gr_parser.add_argument(
        "--patch-radii",
        default="",
        help="Comma-separated multiscale patch radii, e.g. 2,4,8,16",
    )
    true_path_gr_parser.add_argument(
        "--stretch-factors",
        default="1.0",
        help="Comma-separated vertical stretch factors, e.g. 0.5,0.75,1.0,1.25,1.5",
    )
    true_path_gr_parser.add_argument("--k-wells", type=int, default=-1)
    true_path_gr_parser.add_argument("--seed", type=int, default=42)
    true_path_gr_parser.add_argument("--tail-classes-path", type=Path)

    formation_parser = sub.add_parser(
        "formation-corr",
        help="Compare global, oracle-geology, and anchor-geology typewell correlation",
    )
    formation_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    formation_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/formation_correlation_v0")
    )
    formation_parser.add_argument("--rows-per-step", type=int, default=32)
    formation_parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    formation_parser.add_argument("--patch-radius", type=int, default=3)
    formation_parser.add_argument(
        "--patch-radii",
        default="",
        help="Comma-separated multiscale patch radii, e.g. 2,4,8,16",
    )
    formation_parser.add_argument(
        "--stretch-factors",
        default="1.0",
        help="Comma-separated vertical stretch factors, e.g. 0.5,0.75,1.0,1.25,1.5",
    )
    formation_parser.add_argument("--k-wells", type=int, default=-1)
    formation_parser.add_argument("--seed", type=int, default=42)

    pseudo_zone_parser = sub.add_parser(
        "pseudo-zone",
        help="Cross-fit test-schema-safe pseudo-geology zones from train typewell labels",
    )
    pseudo_zone_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    pseudo_zone_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/pseudo_zone_v0")
    )
    pseudo_zone_parser.add_argument("--rows-per-step", type=int, default=32)
    pseudo_zone_parser.add_argument("--n-bins", type=int, default=128)
    pseudo_zone_parser.add_argument("--n-folds", type=int, default=5)
    pseudo_zone_parser.add_argument("--k-wells", type=int, default=-1)
    pseudo_zone_parser.add_argument("--seed", type=int, default=42)

    pseudo_zone_v1_parser = sub.add_parser(
        "pseudo-zone-v1",
        help="Build test-safe pseudo-zone predictions and masks for alignment diagnostics",
    )
    pseudo_zone_v1_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    pseudo_zone_v1_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/pseudo_zone_v1")
    )
    pseudo_zone_v1_parser.add_argument("--rows-per-step", type=int, default=32)
    pseudo_zone_v1_parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    pseudo_zone_v1_parser.add_argument("--n-bins", type=int, default=128)
    pseudo_zone_v1_parser.add_argument("--n-folds", type=int, default=5)
    pseudo_zone_v1_parser.add_argument("--k-wells", type=int, default=-1)
    pseudo_zone_v1_parser.add_argument("--seed", type=int, default=42)
    pseudo_zone_v1_parser.add_argument("--neural-epochs", type=int, default=6)
    pseudo_zone_v1_parser.add_argument("--neural-hidden-dim", type=int, default=64)
    pseudo_zone_v1_parser.add_argument("--neural-batch-size", type=int, default=2048)
    pseudo_zone_v1_parser.add_argument("--neural-lr", type=float, default=1.0e-3)
    pseudo_zone_v1_parser.add_argument(
        "--patch-radii",
        default="2,4,8,16",
        help="Comma-separated correlation patch radii",
    )
    pseudo_zone_v1_parser.add_argument(
        "--stretch-factors",
        default="0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated vertical stretch factors",
    )
    pseudo_zone_v1_parser.add_argument("--known-tail-radius-ft", type=float, default=120.0)
    pseudo_zone_v1_parser.add_argument("--soft-zone-weight", type=float, default=0.5)

    pseudo_zone_corr_parser = sub.add_parser(
        "pseudo-zone-corr",
        help="Run normal-vs-shuffled correlation inside pseudo-zone masks",
    )
    pseudo_zone_corr_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    pseudo_zone_corr_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/pseudo_zone_v1")
    )
    pseudo_zone_corr_parser.add_argument("--rows-per-step", type=int, default=32)
    pseudo_zone_corr_parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    pseudo_zone_corr_parser.add_argument("--n-bins", type=int, default=128)
    pseudo_zone_corr_parser.add_argument("--n-folds", type=int, default=5)
    pseudo_zone_corr_parser.add_argument("--k-wells", type=int, default=-1)
    pseudo_zone_corr_parser.add_argument("--seed", type=int, default=42)
    pseudo_zone_corr_parser.add_argument("--neural-epochs", type=int, default=6)
    pseudo_zone_corr_parser.add_argument("--neural-hidden-dim", type=int, default=64)
    pseudo_zone_corr_parser.add_argument("--neural-batch-size", type=int, default=2048)
    pseudo_zone_corr_parser.add_argument("--neural-lr", type=float, default=1.0e-3)
    pseudo_zone_corr_parser.add_argument(
        "--patch-radii",
        default="2,4,8,16",
        help="Comma-separated correlation patch radii",
    )
    pseudo_zone_corr_parser.add_argument(
        "--stretch-factors",
        default="0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated vertical stretch factors",
    )
    pseudo_zone_corr_parser.add_argument("--known-tail-radius-ft", type=float, default=120.0)
    pseudo_zone_corr_parser.add_argument("--soft-zone-weight", type=float, default=0.5)

    ga_zone_eval_parser = sub.add_parser(
        "ga-zone-eval",
        help="Summarize GeoAligner/DP proxy metrics for pseudo-zone correlation variants",
    )
    ga_zone_eval_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    ga_zone_eval_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/pseudo_zone_v1")
    )
    ga_zone_eval_parser.add_argument("--rows-per-step", type=int, default=32)
    ga_zone_eval_parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    ga_zone_eval_parser.add_argument("--n-bins", type=int, default=128)
    ga_zone_eval_parser.add_argument("--n-folds", type=int, default=5)
    ga_zone_eval_parser.add_argument("--k-wells", type=int, default=-1)
    ga_zone_eval_parser.add_argument("--seed", type=int, default=42)
    ga_zone_eval_parser.add_argument("--neural-epochs", type=int, default=6)
    ga_zone_eval_parser.add_argument("--neural-hidden-dim", type=int, default=64)
    ga_zone_eval_parser.add_argument("--neural-batch-size", type=int, default=2048)
    ga_zone_eval_parser.add_argument("--neural-lr", type=float, default=1.0e-3)
    ga_zone_eval_parser.add_argument(
        "--patch-radii",
        default="2,4,8,16",
        help="Comma-separated correlation patch radii",
    )
    ga_zone_eval_parser.add_argument(
        "--stretch-factors",
        default="0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated vertical stretch factors",
    )
    ga_zone_eval_parser.add_argument("--known-tail-radius-ft", type=float, default=120.0)
    ga_zone_eval_parser.add_argument("--soft-zone-weight", type=float, default=0.5)

    diagnostic_parser = sub.add_parser(
        "diagnostic-report",
        help="Build a research-to-real-wells diagnostic Markdown report with figures",
    )
    diagnostic_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    diagnostic_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/diagnostic_report_v1")
    )

    pdf_parser = sub.add_parser(
        "diagnostic-pdf",
        help="Build a PDF diagnostic pack from diagnostic_report.md and all figures",
    )
    pdf_parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/diagnostic_report_v1/diagnostic_report.md"),
    )
    pdf_parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/diagnostic_report_v1/figures"),
    )
    pdf_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/diagnostic_report_v1/diagnostic_report.pdf"),
    )

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


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in text.split(",") if item.strip())


def _pseudo_zone_v1_config(args: argparse.Namespace):
    from .pseudo_zone_v1 import PseudoZoneV1Config

    return PseudoZoneV1Config(
        rows_per_step=args.rows_per_step,
        vertical_step_ft=args.vertical_step_ft,
        n_bins=args.n_bins,
        n_folds=args.n_folds,
        k_wells=args.k_wells,
        seed=args.seed,
        neural_epochs=args.neural_epochs,
        neural_hidden_dim=args.neural_hidden_dim,
        neural_batch_size=args.neural_batch_size,
        neural_lr=args.neural_lr,
        patch_radii=_parse_int_tuple(args.patch_radii),
        stretch_factors=_parse_float_tuple(args.stretch_factors),
        known_tail_radius_ft=args.known_tail_radius_ft,
        soft_zone_weight=args.soft_zone_weight,
    )


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
    if args.command == "tail-audit":
        from .tail_audit import run_tail_audit

        run_tail_audit(
            run_dir=args.run_dir,
            candidates_path=args.candidates,
            output_dir=args.output_dir,
            primary_candidate=args.primary_candidate,
            top_n=args.top_n,
        )
        return
    if args.command == "residual-stack":
        from .residual_stack import ResidualStackConfig, run_residual_stack

        run_residual_stack(
            ResidualStackConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                rows_per_step=args.rows_per_step,
                n_folds=args.n_folds,
                seed=args.seed,
                k_wells=args.k_wells,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
            )
        )
        return
    if args.command == "candidate-bank":
        from .candidate_bank import run_candidate_bank

        run_candidate_bank(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            residual_predictions_path=args.residual_predictions,
            primary_candidate=args.primary_candidate,
            k_wells=args.k_wells,
            streaming_oracle=args.streaming_oracle,
        )
        return
    if args.command == "candidate-selector":
        from .candidate_selector import CandidateSelectorConfig, run_candidate_selector

        run_candidate_selector(
            CandidateSelectorConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                residual_predictions_path=args.residual_predictions,
                n_folds=args.n_folds,
                seed=args.seed,
                k_wells=args.k_wells,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                progress_every=args.progress_every,
                mode=args.mode,
                ranker_loss=args.ranker_loss,
                guard_gain_threshold=args.guard_gain_threshold,
                dangerous_guard_gain_threshold=args.dangerous_guard_gain_threshold,
            )
        )
        return
    if args.command == "corr-panel":
        from .correlation_panel import run_correlation_panel

        run_correlation_panel(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            vertical_step_ft=args.vertical_step_ft,
            patch_radius=args.patch_radius,
            patch_radii=tuple(
                int(item)
                for item in args.patch_radii.split(",")
                if item.strip()
            ),
            stretch_factors=tuple(
                float(item)
                for item in args.stretch_factors.split(",")
                if item.strip()
            ),
            anchor_source=args.anchor_source,
            search_radius_ft=args.search_radius_ft,
            k_wells=args.k_wells,
            include_shuffled=not args.no_shuffled,
            seed=args.seed,
        )
        return
    if args.command == "typewell-mismatch":
        from .typewell_mismatch import run_typewell_mismatch_audit

        run_typewell_mismatch_audit(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            vertical_step_ft=args.vertical_step_ft,
            patch_radius=args.patch_radius,
            patch_radii=tuple(
                int(item)
                for item in args.patch_radii.split(",")
                if item.strip()
            ),
            stretch_factors=tuple(
                float(item)
                for item in args.stretch_factors.split(",")
                if item.strip()
            ),
            k_wells=args.k_wells,
            seed=args.seed,
        )
        return
    if args.command == "true-path-gr-audit":
        from .true_path_gr_audit import run_true_path_gr_audit

        run_true_path_gr_audit(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            vertical_step_ft=args.vertical_step_ft,
            patch_radius=args.patch_radius,
            patch_radii=tuple(
                int(item)
                for item in args.patch_radii.split(",")
                if item.strip()
            ),
            stretch_factors=tuple(
                float(item)
                for item in args.stretch_factors.split(",")
                if item.strip()
            ),
            k_wells=args.k_wells,
            seed=args.seed,
            tail_classes_path=args.tail_classes_path,
        )
        return
    if args.command == "formation-corr":
        from .formation_correlation import run_formation_correlation_audit

        run_formation_correlation_audit(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            vertical_step_ft=args.vertical_step_ft,
            patch_radius=args.patch_radius,
            patch_radii=tuple(
                int(item)
                for item in args.patch_radii.split(",")
                if item.strip()
            ),
            stretch_factors=tuple(
                float(item)
                for item in args.stretch_factors.split(",")
                if item.strip()
            ),
            k_wells=args.k_wells,
            seed=args.seed,
        )
        return
    if args.command == "pseudo-zone":
        from .pseudo_zone import run_pseudo_zone_audit

        run_pseudo_zone_audit(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            n_bins=args.n_bins,
            n_folds=args.n_folds,
            k_wells=args.k_wells,
            seed=args.seed,
        )
        return
    if args.command == "pseudo-zone-v1":
        from .pseudo_zone_v1 import run_pseudo_zone_v1

        run_pseudo_zone_v1(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cfg=_pseudo_zone_v1_config(args),
        )
        return
    if args.command == "pseudo-zone-corr":
        from .pseudo_zone_v1 import run_pseudo_zone_corr

        run_pseudo_zone_corr(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cfg=_pseudo_zone_v1_config(args),
        )
        return
    if args.command == "ga-zone-eval":
        from .pseudo_zone_v1 import run_pseudo_zone_ga_eval

        run_pseudo_zone_ga_eval(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cfg=_pseudo_zone_v1_config(args),
        )
        return
    if args.command == "diagnostic-report":
        from .diagnostic_report import write_diagnostic_report

        report = write_diagnostic_report(
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
        )
        print(f"Wrote diagnostic report to {report}", flush=True)
        return
    if args.command == "diagnostic-pdf":
        from .pdf_report import write_diagnostic_pdf

        path = write_diagnostic_pdf(
            report_path=args.report_path,
            figures_dir=args.figures_dir,
            output_path=args.output_path,
        )
        print(f"Wrote diagnostic PDF to {path}", flush=True)
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
