from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from bphwt.config import load_config
from bphwt.data.build_cache import build_cache, cache_status
from bphwt.infer.make_submission import make_oof, make_submission
from bphwt.infer.oof_diagnostics import make_oof_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BPHWT inference")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--output_dir", default=None, help="Override output_dir from config")
    parser.add_argument("--submission_path", default="submission.csv")
    parser.add_argument("--oof", action="store_true", help="Generate OOF predictions")
    parser.add_argument("--oof_path", default="artifacts/oof.csv")
    parser.add_argument("--oof_diagnostics", action="store_true", help="Generate full-length OOF diagnostics")
    parser.add_argument("--oof_diag_path", default="artifacts/oof_diagnostics.csv")
    parser.add_argument("--oof_summary_path", default="artifacts/oof_diagnostics_summary.json")
    parser.add_argument("--rebuild-test-cache", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg.run.output_dir)
    if args.output_dir is not None:
        cfg.run.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_test_cache:
        os.environ["BPHWT_REBUILD_CACHE"] = "1"

    if args.oof:
        logger.info("Generating OOF predictions...")
        make_oof(cfg, output_dir, Path(args.oof_path))
        return

    if args.oof_diagnostics:
        logger.info("Generating OOF diagnostics...")
        make_oof_diagnostics(
            cfg=cfg,
            output_dir=output_dir,
            diag_path=Path(args.oof_diag_path),
            summary_path=Path(args.oof_summary_path),
        )
        return

    status = cache_status(cfg, split="test")
    if args.rebuild_test_cache or not status.valid:
        if not args.rebuild_test_cache:
            logger.info("Test cache invalid: %s", status.reason)
        logger.info("Building test feature cache...")
        build_cache(cfg, split="test")
    else:
        logger.info(
            "Test cache valid: %s wells from %s (%s extra npz ignored)",
            status.meta_count,
            cfg.resolved_cache_dir() / "meta_test.csv",
            status.extra_npz_count,
        )

    logger.info("Generating submission...")
    make_submission(cfg=cfg, output_dir=output_dir, submission_path=Path(args.submission_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()
