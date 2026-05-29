"""Training entry point.

Usage:
    python -m bphwt.train configs/bphwt_lite.yml [--rebuild-cache]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="BPHWT training")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="Force rebuild of feature cache even if it exists"
    )
    args = parser.parse_args(argv)

    from bphwt.config import load_config

    cfg = load_config(args.config)
    from bphwt.clearml_utils import (
        apply_runtime_overrides,
        close_clearml_task,
        init_clearml_task,
        log_training_outputs,
        resolve_clearml_data_dir,
    )

    apply_runtime_overrides(cfg)

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    from bphwt.config import save_config

    config_snapshot_path = output_dir / "config.yml"
    save_config(cfg, config_snapshot_path)

    clearml_task = init_clearml_task(cfg, config_snapshot_path)
    try:
        resolve_clearml_data_dir(cfg, task=clearml_task)
        save_config(cfg, config_snapshot_path)

        import os

        if args.rebuild_cache:
            os.environ["BPHWT_REBUILD_CACHE"] = "1"

        # ---- Step 1: Build cache ----------------------------------------
        cache_dir = cfg.resolved_cache_dir()
        from bphwt.data.build_cache import cache_status

        status = cache_status(cfg, split="train")
        if args.rebuild_cache or not status.valid:
            if not args.rebuild_cache:
                logger.info("Train cache invalid: %s", status.reason)
            logger.info("Building train feature cache...")
            from bphwt.data.build_cache import build_cache

            build_cache(cfg, split="train")
        else:
            logger.info(
                "Train cache valid: %s wells from %s (%s extra npz ignored)",
                status.meta_count,
                cache_dir / "meta_train.csv",
                status.extra_npz_count,
            )

        # ---- Step 2: Run CV training ------------------------------------
        logger.info("Starting cross-validation training...")
        from bphwt.train.train_fold import run_cv

        summary = run_cv(cfg)
        log_training_outputs(clearml_task, cfg, output_dir, summary)

        logger.info(
            f"\n{'=' * 50}\n"
            f"Training complete\n"
            f"CV RMSE: {summary['oof_rmse_mean']:.4f} ± {summary['oof_rmse_std']:.4f}\n"
            f"Run `make error-analysis` for per-well diagnostics.\n"
            f"{'=' * 50}"
        )
        return summary
    except BaseException as exc:
        close_clearml_task(clearml_task, failed=True, status_message=f"{type(exc).__name__}: {exc}")
        clearml_task = None
        raise
    finally:
        close_clearml_task(clearml_task)


if __name__ == "__main__":
    main()
