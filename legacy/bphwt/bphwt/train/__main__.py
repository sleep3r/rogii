from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from bphwt.clearml_utils import (
    apply_runtime_overrides,
    close_clearml_task,
    init_clearml_task,
    log_training_outputs,
    resolve_clearml_data_dir,
)
from bphwt.config import load_config, save_config
from bphwt.data.build_cache import build_cache, cache_status
from bphwt.train.train_fold import run_cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="BPHWT training")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild of feature cache even if it exists",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    apply_runtime_overrides(cfg)
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_dir / "config.yml"
    save_config(cfg, config_snapshot_path)

    clearml_task = init_clearml_task(cfg, config_snapshot_path)
    try:
        resolve_clearml_data_dir(cfg, task=clearml_task)
        save_config(cfg, config_snapshot_path)

        if args.rebuild_cache:
            os.environ["BPHWT_REBUILD_CACHE"] = "1"

        status = cache_status(cfg, split="train")
        if args.rebuild_cache or not status.valid:
            if not args.rebuild_cache:
                logger.info("Train cache invalid: %s", status.reason)
            logger.info("Building train feature cache...")
            build_cache(cfg, split="train")
        else:
            logger.info(
                "Train cache valid: %s wells from %s (%s extra npz ignored)",
                status.meta_count,
                cfg.resolved_cache_dir() / "meta_train.csv",
                status.extra_npz_count,
            )

        logger.info("Starting cross-validation training...")
        summary = run_cv(cfg)
        log_training_outputs(clearml_task, cfg, output_dir, summary)
        logger.info(
            "\n%s\nTraining complete\nCV RMSE: %.4f +/- %.4f\nRun `make error-analysis` for per-well diagnostics.\n%s",
            "=" * 50,
            summary["oof_rmse_mean"],
            summary["oof_rmse_std"],
            "=" * 50,
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
