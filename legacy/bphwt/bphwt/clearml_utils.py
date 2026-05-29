"""ClearML integration helpers for server training."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def apply_runtime_overrides(cfg) -> None:
    """Apply BPHWT_* environment overrides to a loaded config in-place."""
    _set_bool_env("BPHWT_TRACKING_ENABLED", lambda v: setattr(cfg.tracking, "enabled", v))
    _set_str_env("BPHWT_TRACKING_PROJECT", lambda v: setattr(cfg.tracking, "project", v))
    _set_str_env("BPHWT_TRACKING_TASK_NAME", lambda v: setattr(cfg.tracking, "task_name", v))
    _set_str_env("BPHWT_TRACKING_OUTPUT_URI", lambda v: setattr(cfg.tracking, "output_uri", v))
    _set_bool_env("BPHWT_TRACKING_LOG_ARTIFACTS", lambda v: setattr(cfg.tracking, "log_artifacts", v))
    _set_bool_env("BPHWT_TRACKING_LOG_MODEL", lambda v: setattr(cfg.tracking, "log_model", v))
    _set_bool_env("BPHWT_TRACKING_FAIL_ON_ERROR", lambda v: setattr(cfg.tracking, "fail_on_error", v))
    if os.getenv("BPHWT_TRACKING_TAGS"):
        cfg.tracking.tags = [x.strip() for x in os.environ["BPHWT_TRACKING_TAGS"].split(",") if x.strip()]

    _set_bool_env("BPHWT_DATA_CLEARML_ENABLED", lambda v: setattr(cfg.data_clearml, "enabled", v))
    _set_str_env("BPHWT_DATA_CLEARML_PROJECT", lambda v: setattr(cfg.data_clearml, "project", v))
    _set_str_env("BPHWT_DATA_CLEARML_NAME", lambda v: setattr(cfg.data_clearml, "name", v))
    _set_str_env("BPHWT_DATA_CLEARML_VERSION", lambda v: setattr(cfg.data_clearml, "version", v))
    _set_str_env("BPHWT_DATA_CLEARML_CACHE_DIR", lambda v: setattr(cfg.data_clearml, "cache_dir", v))
    _set_int_env("BPHWT_DATA_CLEARML_MAX_WORKERS", lambda v: setattr(cfg.data_clearml, "max_workers", v))

    _set_str_env("BPHWT_DATA_DIR", lambda v: setattr(cfg, "data_dir", v))
    _set_str_env("BPHWT_OUTPUT_DIR", lambda v: setattr(cfg.run, "output_dir", v))
    _set_str_env("BPHWT_CACHE_DIR", lambda v: setattr(cfg, "cache_dir", v))
    _set_bool_env("BPHWT_DATA_RANDOM_TRAIN_CROP", lambda v: setattr(cfg.data, "random_train_crop", v))
    _set_int_env("BPHWT_DATA_VAL_MAX_SEQ_LEN", lambda v: setattr(cfg.data, "val_max_seq_len", v))
    _set_str_env("BPHWT_TRAIN_DEVICE", lambda v: setattr(cfg.train, "device", v))
    _set_int_env("BPHWT_K_WELLS", lambda v: setattr(cfg, "k_wells", v))


def init_clearml_task(cfg, config_path: Path | None = None):
    """Initialize ClearML Task when tracking.enabled is true."""
    if not cfg.tracking.enabled:
        return None

    try:
        from clearml import Task

        task_name = cfg.tracking.task_name or f"{cfg.run.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        task = Task.init(
            project_name=cfg.tracking.project,
            task_name=task_name,
            tags=cfg.tracking.tags,
            output_uri=cfg.tracking.output_uri or None,
            reuse_last_task_id=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_connect_streams=True,
        )
        task.connect(cfg.model_dump(), name="config")
        if config_path is not None and Path(config_path).exists():
            task.upload_artifact("config_yaml", artifact_object=str(config_path))
        logger.info("ClearML task initialized: project=%s task=%s", cfg.tracking.project, task_name)
        return task
    except Exception as exc:
        if cfg.tracking.fail_on_error:
            raise
        logger.warning("ClearML task initialization failed: %s", exc)
        return None


def resolve_clearml_data_dir(cfg, task=None) -> None:
    """Download/resolve ClearML Dataset and set cfg.data_dir in-place."""
    if not cfg.data_clearml.enabled:
        return

    try:
        _configure_clearml_cache(cfg.data_clearml.cache_dir)
        from clearml import Dataset

        dataset = Dataset.get(
            dataset_project=cfg.data_clearml.project,
            dataset_name=cfg.data_clearml.name,
            dataset_version=cfg.data_clearml.version,
            only_completed=True,
        )
        local_path = dataset.get_local_copy(max_workers=cfg.data_clearml.max_workers)
        cfg.data_dir = str(local_path)
        logger.info(
            "ClearML dataset resolved: project=%s name=%s version=%s path=%s",
            cfg.data_clearml.project,
            cfg.data_clearml.name,
            cfg.data_clearml.version,
            local_path,
        )
        if task is not None:
            task.connect(
                {
                    "dataset_project": cfg.data_clearml.project,
                    "dataset_name": cfg.data_clearml.name,
                    "dataset_version": cfg.data_clearml.version,
                    "dataset_local_path": str(local_path),
                },
                name="dataset",
            )
    except Exception:
        local_data_dir = Path(cfg.data_dir)
        if local_data_dir.exists() and not cfg.tracking.fail_on_error:
            logger.warning("ClearML dataset resolution failed; falling back to local data_dir=%s", local_data_dir)
            return
        raise


def log_training_outputs(task, cfg, output_dir: Path, summary: dict[str, Any]) -> None:
    """Log final metrics and artifacts to ClearML if a task exists."""
    if task is None:
        return

    output_dir = Path(output_dir)
    try:
        logger_obj = task.get_logger()
        logger_obj.report_single_value("cv_rmse_mean", float(summary.get("oof_rmse_mean", float("nan"))))
        logger_obj.report_single_value("cv_rmse_std", float(summary.get("oof_rmse_std", float("nan"))))
        for result in summary.get("fold_results", []):
            fold = int(result["fold"])
            logger_obj.report_scalar("cv", "val_rmse", value=float(result["val_rmse"]), iteration=fold)
            logger_obj.report_scalar("cv", "val_loss", value=float(result.get("val_loss", float("nan"))), iteration=fold)

        if cfg.tracking.log_artifacts:
            _upload_if_exists(task, "effective_config", output_dir / "config.yml")
            _upload_if_exists(task, "cv_summary", output_dir / "cv_summary.json")
            _upload_if_exists(task, "fold_summary", output_dir / "fold_summary.csv")
            for metrics_path in sorted(output_dir.glob("fold_*/metrics.csv")):
                task.upload_artifact(f"metrics/{metrics_path.parent.name}", artifact_object=str(metrics_path))

        if cfg.tracking.log_model:
            for ckpt_path in sorted(output_dir.glob("fold_*/best_ema.pt")):
                task.upload_artifact(f"checkpoints/{ckpt_path.parent.name}", artifact_object=str(ckpt_path))
    except Exception as exc:
        if cfg.tracking.fail_on_error:
            raise
        logger.warning("ClearML output logging failed: %s", exc)


def close_clearml_task(task, *, failed: bool = False, status_message: str | None = None) -> None:
    if task is None:
        return
    if failed and hasattr(task, "mark_failed"):
        try:
            task.mark_failed(force=True, status_message=status_message)
        except Exception as exc:
            logger.warning("ClearML mark_failed failed before close: %s", exc)
    task.close()


def _upload_if_exists(task, name: str, path: Path) -> None:
    if path.exists():
        task.upload_artifact(name, artifact_object=str(path))


def _configure_clearml_cache(cache_dir: str) -> None:
    if cache_dir:
        path = Path(cache_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CLEARML_CACHE_DIR", str(path))


def _set_bool_env(name: str, setter) -> None:
    if name in os.environ:
        setter(_parse_bool(os.environ[name]))


def _set_int_env(name: str, setter) -> None:
    if name in os.environ:
        setter(int(os.environ[name]))


def _set_str_env(name: str, setter) -> None:
    value = os.getenv(name)
    if value:
        setter(value)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
