from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .runlog import RunLogger

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _set_if_present(config: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        config[key] = value


def _set_bool_if_present(config: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        config[key] = parse_bool(value)


def apply_clearml_overrides(
    config: dict[str, Any],
    args: Any,
    run_id: str,
    config_path: Path,
) -> dict[str, Any]:
    """Apply env/CLI ClearML overrides without making ClearML mandatory."""
    tracking_cfg = config.setdefault("tracking", {})
    clearml_cfg = tracking_cfg.setdefault("clearml", {})
    if clearml_cfg.get("project") in (None, ""):
        clearml_cfg["project"] = config.get("project_name") or "ROGII/Wellbore"
    if clearml_cfg.get("output_uri") in (None, ""):
        clearml_cfg["output_uri"] = config.get("output_uri")

    env_map = {
        "enabled": os.getenv("ROGII_CLEARML_ENABLED"),
        "project": os.getenv("ROGII_CLEARML_PROJECT"),
        "task_name": os.getenv("ROGII_CLEARML_TASK_NAME"),
        "output_uri": os.getenv("ROGII_CLEARML_OUTPUT_URI"),
        "tags": os.getenv("ROGII_CLEARML_TAGS"),
        "log_artifacts": os.getenv("ROGII_CLEARML_LOG_ARTIFACTS"),
        "log_model": os.getenv("ROGII_CLEARML_LOG_MODEL"),
        "fail_on_error": os.getenv("ROGII_CLEARML_FAIL_ON_ERROR"),
    }
    _set_bool_if_present(clearml_cfg, "enabled", env_map["enabled"])
    _set_if_present(clearml_cfg, "project", env_map["project"])
    _set_if_present(clearml_cfg, "task_name", env_map["task_name"])
    _set_if_present(clearml_cfg, "output_uri", env_map["output_uri"])
    if env_map["tags"] not in (None, ""):
        clearml_cfg["tags"] = parse_tags(env_map["tags"])
    _set_bool_if_present(clearml_cfg, "log_artifacts", env_map["log_artifacts"])
    _set_bool_if_present(clearml_cfg, "log_model", env_map["log_model"])
    _set_bool_if_present(clearml_cfg, "fail_on_error", env_map["fail_on_error"])

    _set_bool_if_present(clearml_cfg, "enabled", getattr(args, "clearml_enabled", None))
    _set_if_present(clearml_cfg, "project", getattr(args, "clearml_project", None))
    _set_if_present(clearml_cfg, "task_name", getattr(args, "clearml_task_name", None))
    _set_if_present(
        clearml_cfg, "output_uri", getattr(args, "clearml_output_uri", None)
    )
    cli_tags = getattr(args, "clearml_tags", None)
    if cli_tags not in (None, ""):
        clearml_cfg["tags"] = parse_tags(cli_tags)
    _set_bool_if_present(
        clearml_cfg, "log_artifacts", getattr(args, "clearml_log_artifacts", None)
    )
    _set_bool_if_present(
        clearml_cfg, "log_model", getattr(args, "clearml_log_model", None)
    )
    _set_bool_if_present(
        clearml_cfg, "fail_on_error", getattr(args, "clearml_fail_on_error", None)
    )

    if clearml_cfg.get("task_name") in (None, ""):
        clearml_cfg["task_name"] = f"rogii-{config_path.stem}-{run_id}"
    return clearml_cfg


class ClearMLTracker:
    def __init__(
        self,
        task: Any | None,
        logger: RunLogger,
        enabled: bool,
        fail_on_error: bool,
        log_artifacts_enabled: bool,
        log_model: bool,
    ) -> None:
        self.task = task
        self.logger = logger
        self.enabled = enabled and task is not None
        self.fail_on_error = fail_on_error
        self.log_artifacts_enabled = log_artifacts_enabled
        self.log_model = log_model

    @classmethod
    def start(
        cls,
        config: dict[str, Any],
        logger: RunLogger,
        run_id: str,
        config_path: Path,
    ) -> "ClearMLTracker":
        clearml_cfg = config.get("tracking", {}).get("clearml", {})
        enabled = parse_bool(clearml_cfg.get("enabled", False))
        fail_on_error = parse_bool(clearml_cfg.get("fail_on_error", False))
        log_artifacts_enabled = parse_bool(clearml_cfg.get("log_artifacts", True))
        log_model = parse_bool(clearml_cfg.get("log_model", False))
        if not enabled:
            return cls(
                None, logger, False, fail_on_error, log_artifacts_enabled, log_model
            )

        try:
            from clearml import Task  # type: ignore

            project = str(
                clearml_cfg.get("project") or config.get("project_name") or "ROGII"
            )
            task_name = str(clearml_cfg.get("task_name") or f"rogii-{run_id}")
            output_uri = (
                clearml_cfg.get("output_uri") or config.get("output_uri") or None
            )
            task = Task.init(
                project_name=project,
                task_name=task_name,
                output_uri=output_uri,
                reuse_last_task_id=False,
            )
            tags = parse_tags(clearml_cfg.get("tags")) + [config_path.stem, run_id]
            if tags:
                task.add_tags(tags)
            tracker = cls(
                task,
                logger,
                True,
                fail_on_error,
                log_artifacts_enabled,
                log_model,
            )
            tracker.connect_config(config)
            logger.info(
                "ClearML task started",
                project=project,
                task_name=task_name,
                output_uri=output_uri,
            )
            return tracker
        except Exception as exc:
            if fail_on_error:
                raise
            logger.warn("ClearML disabled after startup failure", error=exc)
            return cls(
                None, logger, False, fail_on_error, log_artifacts_enabled, log_model
            )

    def _guard(self, action: str, func: Any) -> None:
        if not self.enabled or self.task is None:
            return
        try:
            func()
        except Exception as exc:
            if self.fail_on_error:
                raise
            self.logger.warn("ClearML action failed", action=action, error=exc)

    def connect_config(self, config: dict[str, Any]) -> None:
        def action() -> None:
            assert self.task is not None
            try:
                self.task.connect_configuration(config, name="resolved_config")
            except Exception:
                self.task.connect(config)

        self._guard("connect_config", action)

    def report_metrics(self, metrics: dict[str, Any]) -> None:
        def report_scalar(path: tuple[str, ...], value: float, iteration: int) -> None:
            assert self.task is not None
            title = path[0] if path else "metrics"
            series = ".".join(path[1:]) if len(path) > 1 else "value"
            self.task.get_logger().report_scalar(
                title=title,
                series=series,
                value=float(value),
                iteration=int(iteration),
            )

        def walk(path: tuple[str, ...], value: Any) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, int | float):
                report_scalar(path, float(value), 0)
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    walk((*path, str(key)), child)
                return
            if isinstance(value, list):
                if all(isinstance(item, int | float) for item in value):
                    for index, item in enumerate(value):
                        report_scalar(path, float(item), index)
                return

        self._guard("report_metrics", lambda: walk(("metrics",), metrics))

    def upload_artifacts(
        self,
        output_dir: Path,
        submission_path: Path,
        registry_path: Path,
    ) -> None:
        if not self.log_artifacts_enabled:
            return

        artifacts = {
            "submission.csv": submission_path,
            "runs.csv": registry_path,
        }
        if output_dir.is_dir():
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(output_dir).as_posix()
                if relative == "model.pkl" and not self.log_model:
                    continue
                artifacts[f"output/{relative}"] = path

        def action() -> None:
            assert self.task is not None
            for name, path in artifacts.items():
                if path.is_file():
                    self.task.upload_artifact(name=name, artifact_object=str(path))

        self._guard("upload_artifacts", action)

    def close(self) -> None:
        self._guard(
            "close", lambda: self.task.close() if self.task is not None else None
        )
