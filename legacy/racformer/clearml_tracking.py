"""ClearML experiment tracking for RAC-Former.

Slim version of old/rogii/clearml_tracking.py.  No-ops gracefully when
ClearML is not installed or `tracking.clearml.enabled = false`.

Usage:
    tracker = ClearMLTracker.start(cfg, run_id="20260525-1") or NullTracker()
    tracker.connect_config(config_to_dict(cfg))
    for epoch in ...:
        tracker.report_scalars({"fold_0/val_rmse": val_rmse, ...}, iteration=epoch)
    tracker.upload_artifacts(output_dir)
    tracker.close()
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _info(*args, **kwargs) -> None:
    parts = [str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()]
    print("[clearml] " + " ".join(parts), file=sys.stderr)


# ---------------------------------------------------------------------------
# Null tracker — same interface, no-ops everywhere
# ---------------------------------------------------------------------------

class NullTracker:
    enabled: bool = False

    def connect_config(self, config: dict[str, Any]) -> None: ...
    def report_scalars(self, metrics: dict[str, float], iteration: int = 0) -> None: ...
    def report_scalar(self, title: str, series: str, value: float, iteration: int = 0) -> None: ...
    def report_text(self, name: str, value: str) -> None: ...
    def upload_artifact(self, name: str, path: Path) -> None: ...
    def upload_artifacts(self, output_dir: Path) -> None: ...
    def add_tags(self, tags: list[str]) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Real tracker
# ---------------------------------------------------------------------------

class ClearMLTracker:
    """Thin wrapper over clearml.Task with safety guards."""

    enabled: bool = True

    def __init__(self, task: Any, fail_on_error: bool, log_artifacts: bool, log_model: bool):
        self.task = task
        self.fail_on_error = fail_on_error
        self.log_artifacts = log_artifacts
        self.log_model = log_model

    # ---------------------------------------------------------------------
    @classmethod
    def start(
        cls,
        project: str,
        task_name: str,
        output_uri: str | None = None,
        tags: list[str] | None = None,
        fail_on_error: bool = False,
        log_artifacts: bool = True,
        log_model: bool = True,
    ) -> ClearMLTracker | NullTracker:
        """Return a real tracker if ClearML is importable and Task.init succeeds.

        Falls back to NullTracker on any failure (unless fail_on_error=True).
        """
        try:
            from clearml import Task  # type: ignore
        except ImportError:
            _info("clearml not installed — tracking disabled")
            if fail_on_error:
                raise
            return NullTracker()

        try:
            task = Task.init(
                project_name=project,
                task_name=task_name,
                output_uri=output_uri or None,
                reuse_last_task_id=False,
            )
            if tags:
                task.add_tags(tags)
            _info("task started", project=project, task=task_name, output_uri=output_uri)
            return cls(task, fail_on_error, log_artifacts, log_model)
        except Exception as exc:
            _info("Task.init failed", error=exc)
            if fail_on_error:
                raise
            return NullTracker()

    # ---------------------------------------------------------------------
    def _guard(self, action: str, func) -> None:
        try:
            func()
        except Exception as exc:
            _info(f"{action} failed", error=exc)
            if self.fail_on_error:
                raise

    # ---------------------------------------------------------------------
    def connect_config(self, config: dict[str, Any]) -> None:
        def _do():
            try:
                self.task.connect_configuration(config, name="resolved_config")
            except Exception:
                self.task.connect(config)
        self._guard("connect_config", _do)

    def report_scalar(self, title: str, series: str, value: float, iteration: int = 0) -> None:
        def _do():
            self.task.get_logger().report_scalar(
                title=title, series=series, value=float(value), iteration=int(iteration),
            )
        self._guard("report_scalar", _do)

    def report_scalars(self, metrics: dict[str, float], iteration: int = 0) -> None:
        """Flat dict where keys can be 'group/metric' (slash → title/series)."""
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if "/" in key:
                title, series = key.split("/", 1)
            else:
                title, series = "metrics", key
            self.report_scalar(title, series, v, iteration)

    def report_text(self, name: str, value: str) -> None:
        def _do():
            self.task.get_logger().report_text(value, level=20)  # INFO
        self._guard(f"report_text[{name}]", _do)

    def upload_artifact(self, name: str, path: Path) -> None:
        if not self.log_artifacts:
            return
        def _do():
            self.task.upload_artifact(name=name, artifact_object=str(path))
        self._guard(f"upload_artifact[{name}]", _do)

    def upload_artifacts(self, output_dir: Path) -> None:
        """Walk output_dir and upload every file (skips model weights if log_model=False)."""
        if not self.log_artifacts:
            return
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            return

        def _do():
            for p in sorted(output_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(output_dir).as_posix()
                if rel.endswith(".pt") and not self.log_model:
                    continue
                self.task.upload_artifact(name=rel, artifact_object=str(p))
        self._guard("upload_artifacts", _do)

    def add_tags(self, tags: list[str]) -> None:
        if not tags:
            return
        def _do():
            self.task.add_tags(tags)
        self._guard("add_tags", _do)

    def close(self) -> None:
        def _do():
            self.task.close()
        self._guard("close", _do)
