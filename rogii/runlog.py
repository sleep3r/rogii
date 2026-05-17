from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
from typing import Any


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:05.2f}s"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}"
    return f"{minutes:02d}:{whole_seconds:02d}"


def format_log_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.5f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


class RunLogger:
    def __init__(self) -> None:
        self.started_at = perf_counter()

    def elapsed(self) -> str:
        return format_duration(perf_counter() - self.started_at)

    def log(self, status: str, message: str, **fields: Any) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        suffix = ""
        if fields:
            suffix = " | " + " ".join(
                f"{key}={format_log_value(value)}"
                for key, value in fields.items()
                if value is not None
            )
        print(
            f"{timestamp} | +{self.elapsed()} | {status:<6} | {message}{suffix}",
            flush=True,
        )

    def info(self, message: str, **fields: Any) -> None:
        self.log("INFO", message, **fields)

    def metric(self, message: str, **fields: Any) -> None:
        self.log("METRIC", message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self.log("WARN", message, **fields)

    @contextmanager
    def step(self, message: str, **fields: Any):
        step_started_at = perf_counter()
        self.log("START", message, **fields)
        try:
            yield
        except Exception as exc:
            self.log(
                "FAIL",
                message,
                duration=format_duration(perf_counter() - step_started_at),
                error=exc,
            )
            raise
        self.log(
            "OK", message, duration=format_duration(perf_counter() - step_started_at)
        )
