"""Structured progress logging used across training, OOF, stitching, and tracking.

All events are emitted as one-line JSON to stdout so that downstream tooling can
parse the stream without ambiguity. Every record carries a wall-clock timestamp
and an optional run-relative ``elapsed_seconds`` field; ``stage_seconds`` shows
how long a measured sub-stage took. Records are printed with ``flush=True`` so
they appear in real time on Make-driven runs.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ProgressLogger:
    """Tiny JSON-line emitter with optional run-wide stopwatch."""

    run: str = ""
    started_at: float = field(default_factory=time.monotonic)

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": event,
            "ts": time.time(),
            "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
        }
        if self.run:
            record["run"] = self.run
        for key, value in fields.items():
            record[key] = _coerce_json(value)
        print(json.dumps(record), flush=True)
        return record

    @contextmanager
    def stage(self, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Emit ``{event}_start`` and ``{event}_done`` around a block.

        The yielded mapping is mutated in place: anything added to it before the
        block exits is included in the ``done`` event. The ``done`` event also
        carries ``stage_seconds`` with the wall-clock duration.
        """

        self.log(f"{event}_start", **fields)
        extra: dict[str, Any] = {}
        start = time.monotonic()
        try:
            yield extra
        finally:
            done_fields = {**fields, **extra}
            self.log(
                f"{event}_done",
                stage_seconds=round(time.monotonic() - start, 3),
                **done_fields,
            )


def _coerce_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _coerce_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_json(item) for item in value]
    return str(value)


def format_eta(seconds: float) -> str:
    if not (seconds == seconds) or seconds <= 0.0 or seconds == float("inf"):
        return "n/a"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"
