from __future__ import annotations

import json
from pathlib import Path


def evaluate_run(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(metrics.get("valid", metrics), indent=2), flush=True)
    return metrics
