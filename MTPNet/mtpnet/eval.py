from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .train import write_geometry_report


def evaluate_run(run_dir: str | Path) -> dict[str, object]:
    run_path = Path(run_dir)
    path = run_path / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics.setdefault("run_name", run_path.name)
    predictions_path = run_path / "window_predictions.parquet"
    parquet_rows = None
    if predictions_path.exists():
        parquet_rows = len(pd.read_parquet(predictions_path))
    write_geometry_report(metrics, run_path, parquet_rows=parquet_rows)
    print(json.dumps(metrics.get("valid", metrics), indent=2), flush=True)
    return metrics
