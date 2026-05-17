from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import build_well_features
from .io import well_name
from .modeling import apply_postprocess
from .runlog import RunLogger
from .spatial import KaggleTopContext


def predict_test(
    model: HistGradientBoostingRegressor,
    test_paths: list[Path],
    sample_submission_path: Path | None,
    config: dict[str, Any],
    feature_names: list[str],
    top_context: KaggleTopContext | None = None,
    logger: RunLogger | None = None,
) -> pd.DataFrame:
    predictions_by_well: dict[str, np.ndarray] = {}

    for i, path in enumerate(test_paths, start=1):
        wf = build_well_features(path, config, train=False, top_context=top_context)
        test_features = wf.features.reindex(columns=feature_names).astype("float32")
        residual_pred = model.predict(test_features)
        predictions_by_well[wf.well] = apply_postprocess(
            wf.flat_prediction, residual_pred, config
        )
        if i % 50 == 0 or i == len(test_paths):
            if logger is not None:
                logger.info("Predicted test wells", current=i, total=len(test_paths))

    if sample_submission_path is not None:
        sample = pd.read_csv(sample_submission_path)
        rows: list[tuple[str, float]] = []
        missing: set[str] = set()
        for row_id in sample["id"].astype(str):
            well, row_index_text = row_id.rsplit("_", 1)
            row_index = int(row_index_text)
            pred = predictions_by_well.get(well)
            if pred is None:
                missing.add(well)
                continue
            rows.append((row_id, float(pred[row_index])))
        if missing:
            raise FileNotFoundError(
                f"Missing test predictions for wells: {', '.join(sorted(missing))}"
            )
        return pd.DataFrame(rows, columns=["id", "tvt"])

    rows = []
    for path in test_paths:
        well = well_name(path)
        df = pd.read_csv(path, usecols=["TVT_input"])
        target_mask = pd.to_numeric(df["TVT_input"], errors="coerce").isna().to_numpy()
        for row_index in np.flatnonzero(target_mask):
            rows.append(
                (f"{well}_{row_index}", float(predictions_by_well[well][row_index]))
            )
    return pd.DataFrame(rows, columns=["id", "tvt"])


def save_outputs(
    model: HistGradientBoostingRegressor,
    feature_names: list[str],
    config: dict[str, Any],
    metrics: dict[str, Any],
    config_path: Path,
) -> None:
    output_dir = Path(config["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    with (output_dir / "features.json").open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    if config_path.exists():
        shutil.copyfile(config_path, output_dir / "config.yml")
