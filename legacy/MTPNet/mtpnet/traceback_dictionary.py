from __future__ import annotations

import numpy as np
import pandas as pd

from .correlation_panel import _regular_typewell_grid
from .traceback_events import TracebackConfig, extract_traceback_events


DICT_COLUMNS = [
    "source",
    "source_well_id",
    "source_step",
    "source_TVT",
    "event_type",
    "prominence",
    "finite_frac",
    "patch_radius",
    "patch_gr",
    "patch_dgr",
]


def _empty_dictionary() -> pd.DataFrame:
    return pd.DataFrame(columns=DICT_COLUMNS)


def _dictionary_from_events(
    events: pd.DataFrame,
    *,
    source: str,
    tvt_column: str,
) -> pd.DataFrame:
    if events.empty or tvt_column not in events:
        return _empty_dictionary()
    rows = []
    for _, row in events.iterrows():
        tvt = pd.to_numeric(pd.Series([row[tvt_column]]), errors="coerce").iloc[0]
        if not np.isfinite(tvt):
            continue
        rows.append(
            {
                "source": source,
                "source_well_id": str(row["well_id"]),
                "source_step": int(row["step"]),
                "source_TVT": float(tvt),
                "event_type": str(row["event_type"]),
                "prominence": float(row["prominence"]),
                "finite_frac": float(row["finite_frac"]),
                "patch_radius": int(row["patch_radius"]),
                "patch_gr": np.asarray(row["patch_gr"], dtype=np.float32),
                "patch_dgr": np.asarray(row["patch_dgr"], dtype=np.float32),
            }
        )
    return pd.DataFrame(rows, columns=DICT_COLUMNS)


def build_same_well_dictionary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "known_mask" not in events.columns:
        return _empty_dictionary()
    known = events[events["known_mask"].astype(bool)].copy() if not events.empty else events
    return _dictionary_from_events(
        known, source="same_well_known", tvt_column="TVT_input"
    )


def build_fold_safe_train_dictionary(
    events: pd.DataFrame,
    *,
    train_wells: list[str] | tuple[str, ...] | set[str],
) -> pd.DataFrame:
    if events.empty or "well_id" not in events.columns:
        return _empty_dictionary()
    train_set = {str(well_id) for well_id in train_wells}
    train_events = events[events["well_id"].astype(str).isin(train_set)].copy()
    return _dictionary_from_events(
        train_events, source="fold_train", tvt_column="true_TVT"
    )


def build_typewell_dictionary(
    typewell: pd.DataFrame,
    *,
    vertical_step_ft: float,
    patch_radii: tuple[int, ...],
    min_prominence_z: float,
) -> pd.DataFrame:
    grid, gr = _regular_typewell_grid(typewell, vertical_step_ft)
    if grid.size == 0:
        return _empty_dictionary()
    comp = pd.DataFrame(
        {
            "well_id": "typewell",
            "step": np.arange(grid.size, dtype=int),
            "row_start": np.arange(grid.size, dtype=int),
            "row_end": np.arange(grid.size, dtype=int),
            "MD": np.arange(grid.size, dtype=float),
            "X": np.zeros(grid.size),
            "Y": np.zeros(grid.size),
            "Z": np.zeros(grid.size),
            "GR_filled": gr.astype(np.float32),
            "GR_finite_frac": np.ones(grid.size),
            "TVT_input": grid.astype(np.float32),
            "known_mask": np.ones(grid.size, dtype=bool),
            "true_TVT": grid.astype(np.float32),
        }
    )
    cfg = TracebackConfig(
        rows_per_step=1,
        patch_radii=patch_radii,
        min_prominence_z=min_prominence_z,
        vertical_step_ft=vertical_step_ft,
    )
    events = extract_traceback_events(comp, cfg)
    out = _dictionary_from_events(events, source="typewell", tvt_column="true_TVT")
    if not out.empty:
        out["source_well_id"] = "typewell"
    return out
