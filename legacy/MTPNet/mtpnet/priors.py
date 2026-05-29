from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PriorConfig


@dataclass(frozen=True)
class PriorTables:
    frame: pd.DataFrame


def _read_frame(
    path: Path,
    *,
    id_column: str,
    required: dict[str, str],
    optional: dict[str, str] | None = None,
) -> pd.DataFrame:
    optional = optional or {}
    source_columns = [id_column, *required.keys(), *optional.keys()]
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, usecols=source_columns)
        else:
            frame = pd.read_parquet(path, columns=source_columns)
    except Exception as exc:
        if optional:
            return _read_frame(
                path,
                id_column=id_column,
                required=required,
                optional={},
            )
        missing = ", ".join(required)
        raise ValueError(f"{path} must contain columns: {id_column}, {missing}") from exc
    rename = {**required, **optional}
    frame = frame.rename(columns=rename)
    keep = [id_column, *rename.values()]
    return frame[keep]


def _merge_part(
    base: pd.DataFrame | None, part: pd.DataFrame, id_column: str
) -> pd.DataFrame:
    part = part.copy()
    part[id_column] = part[id_column].astype(str)
    part = part.drop_duplicates(subset=[id_column], keep="first")
    if base is None:
        return part
    return base.merge(part, on=id_column, how="outer")


def load_prior_tables(cfg: PriorConfig) -> PriorTables | None:
    if not cfg.enabled:
        return None
    if cfg.base_path is None and cfg.b2_path is None and cfg.a_path is None:
        raise ValueError("priors.enabled=true requires at least one prior source path")

    merged: pd.DataFrame | None = None
    if cfg.base_path is not None:
        part = _read_frame(
            cfg.base_path,
            id_column=cfg.id_column,
            required={cfg.base_column: "base_tvt"},
        )
        merged = _merge_part(merged, part, cfg.id_column)
    if cfg.b2_path is not None:
        part = _read_frame(
            cfg.b2_path,
            id_column=cfg.id_column,
            required={cfg.b2_column: "b2_tvt"},
            optional={cfg.b2_danger_column: "b2_danger"},
        )
        merged = _merge_part(merged, part, cfg.id_column)
    if cfg.a_path is not None:
        part = _read_frame(
            cfg.a_path,
            id_column=cfg.id_column,
            required={
                cfg.a_p50_column: "a_p50_tvt",
                cfg.a_p10_column: "a_p10_tvt",
                cfg.a_p90_column: "a_p90_tvt",
            },
        )
        merged = _merge_part(merged, part, cfg.id_column)

    if merged is None:
        return None
    merged = merged.set_index(cfg.id_column).sort_index()
    return PriorTables(frame=merged)
