from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DataConfig


@dataclass(frozen=True)
class WellPaths:
    well_id: str
    horizontal_path: Path
    typewell_path: Path


def well_id_from_horizontal(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not a horizontal well file: {path}")
    return path.name[: -len(suffix)]


def resolve_train_dir(config: DataConfig) -> Path:
    candidates = []
    if config.train_dir is not None:
        candidates.append(config.train_dir)
    candidates.append(config.data_dir / "train")
    for path in candidates:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No train directory found. Checked: {checked}")


def discover_wells(config: DataConfig) -> list[WellPaths]:
    train_dir = resolve_train_dir(config)
    horizontal_paths = sorted(
        train_dir.glob("*__horizontal_well.csv"), key=well_id_from_horizontal
    )
    if not horizontal_paths:
        raise FileNotFoundError(f"No horizontal well files found in {train_dir}")
    if config.k_wells > 0:
        horizontal_paths = horizontal_paths[: config.k_wells]

    wells: list[WellPaths] = []
    for horizontal_path in horizontal_paths:
        well_id = well_id_from_horizontal(horizontal_path)
        typewell_path = horizontal_path.with_name(f"{well_id}__typewell.csv")
        if not typewell_path.exists():
            raise FileNotFoundError(f"Missing typewell for {well_id}: {typewell_path}")
        wells.append(
            WellPaths(
                well_id=well_id,
                horizontal_path=horizontal_path,
                typewell_path=typewell_path,
            )
        )
    return wells


def load_well(well: WellPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(well.horizontal_path)
    typewell = pd.read_csv(well.typewell_path)
    required_horizontal = {"TVT", "TVT_input", "GR"}
    required_typewell = {"TVT", "GR"}
    missing_horizontal = required_horizontal.difference(horizontal.columns)
    missing_typewell = required_typewell.difference(typewell.columns)
    if missing_horizontal:
        raise ValueError(
            f"{well.horizontal_path} missing columns: {sorted(missing_horizontal)}"
        )
    if missing_typewell:
        raise ValueError(
            f"{well.typewell_path} missing columns: {sorted(missing_typewell)}"
        )
    return horizontal, typewell


def copy_data_tree(source: str | Path, target: str | Path) -> None:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise FileNotFoundError(f"Data source does not exist: {source_path}")
    target_path.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "public_train", "public_test"):
        src = source_path / name
        if src.exists():
            dst = target_path / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    for filename in ("sample_submission.csv",):
        src = source_path / filename
        if src.exists():
            shutil.copy2(src, target_path / filename)
