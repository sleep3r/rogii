from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import KAGGLE_INPUT_DIR

def path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def resolve_data_dir(config: dict[str, Any]) -> Path:
    configured = path_or_none(config["data"].get("data_dir"))
    if configured is not None:
        return configured
    if KAGGLE_INPUT_DIR.exists():
        return KAGGLE_INPUT_DIR
    return Path("data")


def well_name(path: Path) -> str:
    return path.name.split("__horizontal_well.csv", 1)[0]


def limited(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    return paths[: int(limit)]


def resolve_train_dir(data_dir: Path, config: dict[str, Any]) -> Path:
    configured = path_or_none(config["data"].get("train_dir"))
    candidates = [configured, data_dir / "train"]
    for path in candidates:
        if path is not None and path.exists():
            return path
    raise FileNotFoundError(
        f"No train directory found under {data_dir}. Run `make unzip-data` first, "
        "or use configs/quick.yml for the small public sample."
    )


def resolve_test_dir(data_dir: Path, config: dict[str, Any]) -> Path:
    configured = path_or_none(config["data"].get("test_dir"))
    candidates = [configured, data_dir / "test"]
    for path in candidates:
        if path is not None and path.exists() and list(path.glob("*__horizontal_well.csv")):
            return path
    raise FileNotFoundError(
        f"No test horizontal well files found under {data_dir}. Run `make unzip-data` first, "
        "or use configs/quick.yml for the small public sample."
    )


def resolve_sample_submission(data_dir: Path, test_dir: Path, config: dict[str, Any]) -> Path | None:
    configured = path_or_none(config["data"].get("sample_submission"))
    candidates = [
        configured,
        data_dir / "sample_submission.csv",
        test_dir / "sample_submission.csv",
        data_dir / "public_test" / "sample_submission.csv",
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    matches = sorted(data_dir.rglob("sample_submission.csv"))
    return matches[0] if matches else None


def horizontal_files(directory: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(directory.glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No horizontal well files in {directory}")
    return limited(paths, limit)


def typewell_path(horizontal_path: Path) -> Path | None:
    name = well_name(horizontal_path)
    root = horizontal_path.parent.parent
    candidates = [
        horizontal_path.with_name(f"{name}__typewell.csv"),
        root / f"{name}__typewell.csv",
        root / "train" / f"{name}__typewell.csv",
        root / "test" / f"{name}__typewell.csv",
        root / "public_train" / f"{name}__typewell.csv",
        root / "public_test" / f"{name}__typewell.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None
