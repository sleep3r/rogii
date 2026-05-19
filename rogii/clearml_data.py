from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Any

from .runlog import RunLogger

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}
DEFAULT_EXCLUDES = ["*.zip", "__pycache__", ".DS_Store"]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def excluded(path: Path, root: Path, patterns: list[str]) -> bool:
    relative = path.relative_to(root).as_posix()
    return any(
        fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(relative, pattern)
        for pattern in patterns
    )


def selected_top_level_paths(data_dir: Path, excludes: list[str]) -> list[Path]:
    paths: list[Path] = []
    for child in sorted(data_dir.iterdir()):
        if excluded(child, data_dir, excludes):
            continue
        if child.is_dir() or child.is_file():
            paths.append(child)
    return paths


def upload_dataset(
    data_dir: Path,
    project: str,
    name: str,
    version: str | None,
    output_uri: str | None,
    tags: list[str],
    excludes: list[str],
    max_workers: int | None,
    require_output_uri: bool,
    s3_only: bool,
    logger: RunLogger,
) -> str:
    from clearml import Dataset  # type: ignore

    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    output_uri = output_uri or None
    if require_output_uri and output_uri is None:
        raise ValueError("ClearML dataset upload requires --output-uri.")
    if s3_only and (output_uri is None or not output_uri.startswith("s3://")):
        raise ValueError("ClearML dataset upload requires an s3:// --output-uri.")

    paths = selected_top_level_paths(data_dir, excludes)
    if not paths:
        raise FileNotFoundError(f"No data files selected under {data_dir}")

    logger.info(
        "Creating ClearML dataset",
        project=project,
        name=name,
        version=version,
        paths=len(paths),
        output_uri=output_uri,
    )
    dataset = Dataset.create(
        dataset_project=project,
        dataset_name=name,
        dataset_version=version,
        dataset_tags=tags or None,
        output_uri=output_uri or None,
        description=(
            "ROGII Kaggle competition data. Uploaded from the local data/ "
            "directory; zip archives are excluded by default."
        ),
    )

    added = 0
    for path in paths:
        with logger.step("Add files to ClearML dataset", path=path.name):
            added += int(
                dataset.add_files(
                    path=path,
                    local_base_folder=str(data_dir),
                    recursive=True,
                    verbose=False,
                    max_workers=max_workers,
                )
            )
    logger.info("ClearML dataset files selected", files=added)
    with logger.step("Upload ClearML dataset", files=added):
        dataset.upload(
            show_progress=True,
            verbose=False,
            output_url=output_uri,
            max_workers=max_workers,
        )
    with logger.step("Finalize ClearML dataset"):
        dataset.finalize(auto_upload=False)
    dataset_id = str(dataset.id)
    logger.info("ClearML dataset ready", dataset_id=dataset_id)
    return dataset_id


def get_dataset_local_path(
    project: str,
    name: str,
    version: str | None = None,
    dataset_id: str | None = None,
    alias: str | None = None,
    cache_dir: Path | None = None,
    logger: RunLogger | None = None,
) -> Path:
    from clearml import Dataset  # type: ignore

    if logger is not None:
        logger.info(
            "Fetching ClearML dataset",
            project=project,
            name=name,
            version=version,
            dataset_id=dataset_id,
            alias=alias,
        )
    dataset = Dataset.get(
        dataset_id=dataset_id or None,
        dataset_project=project if not dataset_id else None,
        dataset_name=name if not dataset_id else None,
        dataset_version=version or None,
        alias=alias or None,
        only_completed=True,
    )
    local_path = dataset.get_local_copy(
        local_cache_path=str(cache_dir.expanduser()) if cache_dir is not None else None
    )
    if not local_path:
        raise RuntimeError("ClearML returned an empty dataset local path.")
    path = Path(local_path)
    if logger is not None:
        logger.info("ClearML dataset local path ready", path=path)
    return path


def apply_data_clearml_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    data_cfg = config.setdefault("data", {})
    clearml_cfg = data_cfg.setdefault("clearml", {})
    if clearml_cfg.get("project") in (None, ""):
        clearml_cfg["project"] = config.get("project_name") or "ROGII/Wellbore"

    env_map = {
        "enabled": os.getenv("ROGII_DATA_CLEARML_ENABLED"),
        "project": os.getenv("ROGII_DATA_CLEARML_PROJECT"),
        "name": os.getenv("ROGII_DATA_CLEARML_NAME"),
        "version": os.getenv("ROGII_DATA_CLEARML_VERSION"),
        "dataset_id": os.getenv("ROGII_DATA_CLEARML_ID"),
        "alias": os.getenv("ROGII_DATA_CLEARML_ALIAS"),
        "cache_dir": os.getenv("ROGII_DATA_CLEARML_CACHE_DIR"),
    }
    for key, value in env_map.items():
        if value in (None, ""):
            continue
        clearml_cfg[key] = parse_bool(value) if key == "enabled" else value

    for key in [
        "enabled",
        "project",
        "name",
        "version",
        "dataset_id",
        "alias",
        "cache_dir",
    ]:
        value = getattr(args, f"data_clearml_{key}", None)
        if value in (None, ""):
            continue
        clearml_cfg[key] = parse_bool(value) if key == "enabled" else value
    return clearml_cfg


def prepare_clearml_data_if_needed(
    config: dict[str, Any], logger: RunLogger | None = None
) -> Path | None:
    clearml_cfg = config.get("data", {}).get("clearml") or {}
    if not parse_bool(clearml_cfg.get("enabled", False)):
        return None

    project = str(clearml_cfg.get("project") or "")
    name = str(clearml_cfg.get("name") or "")
    dataset_id = clearml_cfg.get("dataset_id") or None
    if not dataset_id and (not project or not name):
        raise ValueError("data.clearml.project and data.clearml.name are required.")

    cache_dir_value = clearml_cfg.get("cache_dir")
    cache_dir = Path(cache_dir_value) if cache_dir_value not in (None, "") else None
    data_dir = get_dataset_local_path(
        project=project,
        name=name,
        version=clearml_cfg.get("version") or None,
        dataset_id=dataset_id,
        alias=clearml_cfg.get("alias") or None,
        cache_dir=cache_dir,
        logger=logger,
    )
    config["data"]["data_dir"] = str(data_dir)
    config["data"]["train_dir"] = None
    config["data"]["test_dir"] = None
    config["data"]["sample_submission"] = None
    return data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload/download ROGII data via ClearML Dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser(
        "upload", help="Upload local data/ as a ClearML Dataset."
    )
    upload.add_argument("--data-dir", type=Path, default=Path("data"))
    upload.add_argument("--project", default="ROGII/Wellbore")
    upload.add_argument("--name", default="rogii-wellbore-geology-prediction")
    upload.add_argument("--version", default=None)
    upload.add_argument("--output-uri", default=None)
    upload.add_argument("--require-output-uri", action="store_true")
    upload.add_argument("--s3-only", action="store_true")
    upload.add_argument("--tags", default="rogii,kaggle,data")
    upload.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob to exclude. Defaults always include zip archives.",
    )
    upload.add_argument("--max-workers", type=int, default=8)

    download = subparsers.add_parser(
        "download", help="Resolve a ClearML Dataset locally."
    )
    download.add_argument("--project", default="ROGII/Wellbore")
    download.add_argument("--name", default="rogii-wellbore-geology-prediction")
    download.add_argument("--version", default=None)
    download.add_argument("--dataset-id", default=None)
    download.add_argument("--alias", default=None)
    download.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = RunLogger()
    if args.command == "upload":
        dataset_id = upload_dataset(
            data_dir=args.data_dir,
            project=args.project,
            name=args.name,
            version=args.version,
            output_uri=args.output_uri,
            tags=parse_tags(args.tags),
            excludes=DEFAULT_EXCLUDES + list(args.exclude or []),
            max_workers=args.max_workers,
            require_output_uri=args.require_output_uri,
            s3_only=args.s3_only,
            logger=logger,
        )
        print(dataset_id, flush=True)
    elif args.command == "download":
        path = get_dataset_local_path(
            project=args.project,
            name=args.name,
            version=args.version,
            dataset_id=args.dataset_id,
            alias=args.alias,
            cache_dir=args.cache_dir,
            logger=logger,
        )
        print(path, flush=True)


if __name__ == "__main__":
    main()
