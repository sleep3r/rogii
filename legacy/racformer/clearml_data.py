"""ClearML Dataset upload/download for RAC-Former.

Mirrors old/rogii/clearml_data.py with the same dataset coordinates so the
existing uploaded `rogii-wellbore-geology-prediction` dataset can be reused
without re-uploading.

CLI:
    python -m racformer.clearml_data download \
        --project ROGII/Wellbore \
        --name rogii-wellbore-geology-prediction \
        --version 20260519_s3 \
        --cache-dir ~/.cache/clearml/rogii

    python -m racformer.clearml_data upload \
        --data-dir MTPNet/data \
        --output-uri s3://s3-basket-cold.wb.ru/ds-experiments
"""
from __future__ import annotations

import argparse
import fnmatch
import inspect
import sys
from pathlib import Path
from typing import Any

DEFAULT_EXCLUDES = ["*.zip", "__pycache__", ".DS_Store", "*.pptx"]


def _info(*args, **kwargs) -> None:
    parts = [str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()]
    print("[clearml_data] " + " ".join(parts), file=sys.stderr)


def _excluded(path: Path, root: Path, patterns: list[str]) -> bool:
    relative = path.relative_to(root).as_posix()
    return any(
        fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(relative, pat)
        for pat in patterns
    )


def _selected(data_dir: Path, excludes: list[str]) -> list[Path]:
    return [p for p in sorted(data_dir.iterdir()) if not _excluded(p, data_dir, excludes)]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_dataset(
    data_dir: Path,
    project: str,
    name: str,
    version: str | None,
    output_uri: str | None,
    tags: list[str],
    excludes: list[str],
    max_workers: int = 8,
) -> str:
    from clearml import Dataset  # type: ignore

    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if output_uri and not output_uri.startswith("s3://"):
        _info("output_uri is not s3:// — uploads will go to the ClearML server", uri=output_uri)

    paths = _selected(data_dir, excludes)
    if not paths:
        raise FileNotFoundError(f"No files selected under {data_dir}")

    _info("creating dataset", project=project, name=name, version=version, paths=len(paths))
    dataset = Dataset.create(
        dataset_project=project,
        dataset_name=name,
        dataset_version=version,
        dataset_tags=tags or None,
        output_uri=output_uri or None,
        description="RAC-Former training data (TVT prediction).",
    )

    added = 0
    for p in paths:
        added += int(dataset.add_files(
            path=p,
            local_base_folder=str(data_dir),
            recursive=True,
            verbose=False,
            max_workers=max_workers,
        ))
    _info("files staged", count=added)

    dataset.upload(show_progress=True, verbose=False, output_url=output_uri, max_workers=max_workers)
    dataset.finalize(auto_upload=False)
    _info("dataset ready", id=dataset.id)
    return str(dataset.id)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _get_local_copy(dataset: Any, cache_dir: Path | None) -> str:
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        sig = inspect.signature(dataset.get_local_copy)
        if "local_cache_path" in sig.parameters:
            kwargs["local_cache_path"] = str(cache_dir.expanduser())
    return str(dataset.get_local_copy(**kwargs))


def get_dataset_local_path(
    project: str,
    name: str,
    version: str | None = None,
    dataset_id: str | None = None,
    alias: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a ClearML Dataset to a local directory (download if not cached)."""
    from clearml import Dataset  # type: ignore

    _info("fetching dataset", project=project, name=name, version=version,
          dataset_id=dataset_id, alias=alias)
    dataset = Dataset.get(
        dataset_id=dataset_id or None,
        dataset_project=project if not dataset_id else None,
        dataset_name=name if not dataset_id else None,
        dataset_version=version or None,
        alias=alias or None,
        only_completed=True,
    )
    local = _get_local_copy(dataset, cache_dir)
    if not local:
        raise RuntimeError("ClearML returned empty dataset local path")
    p = Path(local)
    _info("local copy ready", path=p)
    return p


# ---------------------------------------------------------------------------
# Integration helper for train.py
# ---------------------------------------------------------------------------

def resolve_data_dir(data_clearml_cfg: Any) -> Path | None:
    """If data.clearml.enabled, download the dataset and return its local path.

    Returns None if ClearML data resolution is disabled.
    Expects a RACDataClearMLConfig (or similar dataclass) with attributes
    enabled, project, name, version, dataset_id, alias, cache_dir.
    """
    if data_clearml_cfg is None or not getattr(data_clearml_cfg, "enabled", False):
        return None

    project = getattr(data_clearml_cfg, "project", "") or ""
    name = getattr(data_clearml_cfg, "name", "") or ""
    dataset_id = getattr(data_clearml_cfg, "dataset_id", None) or None
    if not dataset_id and (not project or not name):
        raise ValueError("data.clearml.project and data.clearml.name are required when enabled=true")

    cache_dir_value = getattr(data_clearml_cfg, "cache_dir", None)
    cache_dir = Path(cache_dir_value) if cache_dir_value else None

    return get_dataset_local_path(
        project=project,
        name=name,
        version=getattr(data_clearml_cfg, "version", None) or None,
        dataset_id=dataset_id,
        alias=getattr(data_clearml_cfg, "alias", None) or None,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClearML Dataset upload/download for RAC-Former.")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload local data dir as a ClearML Dataset.")
    up.add_argument("--data-dir", type=Path, default=Path("MTPNet/data"))
    up.add_argument("--project", default="ROGII/Wellbore")
    up.add_argument("--name", default="rogii-wellbore-geology-prediction")
    up.add_argument("--version", default=None)
    up.add_argument("--output-uri", default=None)
    up.add_argument("--tags", default="racformer,rogii,kaggle,data")
    up.add_argument("--exclude", action="append", default=[])
    up.add_argument("--max-workers", type=int, default=8)

    dn = sub.add_parser("download", help="Resolve a ClearML Dataset locally and print its path.")
    dn.add_argument("--project", default="ROGII/Wellbore")
    dn.add_argument("--name", default="rogii-wellbore-geology-prediction")
    dn.add_argument("--version", default=None)
    dn.add_argument("--dataset-id", default=None)
    dn.add_argument("--alias", default=None)
    dn.add_argument("--cache-dir", type=Path, default=Path("~/.cache/clearml/rogii"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "upload":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        ds_id = upload_dataset(
            data_dir=args.data_dir,
            project=args.project,
            name=args.name,
            version=args.version,
            output_uri=args.output_uri,
            tags=tags,
            excludes=DEFAULT_EXCLUDES + list(args.exclude or []),
            max_workers=args.max_workers,
        )
        print(ds_id, flush=True)
    elif args.command == "download":
        path = get_dataset_local_path(
            project=args.project,
            name=args.name,
            version=args.version,
            dataset_id=args.dataset_id,
            alias=args.alias,
            cache_dir=args.cache_dir,
        )
        print(path, flush=True)


if __name__ == "__main__":
    main()
