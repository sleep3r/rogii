"""ClearML Dataset upload/download CLI."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BPHWT ClearML dataset utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    upload = sub.add_parser("upload", help="Create/upload a ClearML Dataset from a local data directory")
    upload.add_argument("--data-dir", required=True)
    upload.add_argument("--project", required=True)
    upload.add_argument("--name", required=True)
    upload.add_argument("--version", required=True)
    upload.add_argument("--output-uri", default="")
    upload.add_argument("--max-workers", type=int, default=8)

    download = sub.add_parser("download", help="Resolve/download a ClearML Dataset local copy")
    download.add_argument("--project", required=True)
    download.add_argument("--name", required=True)
    download.add_argument("--version", required=True)
    download.add_argument("--cache-dir", default="")
    download.add_argument("--max-workers", type=int, default=8)

    check = sub.add_parser("check", help="Check that ClearML imports and config can be used")
    check.add_argument("--project", default="")
    check.add_argument("--name", default="")
    check.add_argument("--version", default="")
    return parser


def upload_dataset(
    *,
    data_dir: Path,
    project: str,
    name: str,
    version: str,
    output_uri: str = "",
    max_workers: int = 8,
) -> str:
    from clearml import Dataset

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    dataset = Dataset.create(
        dataset_project=project,
        dataset_name=name,
        dataset_version=version,
        output_uri=output_uri or None,
    )
    added = dataset.add_files(path=data_dir, recursive=True, max_workers=max_workers)
    logger.info("Added %s files from %s", added, data_dir)
    dataset.upload(output_url=output_uri or None, max_workers=max_workers)
    dataset.finalize()
    logger.info("Uploaded ClearML dataset id=%s project=%s name=%s version=%s", dataset.id, project, name, version)
    return dataset.id


def download_dataset(
    *,
    project: str,
    name: str,
    version: str,
    cache_dir: str = "",
    max_workers: int = 8,
) -> str:
    if cache_dir:
        path = Path(cache_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CLEARML_CACHE_DIR", str(path))

    from clearml import Dataset

    dataset = Dataset.get(
        dataset_project=project,
        dataset_name=name,
        dataset_version=version,
        only_completed=True,
    )
    local_path = dataset.get_local_copy(max_workers=max_workers)
    logger.info("ClearML dataset local path: %s", local_path)
    print(local_path)
    return str(local_path)


def check_clearml(project: str = "", name: str = "", version: str = "") -> None:
    import clearml
    from clearml import Dataset, Task

    logger.info("clearml version: %s", clearml.__version__)
    logger.info("Task.init available: %s", hasattr(Task, "init"))
    logger.info("Dataset.get available: %s", hasattr(Dataset, "get"))
    if project and name and version:
        dataset = Dataset.get(
            dataset_project=project,
            dataset_name=name,
            dataset_version=version,
            only_completed=True,
        )
        logger.info("Dataset found: id=%s", dataset.id)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "upload":
        upload_dataset(
            data_dir=Path(args.data_dir),
            project=args.project,
            name=args.name,
            version=args.version,
            output_uri=args.output_uri,
            max_workers=args.max_workers,
        )
    elif args.command == "download":
        download_dataset(
            project=args.project,
            name=args.name,
            version=args.version,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
        )
    elif args.command == "check":
        check_clearml(project=args.project, name=args.name, version=args.version)


if __name__ == "__main__":
    main()
