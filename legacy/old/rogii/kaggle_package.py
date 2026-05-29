from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

DEFAULT_INCLUDE = [
    Path("rogii"),
    Path("configs"),
    Path("pyproject.toml"),
    Path("README.md"),
]
EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "data", "artifacts", ".kaggle_mining"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def should_include(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return True


def iter_package_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and should_include(item)
            )
        elif path.is_file() and should_include(path):
            files.append(path)
    return sorted(files)


def build_package(output: Path, paths: list[Path] | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = paths or DEFAULT_INCLUDE
    files = iter_package_files(source_paths)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.as_posix())
    print(f"Wrote {output} with {len(files)} files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package ROGII source files for Kaggle notebooks."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/rogii_source.zip")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_package(args.output)


if __name__ == "__main__":
    main()
