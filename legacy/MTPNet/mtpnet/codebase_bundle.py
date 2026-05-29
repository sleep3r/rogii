from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "data",
    "res" + "ources",
    "superpowers",
}
DEFAULT_EXCLUDE_FILES = {
    "research" + "_101.md",
}
DEFAULT_SUFFIXES = {
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
DEFAULT_FILENAMES = {"Makefile", ".gitignore"}


def _is_codebase_file(path: Path) -> bool:
    return path.name in DEFAULT_FILENAMES or path.suffix.lower() in DEFAULT_SUFFIXES


def collect_codebase_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    files: list[Path] = []
    for path in root_path.rglob("*"):
        rel = path.relative_to(root_path)
        if path.is_dir():
            continue
        if any(part in DEFAULT_EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.as_posix() in DEFAULT_EXCLUDE_FILES:
            continue
        if not _is_codebase_file(path):
            continue
        files.append(rel)
    return sorted(files, key=lambda item: item.as_posix())


def _language(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
        ".toml": "toml",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix, "")


def _fence_for(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return fence


def build_bundle(root: str | Path) -> str:
    root_path = Path(root)
    files = collect_codebase_files(root_path)
    lines = [
        "# MTPNet Codebase Bundle",
        "",
        f"Root: {root_path.resolve()}",
        f"File count: {len(files)}",
        "",
    ]
    for rel in files:
        path = root_path / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        fence = _fence_for(text)
        lang = _language(rel)
        lines.extend(
            [
                f"## {rel.as_posix()}",
                "",
                f"{fence}{lang}",
                text.rstrip("\n"),
                fence,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_bundle(root: str | Path, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_bundle(root), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect MTPNet codebase into one Markdown bundle.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/codebase_bundle.md"))
    args = parser.parse_args(argv)

    output = write_bundle(args.root, args.output)
    print(f"Wrote codebase bundle to {output}", flush=True)


if __name__ == "__main__":
    main()
