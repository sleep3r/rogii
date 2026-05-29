"""
dump_codebase.py — собирает все исходные .py файлы проекта в один текстовый файл.

Использование:
    python dump_codebase.py [--out codebase.txt]
"""
import argparse
import os
import sys
from pathlib import Path

EXTENSIONS = {".py", ".toml", ".json"}
SKIP_DIRS  = {"__pycache__", ".git", "artifacts", ".DS_Store"}


def collect(root: Path, out_path: Path) -> int:
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix in EXTENSIONS
        and not any(part in SKIP_DIRS for part in p.parts)
        and p != out_path
    )

    lines_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"# CODEBASE DUMP — {root}\n")
        fout.write(f"# {len(files)} files\n\n")
        for path in files:
            rel = path.relative_to(root)
            header = f"{'=' * 72}\n# FILE: {rel}\n{'=' * 72}\n"
            fout.write(header)
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as e:
                text = f"# [READ ERROR: {e}]\n"
            fout.write(text)
            if not text.endswith("\n"):
                fout.write("\n")
            fout.write("\n")
            n = text.count("\n")
            lines_written += n
            print(f"  {rel}  ({n} lines)")

    return lines_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Корневая директория проекта")
    ap.add_argument("--out",  default="codebase.txt", help="Выходной файл")
    args = ap.parse_args()

    root     = Path(args.root).resolve()
    out_path = (root / args.out).resolve()

    print(f"Сканирую: {root}")
    print(f"Вывод:    {out_path}\n")

    total = collect(root, out_path)
    size  = out_path.stat().st_size

    print(f"\nГотово: {total} строк, {size / 1024:.1f} КБ → {out_path.name}")


if __name__ == "__main__":
    main()
