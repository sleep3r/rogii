from __future__ import annotations

import argparse
from pathlib import Path


def filter_lines(text: str, excluded_fragments: list[str]) -> str:
    kept = []
    for line in text.splitlines():
        if any(fragment in line for fragment in excluded_fragments):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args(argv)

    if not args.exclude:
        return
    text = args.path.read_text(encoding="utf-8", errors="replace")
    args.path.write_text(filter_lines(text, args.exclude), encoding="utf-8")


if __name__ == "__main__":
    main()
