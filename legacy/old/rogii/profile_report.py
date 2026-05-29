from __future__ import annotations

import argparse
import datetime as dt
import pstats
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileRow:
    name: str
    location: str
    primitive_calls: int
    total_calls: int
    self_time: float
    cumulative_time: float


@dataclass(frozen=True)
class StageRow:
    stage: str
    duration: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report from a cProfile file and ROGII log."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--title", default="ROGII Profiling Report")
    return parser.parse_args()


def shorten_path(path: str) -> str:
    cwd = Path.cwd().resolve()
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        parts = resolved.parts
        if "site-packages" in parts:
            index = parts.index("site-packages")
            return str(Path(*parts[index:]))
        return resolved.name


def load_profile_rows(profile_path: Path) -> tuple[pstats.Stats, list[ProfileRow]]:
    stats = pstats.Stats(str(profile_path))
    rows: list[ProfileRow] = []
    for (filename, line, func_name), values in stats.stats.items():
        primitive_calls, total_calls, self_time, cumulative_time, _ = values
        rows.append(
            ProfileRow(
                name=func_name,
                location=f"{shorten_path(filename)}:{line}",
                primitive_calls=int(primitive_calls),
                total_calls=int(total_calls),
                self_time=float(self_time),
                cumulative_time=float(cumulative_time),
            )
        )
    return stats, rows


def parse_stage_rows(log_path: Path | None) -> tuple[list[StageRow], str | None]:
    if log_path is None or not log_path.is_file():
        return [], None
    duration_pattern = re.compile(r"\bduration=([0-9:.]+[smh]?)")
    total_pattern = re.compile(r"\btotal_duration=([0-9:.]+[smh]?)")
    stages: list[StageRow] = []
    total_duration: str | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split("|", maxsplit=4)]
        if len(parts) < 4:
            continue
        level = parts[2]
        message = parts[3]
        details = parts[4] if len(parts) > 4 else ""
        if level == "OK":
            match = duration_pattern.search(details)
            if match:
                stages.append(StageRow(stage=message, duration=match.group(1)))
        elif level == "DONE":
            match = total_pattern.search(details)
            if match:
                total_duration = match.group(1)
    return stages, total_duration


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def rows_to_markdown(rows: list[ProfileRow], limit: int, sort_key: str) -> str:
    if sort_key == "self":
        selected = sorted(rows, key=lambda row: row.self_time, reverse=True)
    else:
        selected = sorted(rows, key=lambda row: row.cumulative_time, reverse=True)
    table_rows = [
        [
            str(index),
            row.name.replace("|", "\\|"),
            row.location.replace("|", "\\|"),
            format_seconds(row.cumulative_time),
            format_seconds(row.self_time),
            f"{row.total_calls:,}",
            f"{row.primitive_calls:,}",
        ]
        for index, row in enumerate(selected[:limit], start=1)
    ]
    return markdown_table(
        [
            "#",
            "function",
            "location",
            "cumtime",
            "selftime",
            "calls",
            "primitive",
        ],
        table_rows,
    )


def write_report(args: argparse.Namespace) -> None:
    stats, rows = load_profile_rows(args.profile)
    stages, total_duration = parse_stage_rows(args.log)
    generated = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()

    lines: list[str] = [
        f"# {args.title}",
        "",
        f"- Generated: `{generated}`",
        f"- Profile: `{args.profile}`",
    ]
    if args.log is not None:
        lines.append(f"- Log: `{args.log}`")
    lines.extend(
        [
            f"- cProfile total time: `{format_seconds(stats.total_tt)}`",
            f"- Total calls: `{stats.total_calls:,}`",
            f"- Primitive calls: `{stats.prim_calls:,}`",
        ]
    )
    if total_duration:
        lines.append(f"- RunLogger total duration: `{total_duration}`")

    if stages:
        lines.extend(
            [
                "",
                "## RunLogger Stage Timings",
                "",
                markdown_table(
                    ["stage", "duration"],
                    [[row.stage.replace("|", "\\|"), row.duration] for row in stages],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Top Cumulative Time",
            "",
            rows_to_markdown(rows, args.limit, "cumulative"),
            "",
            "## Top Self Time",
            "",
            rows_to_markdown(rows, args.limit, "self"),
            "",
            "## Reading Notes",
            "",
            "- `cumtime` includes time spent inside callees and is best for finding expensive call paths.",
            "- `selftime` excludes callees and is best for finding hot loops inside one function.",
            "- Compare reports with the same config, cache state, and machine whenever possible.",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote profile report: {args.output}", flush=True)


def main() -> None:
    write_report(parse_args())


if __name__ == "__main__":
    main()
