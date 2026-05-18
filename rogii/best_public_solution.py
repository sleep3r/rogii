from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLAIMED_SCORE_RE = re.compile(r"(?<!\d)([0-9]{1,2})[._-]([0-9]{3})(?!\d)")


@dataclass(frozen=True)
class Source:
    source_ref: str
    title: str
    author: str
    url: str
    votes: int
    updated_at: str
    score_claim: float | None


def claimed_score(text: str) -> float | None:
    values: list[float] = []
    for match in CLAIMED_SCORE_RE.finditer(text):
        value = float(f"{match.group(1)}.{match.group(2)}")
        if 5.0 <= value <= 30.0:
            values.append(value)
    return min(values) if values else None


def source_claim(row: sqlite3.Row) -> float | None:
    title_score = claimed_score(str(row["title"] or ""))
    ref_score = claimed_score(str(row["source_ref"] or ""))
    values = [value for value in (title_score, ref_score) if value is not None]
    return min(values) if values else None


def load_kernel_sources(db_path: Path) -> list[Source]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT source_ref, title, author, url, votes, updated_at
            FROM sources
            WHERE source_type = 'kernel'
            """
        ).fetchall()
    sources: list[Source] = []
    for row in rows:
        sources.append(
            Source(
                source_ref=str(row["source_ref"] or ""),
                title=str(row["title"] or ""),
                author=str(row["author"] or ""),
                url=str(row["url"] or ""),
                votes=int(row["votes"] or 0),
                updated_at=str(row["updated_at"] or ""),
                score_claim=source_claim(row),
            )
        )
    return sources


def select_best_source(sources: list[Source]) -> Source:
    claimed = [source for source in sources if source.score_claim is not None]
    if not claimed:
        raise RuntimeError("No public kernel with a score-like title was found.")
    return sorted(
        claimed,
        key=lambda source: (
            float(source.score_claim),
            -source.votes,
            source.updated_at,
            source.source_ref,
        ),
    )[0]


def load_ideas(db_path: Path, source_ref: str) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            SELECT idea_type, title, summary, evidence, tags, score
            FROM ideas
            WHERE source_ref = ?
            ORDER BY score DESC, idea_type
            """,
            (source_ref,),
        ).fetchall()


def extracted_source_path(work_dir: Path, source_ref: str) -> Path:
    return work_dir / "extracted" / f"{source_ref.replace('/', '__')}.py"


def code_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    keywords = {
        "artifact_dataset": lower.count("artifacts_path")
        + lower.count("kaggle/input/datasets"),
        "lightgbm": lower.count("lightgbm") + lower.count("lgbm"),
        "catboost": lower.count("catboost"),
        "xgboost": lower.count("xgboost") + lower.count("xgb"),
        "groupkfold": lower.count("groupkfold"),
        "hill_climb": lower.count("hill"),
        "dwt_or_wavelet_terms": lower.count("dwt")
        + lower.count("pywt")
        + lower.count("wavelet"),
        "dtw": lower.count("dtw"),
        "gpu_training": lower.count("gpu") + lower.count("task_type"),
        "optuna": lower.count("optuna"),
        "particle_filter": lower.count("particle") + lower.count("pf_"),
        "savgol": lower.count("savgol"),
        "submission": lower.count("submission"),
    }
    return {
        "exists": True,
        "path": str(path),
        "lines": text.count("\n") + 1,
        "bytes": len(text.encode("utf-8")),
        "keywords": keywords,
    }


def top_claimed_sources(sources: list[Source], limit: int) -> list[Source]:
    claimed = [source for source in sources if source.score_claim is not None]
    return sorted(
        claimed,
        key=lambda source: (
            float(source.score_claim),
            -source.votes,
            source.source_ref,
        ),
    )[:limit]


def markdown_table_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("\n", " ") for value in values) + " |"


def render_doc(
    selected: Source,
    alternatives: list[Source],
    ideas: list[sqlite3.Row],
    inventory: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Best Public Solution Context",
        "",
        "This document is generated from the local Kaggle mining database and is",
        "included in `.kaggle_mining/model_bundle.md`. Treat notebook titles and",
        "claimed LB values as public evidence, not guaranteed private-LB truth.",
        "",
        "## Selected Open Solution",
        "",
        f"- Source: `{selected.source_ref}`",
        f"- Title: {selected.title}",
        f"- Author: {selected.author}",
        f"- Claimed/public title score: {selected.score_claim:.3f}",
        f"- Votes in mining DB: {selected.votes}",
        f"- Updated at: {selected.updated_at}",
        f"- URL: {selected.url}",
        "",
        "Selection rule: among mined public Kaggle kernels, choose the lowest",
        "score-looking value in the title or slug, then prefer higher votes.",
        "",
        "## Technical Takeaways",
        "",
    ]
    if ideas:
        for idea in ideas:
            lines.append(
                f"- `{idea['idea_type']}`: {idea['summary']}"
                + (f" ({idea['tags']})" if idea["tags"] else "")
            )
    else:
        lines.append("- No extracted idea rows were available for this source.")

    lines.extend(
        [
            "",
            "The raw idea rows above come from automatic mining and can be noisy;",
            "for example, notebook logs and array literals may be misread as scores.",
            "Use them as pointers, not as audited metrics.",
            "",
            "## Code Inventory",
            "",
        ]
    )
    if inventory.get("exists"):
        lines.extend(
            [
                f"- Local extracted source: `{inventory['path']}`",
                f"- Size: {inventory['lines']} lines, {inventory['bytes']} bytes",
                "",
                "| signal | count |",
                "| --- | ---: |",
            ]
        )
        for key, value in sorted(inventory["keywords"].items()):
            lines.append(markdown_table_row([key, str(value)]))
    else:
        lines.append(
            "- Local extracted source is not available yet. Run `make mine-code`."
        )

    lines.extend(
        [
            "",
            "## Notebook-Specific Observations",
            "",
            "- The selected public notebook can load prebuilt train/model artifacts when",
            "  they are available, so its runtime and exact feature provenance are not",
            "  fully represented by the visible notebook code.",
            "- The visible training stack is 3 LightGBM + 3 CatBoost, grouped by well id,",
            "  with GPU-oriented parameters in the public notebook.",
            "- The visible alignment block is heavy on beam/DTW/stochastic DTW/PF/ANCC",
            "  signals. The title says DWT-based; in the extracted script, explicit DWT",
            "  terms are sparse, so DWT may be hidden in the prebuilt artifact table or",
            "  in the copied notebook lineage.",
            "- Postprocess is tuned with Optuna in the public notebook; our framework",
            "  uses deterministic grids plus exact fast scoring to keep experiments",
            "  reproducible and cheap.",
            "",
            "## Paraphrased Implementation Shape",
            "",
            "The selected solution family is valuable because it treats TVT prediction",
            "as a geosteering alignment problem, not just a flat tabular regression:",
            "",
            "1. Build a hidden-zone residual target relative to the last known TVT.",
            "2. Generate trajectory, GR, typewell, wavelet/DWT, DTW, PF, and spatial",
            "   calibration signals as candidate explanations of geological position.",
            "3. Train multiple gradient-boosted tree variants with grouped validation by",
            "   well id.",
            "4. Blend OOF predictions with a non-negative hill-climb or similar",
            "   leaderboard-stable weighted average.",
            "5. Apply a conservative geological postprocess that blends model delta with",
            "   a physically plausible alignment/PF delta and optional per-well",
            "   smoothing.",
            "",
            "## Relation To Our Current Framework",
            "",
            "- Already implemented locally: residual target, DWT/DTW/PF/beam/spatial",
            "  feature families, 3 LightGBM + 3 CatBoost stack, hill-climb blending,",
            "  PF-aware postprocess, inference-only Kaggle submit, profiling ledger.",
            "- Current public anchor: our global-context schema-v4 artifact scored 9.946.",
            "- Current validation focus: schema-v6 fold-safe OOF with context-safe",
            "  `KaggleTopContext`, plus profiling-driven speedups.",
            "- Main remaining comparison task: check column-by-column parity with the",
            "  selected public solution only where the idea is genuinely reproducible,",
            "  then keep the simpler local implementation if LB/CV is comparable.",
            "",
            "## Other High-Claim Public References",
            "",
            "| claimed score | votes | source | title |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for source in alternatives:
        lines.append(
            markdown_table_row(
                [
                    f"{source.score_claim:.3f}"
                    if source.score_claim is not None
                    else "",
                    str(source.votes),
                    f"`{source.source_ref}`",
                    source.title,
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path(".kaggle_mining/ideas.sqlite"))
    parser.add_argument("--work-dir", type=Path, default=Path(".kaggle_mining"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".kaggle_mining/best_public_solution.md"),
    )
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)

    sources = load_kernel_sources(args.db)
    selected = select_best_source(sources)
    ideas = load_ideas(args.db, selected.source_ref)
    inventory = code_inventory(
        extracted_source_path(args.work_dir, selected.source_ref)
    )
    alternatives = top_claimed_sources(sources, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_doc(selected, alternatives, ideas, inventory),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected.source_ref,
                "claimed_score": selected.score_claim,
                "votes": selected.votes,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
