from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass(frozen=True)
class ReportFigure:
    path: Path
    relative_path: str
    title: str


FIGURE_TITLE_OVERRIDES = {
    "window_metrics.png": "Window-Level MTP Metrics",
    "sanity_gaps.png": "GR / Prior Sanity Gaps",
    "row_level_progress.png": "Row-Level Tracker Progress",
    "ranker_leakage.png": "Ranker Leakage Audit",
    "corr_panel_hit_rates.png": "Global Correlation Hit Rates",
    "corr_panel_distribution.png": "Correlation Oracle Distribution",
    "corr_panel_global_vs_localized.png": "Global vs Localized vs Stretch Correlation",
    "typewell_mismatch_distribution.png": "True-Path Typewell Mismatch Distribution",
    "typewell_mismatch_scatter.png": "Typewell Mismatch Scatter",
    "formation_correlation_comparison.png": "Formation-Aware Correlation Upper Bound",
    "pseudo_zone_distribution.png": "Test-Schema-Safe Pseudo-Zone Baseline",
    "tail_class_counts.png": "Tail Class Counts",
    "tail_class_rmse.png": "Tail Class RMSE",
}


def _title_from_path(path: Path) -> str:
    if path.name in FIGURE_TITLE_OVERRIDES:
        return FIGURE_TITLE_OVERRIDES[path.name]
    return path.stem.replace("_", " ").replace("-", " ").title()


def collect_report_figures(figures_dir: Path) -> list[ReportFigure]:
    """Collect top-level figures first, then nested panels, in stable order."""
    if not figures_dir.exists():
        return []
    top_level = sorted(path for path in figures_dir.glob("*.png") if path.is_file())
    nested = sorted(path for path in figures_dir.glob("*/*.png") if path.is_file())
    figures = []
    for path in [*top_level, *nested]:
        figures.append(
            ReportFigure(
                path=path,
                relative_path=path.relative_to(figures_dir).as_posix(),
                title=_title_from_path(path),
            )
        )
    return figures


def _read_markdown_text(report_path: Path) -> str:
    return report_path.read_text() if report_path.exists() else ""


def _extract_section(markdown: str, heading: str) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)", re.M | re.S)
    match = pattern.search(markdown)
    if not match:
        return ""
    body = match.group("body")
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", body)
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    return body.strip()


def _wrap_lines(text: str, width: int = 96) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        if raw.startswith("|"):
            lines.append(raw)
            continue
        raw = raw.replace("`", "")
        wrapped = textwrap.wrap(raw, width=width, replace_whitespace=False)
        lines.extend(wrapped or [""])
    return lines


def _new_text_page(title: str, lines: Iterable[str]) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")
    fig.text(0.08, 0.955, title, fontsize=18, fontweight="bold", color="#1f2937")
    y = 0.98
    for line in lines:
        if y < 0.02:
            break
        if line.startswith("#"):
            continue
        if line.startswith("- "):
            ax.text(0.02, y, "• " + line[2:], fontsize=9.2, va="top", color="#111827")
            y -= 0.032
        elif line.startswith("|"):
            ax.text(0.0, y, line, fontsize=6.4, family="monospace", va="top", color="#111827")
            y -= 0.024
        elif not line:
            y -= 0.018
        else:
            ax.text(0.0, y, line, fontsize=9.2, va="top", color="#111827")
            y -= 0.030
    return fig


def _figure_page(report_figure: ReportFigure) -> plt.Figure:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.94, report_figure.title, fontsize=16, fontweight="bold", color="#1f2937")
    fig.text(0.05, 0.905, report_figure.relative_path, fontsize=8, color="#6b7280")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.80])
    ax.axis("off")
    image = mpimg.imread(report_figure.path)
    ax.imshow(image)
    return fig


def write_diagnostic_pdf(
    *,
    report_path: Path = Path("artifacts/diagnostic_report_v1/diagnostic_report.md"),
    figures_dir: Path = Path("artifacts/diagnostic_report_v1/figures"),
    output_path: Path = Path("artifacts/diagnostic_report_v1/diagnostic_report.pdf"),
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _read_markdown_text(report_path)
    figures = collect_report_figures(figures_dir)
    with PdfPages(output_path) as pdf:
        cover_lines = [
            "ROGII / GeoMTP Diagnostics",
            "",
            f"Source report: {report_path.as_posix()}",
            f"Figures included: {len(figures)}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "Purpose: compact PDF pack for external review / stronger-model handoff.",
        ]
        fig = _new_text_page("GeoMTP Diagnostic Pack", cover_lines)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for section in [
            "Executive Summary",
            "Research Expectation vs What We Tested",
            "What Is Actually Not Working",
            "Concrete Next Diagnostics",
            "Kaggle Discussion Draft",
            "Artifact Index",
        ]:
            body = _extract_section(markdown, section)
            if body:
                fig = _new_text_page(section, _wrap_lines(body))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        for report_figure in figures:
            fig = _figure_page(report_figure)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        appendix_lines = [f"- {item.relative_path}" for item in figures]
        fig = _new_text_page("Included Figures", appendix_lines)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return output_path


def main() -> None:
    path = write_diagnostic_pdf()
    print(f"Wrote diagnostic PDF to {path}", flush=True)


if __name__ == "__main__":
    main()
