from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mtpnet.pdf_report import collect_report_figures, write_diagnostic_pdf


def _write_png(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def test_collect_report_figures_includes_nested_tail_panels(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"
    _write_png(figures_dir / "summary.png", "summary")
    _write_png(figures_dir / "tail_well_panels" / "well_a.png", "well")

    figures = collect_report_figures(figures_dir)

    assert [item.relative_path for item in figures] == [
        "summary.png",
        "tail_well_panels/well_a.png",
    ]


def test_write_diagnostic_pdf_writes_valid_pdf_with_figures(tmp_path: Path) -> None:
    report = tmp_path / "diagnostic_report.md"
    report.write_text("# Report\n\n## Executive Summary\n\nA short summary.\n")
    figures_dir = tmp_path / "figures"
    _write_png(figures_dir / "summary.png", "summary")
    _write_png(figures_dir / "tail_well_panels" / "well_a.png", "well")
    output = tmp_path / "report.pdf"

    result = write_diagnostic_pdf(
        report_path=report,
        figures_dir=figures_dir,
        output_path=output,
    )

    assert result == output
    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 10_000
