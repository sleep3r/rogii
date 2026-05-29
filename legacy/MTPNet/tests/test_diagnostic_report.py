from __future__ import annotations

from pathlib import Path

from mtpnet.diagnostic_report import write_diagnostic_report


def test_write_diagnostic_report_creates_markdown_and_figures(tmp_path: Path) -> None:
    report_path = write_diagnostic_report(
        artifacts_dir=Path("artifacts"),
        output_dir=tmp_path,
    )

    text = report_path.read_text()
    assert "GEOMTP_DIAGNOSTIC_REPORT" in text
    assert "Root Diagnosis" in text
    assert "Kaggle Discussion Draft" in text
    assert "Localized + multiscale + stretch/squeeze panel" in text
    assert "True-Path Typewell Mismatch Audit" in text
    assert "Formation-Aware Correlation" in text
    assert "Kaggle test typewells do not include `Geology`" in text
    assert "Test-Schema-Safe Pseudo-Zone Baseline" in text
    assert (tmp_path / "figures" / "corr_panel_hit_rates.png").exists()
    assert (tmp_path / "figures" / "typewell_mismatch_distribution.png").exists()
    assert (tmp_path / "figures" / "formation_correlation_comparison.png").exists()
    assert (tmp_path / "figures" / "pseudo_zone_distribution.png").exists()
    assert (tmp_path / "figures" / "tail_well_panels").exists()
