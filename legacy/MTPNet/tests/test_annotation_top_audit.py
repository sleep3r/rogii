from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_annotation_frame() -> pd.DataFrame:
    rows = []
    for well_id, offset in [("a", 0.0), ("b", 5.0)]:
        for row_idx in range(12):
            tvt = 100.0 + offset + row_idx * 2.0
            rows.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "MD": 1000.0 + row_idx * 10.0,
                    "X": float(row_idx),
                    "Y": 0.0,
                    "Z": -row_idx * 2.0,
                    "GR": 80.0 + row_idx,
                    "TVT": tvt,
                    "TVT_input": tvt if row_idx < 4 else np.nan,
                    "ANCC": tvt + 0.5,
                    "ASTNU": tvt + 10.0,
                    "ASTNL": tvt + 20.0,
                    "EGFDU": tvt + 30.0,
                    "EGFDL": tvt + 40.0,
                    "BUDA": tvt + 50.0,
                }
            )
    return pd.DataFrame(rows)


def test_uniform_piecewise_reconstruction_is_exact_for_linear_segments() -> None:
    from mtpnet.annotation_top_audit import uniform_piecewise_reconstruct

    x = np.arange(9, dtype=np.float64)
    y = np.piecewise(x, [x <= 4, x > 4], [lambda v: 2.0 * v + 1.0, lambda v: -v + 13.0])

    recon, control_count = uniform_piecewise_reconstruct(y, spacing_rows=4)

    assert control_count == 3
    assert np.max(np.abs(recon - y)) < 1e-9


def test_state_agreement_detects_matching_direction() -> None:
    from mtpnet.annotation_top_audit import state_agreement

    ref = np.asarray([0.0, 1.0, 3.0, 6.0])
    same = np.asarray([10.0, 11.0, 13.0, 16.0])
    opposite = np.asarray([10.0, 9.0, 7.0, 4.0])

    assert state_agreement(ref, same)["sign_agree"] == 1.0
    assert state_agreement(ref, opposite)["sign_agree"] == 0.0
    assert state_agreement(ref, same)["corr"] > 0.99
    assert state_agreement(ref, opposite)["corr"] < -0.99


def test_run_annotation_top_audit_writes_metrics_report_and_figures(tmp_path: Path) -> None:
    from mtpnet.annotation_top_audit import AnnotationTopAuditConfig, run_annotation_top_audit_from_frame

    metrics = run_annotation_top_audit_from_frame(
        _tiny_annotation_frame(),
        output_dir=tmp_path,
        config=AnnotationTopAuditConfig(control_spacing_rows=4, n_panel_wells=1),
    )

    assert metrics["wells"] == 2
    assert metrics["summary"]["ANCC"]["piecewise_rmse_ft_mean"] < 1e-9
    assert metrics["summary"]["ANCC"]["sign_agree_top_vs_tvt_hidden"] == 1.0
    assert metrics["summary"]["ANCC"]["sign_agree_top_vs_negz_hidden"] == 1.0
    assert (tmp_path / "annotation_top_metrics.json").exists()
    assert (tmp_path / "annotation_top_report.md").exists()
    assert (tmp_path / "annotation_top_well_metrics.csv").exists()
    assert list((tmp_path / "figures").glob("*.png"))
