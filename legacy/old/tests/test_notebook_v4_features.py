from __future__ import annotations

import numpy as np

from rogii.notebook_v4_features import build_notebook_v4_features


def test_notebook_v4_features_add_dwt_slope_and_divergence_signals() -> None:
    n = 96
    md = np.arange(n, dtype=float)
    gr = 80.0 + 12.0 * np.sin(md / 6.0)
    tvt = 1100.0 + 0.04 * md + 0.0005 * md**2
    tvt_input = tvt.copy()
    tvt_input[60:] = np.nan
    flat = np.full(n, tvt_input[59], dtype=float)

    base_features = {
        "kg_path_stage12_tvt": flat + 1.0,
        "kg_ncc_mean_tvt": flat + 3.5,
        "kg_beam_mean_tvt": flat - 2.0,
        "pf_ancc": flat + 0.5,
        "tvt_structural": flat + 2.0,
    }

    out = build_notebook_v4_features(
        md=md,
        gr=gr,
        tvt_input=tvt_input,
        flat_pred=flat,
        features=base_features,
    )
    hidden = np.flatnonzero(~np.isfinite(tvt_input))

    expected = {
        "nbv4_gr_dwt_approx",
        "nbv4_gr_dwt_detail_energy",
        "nbv4_gr_dwt_residual",
        "nbv4_anchor_slope_k10",
        "nbv4_tvt_extrap_k10",
        "nbv4_tvt_extrap_k50_minus_last",
        "nbv4_slope_accel_10_50",
        "nbv4_path_vs_ncc",
        "nbv4_ncc_vs_beam",
        "nbv4_estimator_drift_range",
    }
    assert expected.issubset(out)
    for name in expected:
        assert out[name].shape == (n,), name
        assert np.isfinite(out[name][hidden]).all(), name

    assert np.allclose(out["nbv4_path_vs_ncc"][hidden], -2.5)
    assert np.allclose(out["nbv4_ncc_vs_beam"][hidden], 5.5)
    assert np.all(out["nbv4_estimator_drift_range"][hidden] >= 5.5)
