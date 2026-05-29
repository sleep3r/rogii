from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bphwt.priors.prob_residual_hmm import (
    ProbResidualHMMConfig,
    forward_backward_banded,
    infer_residual_hmm,
    make_base_tvt,
    robust_zscore,
    viterbi_banded,
)


def test_make_base_tvt_interpolates_and_pins_anchors() -> None:
    md = np.arange(6, dtype=np.float64)
    tvt_input = np.array([np.nan, 100.0, np.nan, np.nan, 110.0, np.nan], dtype=np.float64)

    base, known = make_base_tvt(tvt_input, md)

    np.testing.assert_array_equal(known, [False, True, False, False, True, False])
    np.testing.assert_allclose(base, [100.0, 100.0, 103.333333, 106.666667, 110.0, 110.0])
    np.testing.assert_allclose(base[known], tvt_input[known])


def test_robust_zscore_returns_zeros_when_too_few_valid_values() -> None:
    z, ok, center, scale = robust_zscore(np.array([1.0, np.nan, 2.0, np.nan]), min_valid=3)

    assert not ok
    assert center == 0.0
    assert scale == 1.0
    np.testing.assert_allclose(z, np.zeros(4))


def test_forward_backward_and_viterbi_prefer_observed_residual() -> None:
    r_grid = np.arange(-2.0, 3.0, 1.0)
    log_obs = np.tile(-0.5 * ((r_grid - 1.0) / 0.1) ** 2, (5, 1))

    fb = forward_backward_banded(log_obs, r_grid, sigma_rw=1.0, band=2)
    path, path_loglik = viterbi_banded(log_obs, r_grid, sigma_rw=1.0, band=2)

    np.testing.assert_allclose(fb["posterior"].sum(axis=1), np.ones(5), atol=1e-6)
    np.testing.assert_allclose(fb["posterior_mean"], np.ones(5), atol=1e-3)
    assert np.all(fb["posterior_std"] < 0.05)
    np.testing.assert_allclose(path, np.ones(5))
    assert np.isfinite(fb["log_evidence"])
    assert np.isfinite(path_loglik)


def test_infer_residual_hmm_improves_synthetic_calibrated_well_and_pins_anchors() -> None:
    md = np.arange(120, dtype=np.float64)
    linear_tvt = 1000.0 + 0.32 * md
    true_tvt = linear_tvt + 8.0 * np.sin(md / 18.0)
    tw_tvt = np.linspace(true_tvt.min() - 40.0, true_tvt.max() + 40.0, 500)
    tw_gr = 30.0 + 0.8 * tw_tvt + 3.0 * np.sin(tw_tvt / 9.0)
    gr = np.interp(true_tvt, tw_tvt, tw_gr)
    gr[35:41] = np.nan

    tvt_input = np.full_like(true_tvt, np.nan)
    tvt_input[:12] = true_tvt[:12]
    tvt_input[-12:] = true_tvt[-12:]
    hidden = ~np.isfinite(tvt_input)
    base, _ = make_base_tvt(tvt_input, md)

    cfg = ProbResidualHMMConfig(
        row_stride=3,
        residual_range_ft=30.0,
        residual_step_ft=1.0,
        sigma_rw=2.5,
        sigma_base=20.0,
        sigma_anchor=0.5,
        use_zscore_gr=False,
        use_calibrated_gr_if_possible=True,
        min_calibration_points=8,
        sigma_gr_raw=1.0,
        blend_alpha=1.0,
    )

    result = infer_residual_hmm(md, gr, tvt_input, tw_tvt, tw_gr, cfg)

    np.testing.assert_allclose(result["pred_mean"][~hidden], tvt_input[~hidden], atol=1e-6)
    assert np.isfinite(result["pred_mean"]).all()
    assert np.isfinite(result["posterior_std"]).all()
    base_rmse = float(np.sqrt(np.mean((base[hidden] - true_tvt[hidden]) ** 2)))
    hmm_rmse = float(np.sqrt(np.mean((result["pred_mean"][hidden] - true_tvt[hidden]) ** 2)))
    assert hmm_rmse < base_rmse


def test_infer_residual_hmm_handles_missing_gr_and_sparse_anchors() -> None:
    md = np.arange(20, dtype=np.float64)
    gr = np.full(20, np.nan, dtype=np.float64)
    tvt_input = np.full(20, np.nan, dtype=np.float64)
    tvt_input[0] = 1000.0
    tw_tvt = np.linspace(950.0, 1050.0, 101)
    tw_gr = np.sin(tw_tvt / 10.0)

    result = infer_residual_hmm(
        md,
        gr,
        tvt_input,
        tw_tvt,
        tw_gr,
        ProbResidualHMMConfig(row_stride=5, residual_range_ft=10.0, residual_step_ft=1.0),
    )

    assert np.isfinite(result["pred_mean"]).all()
    assert np.isfinite(result["pred_viterbi"]).all()
    assert np.isfinite(result["posterior_std"]).all()
    assert result["pred_mean"][0] == 1000.0


def test_prob_hmm_score_writes_baseline_artifacts(tmp_path: Path) -> None:
    from scripts import prob_hmm_score

    data_dir = tmp_path / "train"
    out_dir = tmp_path / "artifacts"
    _write_synthetic_csv_well(data_dir, "abc12345")

    summary = prob_hmm_score.run(
        SimpleNamespace(data_dir=str(data_dir), out_dir=str(out_dir), limit=0, no_progress=True)
    )

    assert (out_dir / "base_oof.csv").exists()
    assert (out_dir / "base_summary.json").exists()
    written = json.loads((out_dir / "base_summary.json").read_text())
    assert summary["n_wells"] == 1
    assert written["n_wells"] == 1
    assert written["n_hidden_rows"] > 0
    assert np.isfinite(written["rmse_base"])


def test_run_prob_hmm_oof_writes_experiment_artifacts(tmp_path: Path) -> None:
    from scripts import run_prob_hmm_oof

    data_dir = tmp_path / "train"
    out_dir = tmp_path / "artifacts"
    _write_synthetic_csv_well(data_dir, "abc12345")

    summary = run_prob_hmm_oof.run(
        SimpleNamespace(
            data_dir=str(data_dir),
            out_dir=str(out_dir),
            limit=1,
            no_progress=True,
            blend_alphas="0.5",
        )
    )

    experiments = pd.read_csv(out_dir / "experiments.csv")
    best = pd.read_csv(out_dir / "per_well_best.csv")
    assert summary["n_wells"] == 1
    assert len(experiments) == 3
    assert len(best) == 1
    assert np.isfinite(experiments["rmse_blend"]).all()


def _write_synthetic_csv_well(data_dir: Path, well_id: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    md = np.arange(60, dtype=np.float64)
    tvt = 1000.0 + 0.25 * md + 5.0 * np.sin(md / 12.0)
    tw_tvt = np.linspace(tvt.min() - 30.0, tvt.max() + 30.0, 180)
    tw_gr = 40.0 + 0.5 * tw_tvt
    gr = np.interp(tvt, tw_tvt, tw_gr)
    tvt_input = np.full_like(tvt, np.nan)
    tvt_input[:8] = tvt[:8]
    tvt_input[-8:] = tvt[-8:]

    pd.DataFrame(
        {
            "MD": md,
            "X": md,
            "Y": md,
            "Z": -md,
            "TVT": tvt,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    ).to_csv(data_dir / f"{well_id}__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr, "Geology": ["A"] * len(tw_tvt)}).to_csv(
        data_dir / f"{well_id}__typewell.csv",
        index=False,
    )
