from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bphwt.infer.xyz_curve import XYZCurveConfig, smooth_residual_curve
from tests.test_xyz_steering import _write_synthetic_well


def test_smooth_residual_curve_pins_known_rows_and_reduces_noise() -> None:
    md = np.arange(80, dtype=np.float64)
    true_r = 6.0 * np.sin(md / 14.0)
    rng = np.random.default_rng(123)
    prior = true_r + rng.normal(0.0, 3.0, size=md.size)
    known = np.zeros(md.size, dtype=bool)
    known[:5] = True
    known[-5:] = True
    prior[known] = 20.0

    smoothed = smooth_residual_curve(
        md,
        prior,
        known,
        XYZCurveConfig(lambda_smooth=40.0, lambda_base=0.01, lambda_anchor=1.0e6),
    )

    assert np.isfinite(smoothed).all()
    np.testing.assert_allclose(smoothed[known], 0.0, atol=1.0e-3)
    hidden = ~known
    noisy_rmse = float(np.sqrt(np.mean((prior[hidden] - true_r[hidden]) ** 2)))
    smooth_rmse = float(np.sqrt(np.mean((smoothed[hidden] - true_r[hidden]) ** 2)))
    assert smooth_rmse < noisy_rmse


def test_run_xyz_curve_oof_writes_artifacts(tmp_path: Path) -> None:
    from scripts import run_xyz_curve_oof

    data_dir = tmp_path / "train"
    out_dir = tmp_path / "curve_artifacts"
    for i in range(8):
        _write_synthetic_well(data_dir, f"well{i:04d}", phase=float(i) / 3.0)

    summary = run_xyz_curve_oof.run(
        SimpleNamespace(
            data_dir=str(data_dir),
            out_dir=str(out_dir),
            limit=0,
            n_splits=4,
            train_stride=2,
            max_train_rows=5000,
            max_iter=30,
            learning_rate=0.08,
            l2_regularization=0.01,
            max_leaf_nodes=15,
            mode="full",
            blend_alphas="0.5,1.0",
            smooth_lambdas="0,20",
            base_lambdas="0.0,0.05",
            anchor_lambda=1.0e6,
            clip_residual=80.0,
            no_progress=True,
        )
    )

    experiments = pd.read_csv(out_dir / "experiments.csv")
    per_well = pd.read_csv(out_dir / "per_well.csv")

    assert summary["n_wells"] == 8
    assert "base" in set(experiments["experiment"])
    assert any(str(x).startswith("curve_full") for x in experiments["experiment"])
    assert len(per_well) == 8 * (1 + 2 * 2 * 2)
    assert np.isfinite(experiments["rmse"]).all()
