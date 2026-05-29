from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bphwt.features.xyz_steering import XYZSteeringConfig, build_xyz_steering_features


def test_build_xyz_steering_features_separates_past_and_full_future_signal() -> None:
    md = np.arange(12, dtype=np.float64)
    x = md * 10.0
    y = np.zeros_like(md)
    z = -100.0 - 0.2 * md

    cfg = XYZSteeringConfig(horizons_ft=(3.0,))
    past = build_xyz_steering_features(md, x, y, z, known_mask=np.zeros_like(md, dtype=bool), cfg=cfg, mode="past")
    full = build_xyz_steering_features(md, x, y, z, known_mask=np.zeros_like(md, dtype=bool), cfg=cfg, mode="full")

    assert "future_z_delta_3" not in past.columns
    assert "future_z_delta_3" in full.columns
    assert "future_z_deviation_3" in full.columns

    z_changed = z.copy()
    z_changed[8:] -= 20.0
    past_changed = build_xyz_steering_features(
        md,
        x,
        y,
        z_changed,
        known_mask=np.zeros_like(md, dtype=bool),
        cfg=cfg,
        mode="past",
    )
    full_changed = build_xyz_steering_features(
        md,
        x,
        y,
        z_changed,
        known_mask=np.zeros_like(md, dtype=bool),
        cfg=cfg,
        mode="full",
    )

    pd.testing.assert_series_equal(past.loc[:4, "past_z_delta_3"], past_changed.loc[:4, "past_z_delta_3"])
    assert not np.allclose(full.loc[5:6, "future_z_deviation_3"], full_changed.loc[5:6, "future_z_deviation_3"])


def test_run_xyz_steering_oof_writes_artifacts(tmp_path: Path) -> None:
    from scripts import run_xyz_steering_oof

    data_dir = tmp_path / "train"
    out_dir = tmp_path / "xyz_artifacts"
    for i in range(8):
        _write_synthetic_well(data_dir, f"well{i:04d}", phase=float(i) / 3.0)

    summary = run_xyz_steering_oof.run(
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
            blend_alphas="0.5,1.0",
            modes="past,full",
            no_progress=True,
        )
    )

    experiments = pd.read_csv(out_dir / "experiments.csv")
    per_well = pd.read_csv(out_dir / "per_well.csv")

    assert summary["n_wells"] == 8
    assert {"base", "past_a0p5", "past_a1p0", "full_a0p5", "full_a1p0"}.issubset(
        set(experiments["experiment"])
    )
    assert len(per_well) == 8 * 5
    assert np.isfinite(experiments["rmse"]).all()


def test_make_xyz_steering_submission_writes_sample_ids(tmp_path: Path) -> None:
    from scripts import make_xyz_steering_submission

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    out_path = tmp_path / "submission.csv"
    for i in range(8):
        _write_synthetic_well(train_dir, f"well{i:04d}", phase=float(i) / 3.0)
    _write_synthetic_well(test_dir, "test0001", phase=0.7)
    test_hw = test_dir / "test0001__horizontal_well.csv"
    test_df = pd.read_csv(test_hw).drop(columns=["TVT"])
    test_df.to_csv(test_hw, index=False)

    hidden_rows = np.flatnonzero(~np.isfinite(test_df["TVT_input"].to_numpy(dtype=np.float64)))
    sample = pd.DataFrame({"id": [f"test0001_{int(i)}" for i in hidden_rows[:12]], "tvt": 0.0})
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)

    summary = make_xyz_steering_submission.run(
        SimpleNamespace(
            train_dir=str(train_dir),
            test_dir=str(test_dir),
            sample_submission=str(sample_path),
            out=str(out_path),
            mode="full",
            blend_alpha=1.0,
            train_stride=2,
            max_train_rows=5000,
            max_iter=30,
            learning_rate=0.08,
            l2_regularization=0.01,
            max_leaf_nodes=15,
            clip_residual=80.0,
            no_progress=True,
        )
    )

    sub = pd.read_csv(out_path)
    assert summary["n_rows"] == len(sample)
    assert sub["id"].tolist() == sample["id"].tolist()
    assert np.isfinite(sub["tvt"]).all()
    assert not np.allclose(sub["tvt"].to_numpy(dtype=np.float64), 0.0)


def _write_synthetic_well(data_dir: Path, well_id: str, phase: float) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    md = np.arange(90, dtype=np.float64)
    residual = 7.0 * np.sin(md / 11.0 + phase)
    tvt = 1000.0 + 0.28 * md + residual
    tvt_input = np.full_like(tvt, np.nan)
    tvt_input[:10] = tvt[:10]
    tvt_input[-10:] = tvt[-10:]

    # Future steering response: later trajectory bends after encountering residual.
    x = 1000.0 + 8.0 * md
    y = 2000.0 + 0.2 * md + 3.0 * np.sin(md / 18.0 + phase)
    z = -9000.0 - 0.12 * md - np.roll(residual, 5)
    z[:5] = z[5]
    gr = 80.0 + 0.5 * residual

    pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "TVT": tvt,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    ).to_csv(data_dir / f"{well_id}__horizontal_well.csv", index=False)
