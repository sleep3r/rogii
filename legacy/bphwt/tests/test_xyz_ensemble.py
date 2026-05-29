from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.test_xyz_steering import _write_synthetic_well


def test_run_xyz_ensemble_oof_writes_artifacts(tmp_path: Path) -> None:
    from scripts import run_xyz_ensemble_oof

    data_dir = tmp_path / "train"
    out_dir = tmp_path / "ensemble_artifacts"
    for i in range(8):
        _write_synthetic_well(data_dir, f"well{i:04d}", phase=float(i) / 3.0)

    summary = run_xyz_ensemble_oof.run(
        SimpleNamespace(
            data_dir=str(data_dir),
            out_dir=str(out_dir),
            limit=0,
            n_splits=4,
            train_stride=2,
            max_train_rows=5000,
            blend_alphas="0.5,1.0",
            clip_residual=80.0,
            no_progress=True,
        )
    )

    experiments = pd.read_csv(out_dir / "experiments.csv")
    per_well = pd.read_csv(out_dir / "per_well.csv")

    assert summary["n_wells"] == 8
    assert {"base", "ens_a0p5", "ens_a1p0"}.issubset(set(experiments["experiment"]))
    assert len(per_well) == 8 * 3
    assert np.isfinite(experiments["rmse"]).all()
