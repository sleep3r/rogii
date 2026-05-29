from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bphwt.infer.xyz_gate import XYZGateConfig, choose_gate_alpha
from tests.test_xyz_steering import _write_synthetic_well


def test_choose_gate_alpha_falls_back_on_large_or_disagreeing_corrections() -> None:
    hidden = np.array([False, True, True, True, False])
    full = np.array([0.0, 4.0, 5.0, 4.0, 0.0])
    past = np.array([0.0, 3.0, 4.0, 3.0, 0.0])
    cfg = XYZGateConfig(high_alpha=1.0, fallback_alpha=0.25, max_abs_p95=20.0, max_disagreement_rmse=10.0)

    assert choose_gate_alpha(full, past, hidden, cfg) == 1.0
    assert choose_gate_alpha(full * 10.0, past, hidden, cfg) == 0.25
    assert choose_gate_alpha(full, past - 30.0, hidden, cfg) == 0.25


def test_run_xyz_gate_oof_writes_artifacts(tmp_path: Path) -> None:
    from scripts import run_xyz_gate_oof

    data_dir = tmp_path / "train"
    out_dir = tmp_path / "gate_artifacts"
    for i in range(8):
        _write_synthetic_well(data_dir, f"well{i:04d}", phase=float(i) / 3.0)

    summary = run_xyz_gate_oof.run(
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
            fallback_alphas="0,0.5",
            max_abs_p95_values="10,40",
            max_disagreement_values="5,30",
            clip_residual=80.0,
            no_progress=True,
        )
    )

    experiments = pd.read_csv(out_dir / "experiments.csv")
    per_well = pd.read_csv(out_dir / "per_well.csv")

    assert summary["n_wells"] == 8
    assert "base" in set(experiments["experiment"])
    assert "full_a1p0" in set(experiments["experiment"])
    assert any(str(x).startswith("gate_fb") for x in experiments["experiment"])
    assert len(per_well) > 8
    assert np.isfinite(experiments["rmse"]).all()
