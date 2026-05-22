from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.constants import FORMATIONS
from rogii.formation_plane_knn import (
    FormationPlaneConfig,
    FormationPlaneKNN,
    NearbyPathLibrary,
    build_well_candidates,
)


def _surface_values(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        formation: 0.4 * x - 0.2 * y + 1000.0 - 25.0 * idx
        for idx, formation in enumerate(FORMATIONS)
    }


def _write_well(
    directory: Path,
    well: str,
    *,
    x0: float,
    y0: float,
    hidden_start: int = 40,
    n: int = 70,
) -> Path:
    md = np.arange(n, dtype=float)
    x = x0 + 0.5 * md
    y = y0 + 0.25 * md
    z = -3000.0 - 0.1 * md
    surfaces = _surface_values(x, y)
    b_well = 1200.0
    tvt = -z + surfaces["ANCC"] + b_well
    tvt_input = tvt.copy()
    tvt_input[hidden_start:] = np.nan
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            **surfaces,
            "TVT": tvt,
            "GR": 80.0 + np.sin(md / 5.0),
            "TVT_input": tvt_input,
        }
    )
    path = directory / f"{well}__horizontal_well.csv"
    frame.to_csv(path, index=False)
    return path


def test_formation_plane_knn_recovers_linear_surface(tmp_path: Path) -> None:
    paths = [
        _write_well(tmp_path, "w1", x0=0.0, y0=0.0),
        _write_well(tmp_path, "w2", x0=100.0, y0=0.0),
        _write_well(tmp_path, "w3", x0=0.0, y0=100.0),
        _write_well(tmp_path, "w4", x0=100.0, y0=100.0),
    ]
    cfg = FormationPlaneConfig(sample_rows_per_well=20, min_points=30, bootstrap_samples=0)
    solver = FormationPlaneKNN.from_paths(paths, cfg)
    xy = np.array([[40.0, 60.0], [120.0, 30.0]], dtype=float)

    pred = solver.predict_plane(xy)

    expected = _surface_values(xy[:, 0], xy[:, 1])["ANCC"]
    assert np.allclose(pred.values[:, 0], expected, atol=1e-5)
    assert np.isfinite(pred.dist_min).all()


def test_build_well_candidates_fold_safe_outputs_expected_columns(tmp_path: Path) -> None:
    train_paths = [
        _write_well(tmp_path, "near_a", x0=0.0, y0=0.0),
        _write_well(tmp_path, "near_b", x0=100.0, y0=0.0),
        _write_well(tmp_path, "near_c", x0=0.0, y0=100.0),
    ]
    target = _write_well(tmp_path, "target", x0=30.0, y0=30.0)
    cfg = FormationPlaneConfig(
        k_wells=2,
        sample_rows_per_well=20,
        min_points=20,
        dense_k=20,
        bootstrap_samples=2,
    )
    solver = FormationPlaneKNN.from_paths(train_paths, cfg)
    nearby = NearbyPathLibrary.from_paths(train_paths)

    candidates = build_well_candidates(
        target,
        solver,
        nearby,
        fold_id=1,
        seed=7,
    )

    assert len(candidates) == 30
    assert "tvtF_ANCC_full" in candidates.columns
    assert "nearby_path_weighted_mean" in candidates.columns
    assert "formation_sample_median" in candidates.columns
    assert "anchor_fit_rmse__tvtF_ANCC_full" in candidates.columns
    assert "pseudo_hidden_rmse__tvtF_ANCC_full" in candidates.columns
    assert "b_ANCC_wls" in candidates.columns
    assert np.isfinite(candidates["tvtF_ANCC_full"]).all()
    assert np.isfinite(candidates["S_hat_ANCC"]).all()
    assert float(candidates["anchor_fit_rmse__tvtF_ANCC_full"].iloc[0]) < 1e-2
