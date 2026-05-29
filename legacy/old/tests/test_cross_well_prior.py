from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cross_well_prior import (
    compute_signature,
    collect_train_paths,
    cross_well_typewell_path,
    nearest_train_wells,
    signature_to_vector,
    TrainWellPath,
)
from rogii.direct_solver import fit_geo_candidate


def _make_train_well(
    *,
    well: str,
    last_md: float,
    last_z: float,
    last_tvt: float,
    md_span: float = 400.0,
    z_drift: float = 5.0,
    tvt_slope: float = 0.07,
    xy_x: float = 0.0,
    xy_y: float = 0.0,
    n_points: int = 200,
) -> TrainWellPath:
    md = last_md + np.linspace(-md_span * 0.5, md_span * 0.5, n_points)
    z = last_z + np.linspace(-z_drift, z_drift, n_points)
    tvt = last_tvt + tvt_slope * (md - last_md)
    signature = {
        "xy_x": float(xy_x),
        "xy_y": float(xy_y),
        "last_known_z": float(last_z),
        "last_known_tvt": float(last_tvt),
        "gr_mean_hidden": 80.0,
        "gr_std_hidden": 3.0,
        "surf_ANCC": float(last_z - 10.0),
        "surf_ASTNU": float(last_z - 20.0),
        "surf_ASTNL": float(last_z - 30.0),
        "surf_EGFDU": float(last_z - 40.0),
        "surf_EGFDL": float(last_z - 50.0),
        "surf_BUDA": float(last_z - 60.0),
    }
    return TrainWellPath(
        well=well,
        md=md.astype(float),
        z=z.astype(float),
        tvt=tvt.astype(float),
        last_known_md=float(last_md),
        last_known_tvt=float(last_tvt),
        last_known_z=float(last_z),
        signature=signature,
    )


def test_signature_to_vector_returns_finite_for_complete_input() -> None:
    sig = {
        "xy_x": 1.0,
        "xy_y": 2.0,
        "last_known_z": 100.0,
        "last_known_tvt": 1100.0,
        "gr_mean_hidden": 80.0,
        "gr_std_hidden": 5.0,
        "surf_ANCC": 90.0,
        "surf_ASTNU": 80.0,
        "surf_ASTNL": 70.0,
        "surf_EGFDU": 60.0,
        "surf_EGFDL": 50.0,
        "surf_BUDA": 40.0,
    }
    vector = signature_to_vector(sig)
    assert vector.shape == (12,)
    assert np.isfinite(vector).all()


def test_nearest_train_wells_orders_by_signature_distance() -> None:
    target = _make_train_well(
        well="target", last_md=1000.0, last_z=2400.0, last_tvt=1200.0,
        xy_x=10.0, xy_y=10.0,
    )
    near = _make_train_well(
        well="near", last_md=1010.0, last_z=2405.0, last_tvt=1202.0,
        xy_x=11.0, xy_y=11.0,
    )
    far = _make_train_well(
        well="far", last_md=2000.0, last_z=3000.0, last_tvt=2000.0,
        xy_x=500.0, xy_y=500.0,
    )
    chosen, dists = nearest_train_wells(
        target.signature, [near, far], k=2, self_well="target",
    )
    assert [w.well for w in chosen] == ["near", "far"]
    assert dists[0] < dists[1]


def test_cross_well_typewell_path_aggregates_neighbors() -> None:
    base = _make_train_well(
        well="base", last_md=1000.0, last_z=2400.0, last_tvt=1200.0,
        tvt_slope=0.07,
    )
    near_a = _make_train_well(
        well="near_a", last_md=1005.0, last_z=2402.0, last_tvt=1202.0,
        tvt_slope=0.075, xy_x=2.0, xy_y=2.0,
    )
    near_b = _make_train_well(
        well="near_b", last_md=995.0, last_z=2398.0, last_tvt=1198.0,
        tvt_slope=0.069, xy_x=-2.0, xy_y=2.0,
    )
    far = _make_train_well(
        well="far", last_md=5000.0, last_z=5000.0, last_tvt=2500.0,
        tvt_slope=0.15, xy_x=500.0, xy_y=500.0,
    )

    n = 100
    md = base.last_known_md + np.linspace(0.0, 200.0, n)
    z = base.last_known_z + np.linspace(0.0, 4.0, n)
    hidden = np.arange(40, n, dtype=int)
    md_path, z_path, diag = cross_well_typewell_path(
        test_md=md,
        test_z=z,
        hidden_indices=hidden,
        last_idx=39,
        last_tvt=base.last_known_tvt,
        last_md=base.last_known_md,
        last_z=base.last_known_z,
        test_signature=base.signature,
        train_paths=[near_a, near_b, far],
        k=2,
        self_well="base",
    )
    assert md_path.shape == (n,)
    assert z_path.shape == (n,)
    assert np.isfinite(md_path[hidden]).all()
    assert np.isfinite(z_path[hidden]).all()
    assert int(diag["crosswell_k"]) == 2
    assert "near_a" in diag["crosswell_neighbors"]
    assert "near_b" in diag["crosswell_neighbors"]
    assert "far" not in diag["crosswell_neighbors"]


def test_cross_well_typewell_falls_back_when_no_train_paths() -> None:
    n = 30
    md = np.linspace(0.0, 100.0, n)
    z = np.linspace(2400.0, 2410.0, n)
    hidden = np.arange(15, n, dtype=int)
    md_path, z_path, diag = cross_well_typewell_path(
        test_md=md,
        test_z=z,
        hidden_indices=hidden,
        last_idx=14,
        last_tvt=1200.0,
        last_md=md[14],
        last_z=z[14],
        test_signature={"xy_x": 0.0, "xy_y": 0.0, "last_known_z": z[14], "last_known_tvt": 1200.0, "gr_mean_hidden": 80.0, "gr_std_hidden": 3.0},
        train_paths=[],
        k=4,
        self_well="unknown",
    )
    assert diag["crosswell_k"] == 0.0
    assert np.allclose(md_path, 1200.0)
    assert np.allclose(z_path, 1200.0)


def test_collect_train_paths_reads_train_directory(tmp_path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    n = 90
    md = np.arange(n, dtype=float)
    tvt = 1100.0 + 0.05 * md
    tvt_input = tvt.copy()
    tvt_input[70:] = np.nan
    df = pd.DataFrame(
        {
            "MD": md,
            "X": md * 0.1,
            "Y": md * 0.2,
            "Z": 2500.0 - 0.02 * md,
            "GR": 80.0 + np.sin(md / 6.0),
            "TVT_input": tvt_input,
            "TVT": tvt,
            "ANCC": 2500.0 - 0.02 * md - (tvt - 1000.0),
        }
    )
    df.to_csv(train_dir / "abcd1234__horizontal_well.csv", index=False)
    paths = collect_train_paths(tmp_path)
    assert len(paths) == 1
    assert paths[0].well == "abcd1234"
    assert paths[0].tvt.size > 0
    assert np.isfinite(paths[0].last_known_tvt)


def test_fit_geo_candidate_returns_consensus_path() -> None:
    rng = np.random.default_rng(3)
    n = 160
    md = np.arange(n, dtype=float)
    z = 2400.0 - 0.04 * md
    tvt = 1100.0 + 0.06 * md
    tvt_input = tvt.copy()
    tvt_input[120:] = np.nan
    gr = 80.0 + np.sin(md / 7.0) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "MD": md,
            "X": md * 0.05,
            "Y": md * 0.07,
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": tvt,
            "ANCC": z - (tvt - 1090.0),
            "ASTNU": z - (tvt - 1080.0),
            "ASTNL": z - (tvt - 1070.0),
            "EGFDU": z - (tvt - 1060.0),
            "EGFDL": z - (tvt - 1050.0),
            "BUDA": z - (tvt - 1040.0),
        }
    )
    hidden = np.arange(120, n, dtype=int)
    best, consensus, diag = fit_geo_candidate(
        df, md, z, tvt_input, hidden, 119, float(tvt_input[119]), tail_rows=60,
    )
    assert best.shape == (n,)
    assert consensus.shape == (n,)
    assert np.isfinite(best[hidden]).all()
    assert np.isfinite(consensus[hidden]).all()
    assert diag["geo_consensus_surfaces"] >= 1.0
    # With perfectly synthetic data all surfaces produce identical fits and
    # consensus can collapse onto best_path; the diagnostic just needs to
    # surface that we evaluated more than one surface.
    assert diag["geo_consensus_surfaces"] >= 2.0


def test_compute_signature_handles_missing_columns() -> None:
    n = 40
    md = np.arange(n, dtype=float)
    tvt_input = np.linspace(100.0, 110.0, n)
    tvt_input[20:] = np.nan
    df = pd.DataFrame(
        {
            "MD": md,
            "Z": 200.0 - 0.5 * md,
            "GR": 80.0 + np.sin(md / 5.0),
            "TVT_input": tvt_input,
        }
    )
    hidden = np.arange(20, n, dtype=int)
    sig = compute_signature(df, last_idx=19, hidden_indices=hidden)
    assert "xy_x" in sig and not np.isfinite(sig["xy_x"])
    assert np.isfinite(sig["gr_mean_hidden"])
    assert np.isfinite(sig["last_known_tvt"])
