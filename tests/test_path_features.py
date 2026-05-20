from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.path_features import (
    DEFAULT_PATH_FEATURE_NAMES,
    build_direct_path_features,
    empty_path_features,
)


def _synthetic_well(n: int = 160, hidden_start: int = 110, seed: int = 5):
    rng = np.random.default_rng(seed)
    md = np.arange(n, dtype=float)
    z = 2400.0 - 0.04 * md
    tvt = 1100.0 + 0.07 * md
    tvt_input = tvt.copy()
    tvt_input[hidden_start:] = np.nan
    gr = 80.0 + 12.0 * np.sin(tvt / 18.0) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "MD": md,
            "X": md * 0.1,
            "Y": md * 0.2,
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": tvt,
            "ANCC": z - (tvt - 1090.0),
            "ASTNU": z - (tvt - 1070.0),
            "ASTNL": z - (tvt - 1050.0),
            "EGFDU": z - (tvt - 1030.0),
            "EGFDL": z - (tvt - 1010.0),
            "BUDA": z - (tvt - 990.0),
        }
    )
    return df, md, z, gr, tvt_input


def _path_features_config():
    return {
        "features": {
            "direct_path": {
                "tail_rows": 60,
                "cem_n_iter": 2,
                "cem_pop_size": 40,
                "cem_top_k": 3,
                "stage2_n_knots": 4,
                "stage2_max_offset": 6.0,
                "stage2_passes": 1,
            },
        }
    }


def test_empty_path_features_have_expected_names() -> None:
    feats = empty_path_features(10)
    assert set(feats) == set(DEFAULT_PATH_FEATURE_NAMES)
    for arr in feats.values():
        assert arr.shape == (10,)


def test_build_direct_path_features_produces_finite_hidden_values(tmp_path) -> None:
    df, md, z, gr, tvt_input = _synthetic_well()
    n = len(df)
    horizontal_path = tmp_path / "abc123__horizontal_well.csv"
    df.to_csv(horizontal_path, index=False)
    flat_pred = np.full(n, float(np.nanmean(tvt_input[np.isfinite(tvt_input)])))
    feats = build_direct_path_features(
        df, horizontal_path,
        md, df["X"].to_numpy(float), df["Y"].to_numpy(float),
        z, gr, tvt_input, flat_pred,
        _path_features_config(), None,
    )
    hidden_idx = np.flatnonzero(np.isnan(tvt_input))
    assert set(feats) == set(DEFAULT_PATH_FEATURE_NAMES)
    # These features depend on typewell availability and are NaN if no
    # __typewell.csv exists next to the horizontal_well.csv.
    typewell_dependent = {"kg_path_gr_cal_rmse", "kg_path_gr_cal_a", "kg_path_gr_cal_b"}
    for name, arr in feats.items():
        assert arr.shape == (n,), name
        if name in typewell_dependent:
            continue
        finite_hidden = np.isfinite(arr[hidden_idx])
        assert finite_hidden.any(), f"{name}: no finite hidden value"
    # Anchor-relative deltas should be near zero where geo_consensus is close
    # to last known tvt on this near-linear synthetic well.
    assert (
        abs(feats["kg_path_geo_consensus_minus_last"][hidden_idx[0]])
        < 5.0
    )


def test_build_direct_path_features_returns_empty_when_no_known(tmp_path) -> None:
    df, md, z, gr, tvt_input = _synthetic_well()
    tvt_input = np.full_like(tvt_input, np.nan)
    n = len(df)
    horizontal_path = tmp_path / "abc123__horizontal_well.csv"
    df.to_csv(horizontal_path, index=False)
    flat_pred = np.zeros(n, dtype=float)
    feats = build_direct_path_features(
        df, horizontal_path,
        md, df["X"].to_numpy(float), df["Y"].to_numpy(float),
        z, gr, tvt_input, flat_pred,
        _path_features_config(), None,
    )
    # Every column stays at NaN because the solver had nothing to fit on.
    for arr in feats.values():
        assert arr.shape == (n,)
        assert not np.isfinite(arr).any()
