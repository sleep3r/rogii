import numpy as np
import pandas as pd

from mtpnet.config import WindowConfig
from mtpnet.windows import build_windows_for_well, split_wells


def synthetic_horizontal(n: int = 160) -> pd.DataFrame:
    tvt = np.linspace(1000.0, 1040.0, n)
    gr = 80.0 + np.sin(np.linspace(0.0, 8.0, n)) * 20.0
    tvt_input = tvt.copy()
    tvt_input[n // 2 :] = np.nan
    return pd.DataFrame(
        {"id": [f"row_{i}" for i in range(n)], "TVT": tvt, "TVT_input": tvt_input, "GR": gr}
    )


def synthetic_typewell(n: int = 256) -> pd.DataFrame:
    tvt = np.linspace(960.0, 1080.0, n)
    gr = 80.0 + np.sin(np.linspace(0.0, 12.0, n)) * 20.0
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def test_build_windows_shape_and_metadata() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=32,
        vertical_radius_ft=60.0,
        stride_steps=2,
        max_windows_per_well=3,
    )
    windows = build_windows_for_well("well_a", synthetic_horizontal(), synthetic_typewell(), cfg)
    assert len(windows) > 0
    sample = windows[0]
    assert sample.x.shape == (5, 32, 10)
    assert sample.target_bins.shape == (6,)
    assert sample.target_tvt.shape == (6,)
    assert sample.well_id == "well_a"
    assert np.isfinite(sample.x).all()


def test_split_wells_keeps_well_boundaries() -> None:
    train, valid = split_wells(["a", "b", "c", "d", "e"], valid_fraction=0.4, seed=7)
    assert set(train).isdisjoint(valid)
    assert sorted(train + valid) == ["a", "b", "c", "d", "e"]
    assert len(valid) == 2
