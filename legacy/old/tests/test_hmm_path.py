from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.constants import FORMATIONS
from rogii.hmm_path import HMM_FEATURE_COLUMNS, build_hmm_path_features


def _write_hmm_synthetic_well(tmp_path, name: str = "hmm12345"):
    n = 36
    idx = np.arange(n, dtype=float)
    tvt = 100.0 + 0.65 * idx
    known = idx < 22
    tvt_input = np.where(known, tvt, np.nan)
    z = 850.0 + 0.08 * idx
    x = 1000.0 + 4.0 * idx
    y = 2000.0 + 1.5 * idx
    gr = 70.0 + 9.0 * np.sin((tvt - 95.0) / 7.0)
    frame = pd.DataFrame(
        {
            "MD": idx,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )
    for offset, formation in enumerate(FORMATIONS):
        frame[formation] = z + tvt + 8.0 * offset
    horizontal = tmp_path / f"{name}__horizontal_well.csv"
    frame.to_csv(horizontal, index=False)

    tw_tvt = np.linspace(90.0, 130.0, 161)
    typewell = pd.DataFrame(
        {
            "TVT": tw_tvt,
            "GR": 70.0 + 9.0 * np.sin((tw_tvt - 95.0) / 7.0),
        }
    )
    typewell.to_csv(tmp_path / f"{name}__typewell.csv", index=False)
    return horizontal, frame


def _hmm_config() -> dict:
    return {
        "features": {
            "kaggle_top": {
                "hmm_enabled": True,
                "hmm_max_states": 64,
                "hmm_max_step_states": 6,
                "hmm_state_pad": 20.0,
                "hmm_gr_weight": 1.0,
                "hmm_prior_weight": 0.2,
                "hmm_geo_weight": 0.1,
            }
        }
    }


def test_hmm_path_features_are_finite_on_hidden_rows(tmp_path) -> None:
    path, frame = _write_hmm_synthetic_well(tmp_path)
    n = len(frame)
    flat_pred = np.full(n, frame.loc[frame["TVT_input"].notna(), "TVT_input"].iloc[-1])
    candidate_features = {
        "last_known_tvt": flat_pred,
        "kg_pf_ancc_tvt": frame["TVT"].to_numpy(dtype=float) + 0.5,
        "kg_dtw_r50_tvt": frame["TVT"].to_numpy(dtype=float) - 0.5,
    }

    features = build_hmm_path_features(
        horizontal_df=frame,
        horizontal_path=path,
        md=frame["MD"].to_numpy(dtype=float),
        z=frame["Z"].to_numpy(dtype=float),
        gr=frame["GR"].to_numpy(dtype=float),
        tvt_input=frame["TVT_input"].to_numpy(dtype=float),
        flat_pred=flat_pred,
        candidate_features=candidate_features,
        config=_hmm_config(),
    )

    hidden = frame["TVT_input"].isna().to_numpy()
    for column in HMM_FEATURE_COLUMNS:
        assert column in features
        assert len(features[column]) == n
    assert np.isfinite(features["kg_hmm_tvt"][hidden]).all()
    assert np.isfinite(features["kg_hmm_path_cost"][hidden]).all()
    assert np.isfinite(features["kg_hmm_minus_flat"][hidden]).all()


def test_hmm_path_features_keep_schema_as_nan_without_typewell(tmp_path) -> None:
    path, frame = _write_hmm_synthetic_well(tmp_path, name="notype12")
    (tmp_path / "notype12__typewell.csv").unlink()
    n = len(frame)
    flat_pred = np.full(n, 100.0)

    features = build_hmm_path_features(
        horizontal_df=frame,
        horizontal_path=path,
        md=frame["MD"].to_numpy(dtype=float),
        z=frame["Z"].to_numpy(dtype=float),
        gr=frame["GR"].to_numpy(dtype=float),
        tvt_input=frame["TVT_input"].to_numpy(dtype=float),
        flat_pred=flat_pred,
        candidate_features={"last_known_tvt": flat_pred},
        config=_hmm_config(),
    )

    for column in HMM_FEATURE_COLUMNS:
        assert column in features
        assert np.isnan(features[column]).all()
