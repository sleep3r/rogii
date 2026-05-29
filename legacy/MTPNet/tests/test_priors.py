from pathlib import Path

import pandas as pd
import pytest

from mtpnet.config import PriorConfig
from mtpnet.priors import load_prior_tables


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_load_prior_tables_merges_base_b2_and_a_sources(tmp_path: Path) -> None:
    base_path = write_csv(
        tmp_path / "base.csv",
        pd.DataFrame({"id": ["a", "b"], "schema10_oof_pp": [10.0, 12.0]}),
    )
    b2_path = write_csv(
        tmp_path / "b2.csv",
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "b2_guarded_submit": [11.0, 13.0],
                "b2_danger_score": [0.2, 0.4],
            }
        ),
    )
    a_path = write_csv(
        tmp_path / "a.csv",
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "formation_sample_median": [9.5, 12.5],
                "formation_sample_p10": [8.0, 11.0],
                "formation_sample_p90": [12.0, 15.0],
            }
        ),
    )

    priors = load_prior_tables(
        PriorConfig(
            enabled=True,
            base_path=base_path,
            b2_path=b2_path,
            a_path=a_path,
            b2_danger_column="b2_danger_score",
        )
    )

    assert priors is not None
    assert list(priors.frame.columns) == [
        "base_tvt",
        "b2_tvt",
        "b2_danger",
        "a_p50_tvt",
        "a_p10_tvt",
        "a_p90_tvt",
    ]
    assert priors.frame.loc["a", "base_tvt"] == pytest.approx(10.0)
    assert priors.frame.loc["b", "b2_tvt"] == pytest.approx(13.0)
    assert priors.frame.loc["a", "a_p10_tvt"] == pytest.approx(8.0)


def test_load_prior_tables_requires_configured_columns(tmp_path: Path) -> None:
    base_path = write_csv(tmp_path / "base.csv", pd.DataFrame({"id": ["a"], "x": [1.0]}))

    with pytest.raises(ValueError, match="schema10_oof_pp"):
        load_prior_tables(PriorConfig(enabled=True, base_path=base_path))
