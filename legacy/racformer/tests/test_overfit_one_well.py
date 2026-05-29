from __future__ import annotations

from types import SimpleNamespace

import pytest
from racformer.overfit_one_well import _select_sample


def _sample(well_id: str, n_hidden_rows: int):
    return SimpleNamespace(well_id=well_id, n_hidden_rows=n_hidden_rows)


def test_select_sample_prefers_explicit_well_id() -> None:
    samples = [_sample("well_a", 4), _sample("well_b", 12)]

    selected = _select_sample(samples, well_id="well_a", min_hidden_rows=10)

    assert selected.well_id == "well_a"


def test_select_sample_uses_first_sample_with_enough_hidden_rows() -> None:
    samples = [_sample("well_a", 4), _sample("well_b", 12)]

    selected = _select_sample(samples, well_id=None, min_hidden_rows=10)

    assert selected.well_id == "well_b"


def test_select_sample_raises_when_no_candidate_matches() -> None:
    samples = [_sample("well_a", 4)]

    with pytest.raises(ValueError, match="No sample"):
        _select_sample(samples, well_id=None, min_hidden_rows=10)
