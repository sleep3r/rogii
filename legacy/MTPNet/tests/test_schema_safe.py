from __future__ import annotations

from pathlib import Path

import pytest


def test_schema_safe_guard_rejects_train_only_columns() -> None:
    from mtpnet.schema_safe import assert_schema_safe_columns

    with pytest.raises(ValueError, match="forbidden"):
        assert_schema_safe_columns(["MD", "GR", "TVT", "ANCC"], context="unit")


def test_schema_safe_guard_allows_test_available_columns() -> None:
    from mtpnet.schema_safe import assert_schema_safe_columns

    assert_schema_safe_columns(
        ["MD", "X", "Y", "Z", "GR", "TVT_input", "pseudo_zone_prob_EGFDL"],
        context="unit",
    )


def test_pathformer_is_explicitly_diagnostic_when_formation_columns_exist() -> None:
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS
    from pathformer import dataset as pathformer_dataset

    assert set(pathformer_dataset.FORMATION_COLS).issubset(FORBIDDEN_INFERENCE_COLUMNS)
    assert pathformer_dataset.DEPLOYABLE is False
    assert "formation" in pathformer_dataset.DIAGNOSTIC_ONLY_REASON.lower()


def test_genius_model_review_artifact_is_saved() -> None:
    path = Path("artifacts/genius_model_review.md")

    text = path.read_text(encoding="utf-8")

    assert "External Genius Model Review / Next Direction" in text
    assert "raw GR/typewell matching" in text
    assert "schema-safe spatial/geometry residual modeling" in text
