from __future__ import annotations

import numpy as np

from rogii.modeling import fit_blend_weights, normalize_catboost_bootstrap_params


def test_nnls_blend_weights_are_nonnegative_and_normalized() -> None:
    stack = np.array(
        [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ]
    )
    target = np.array([1.0, 2.0, 3.0])

    weights = fit_blend_weights(stack, target, method="nnls")

    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > 0.99


def test_unknown_blend_method_fails_loudly() -> None:
    stack = np.ones((3, 1))
    target = np.ones(3)

    try:
        fit_blend_weights(stack, target, method="mystery")
    except ValueError as exc:
        assert "Unsupported blend method" in str(exc)
    else:
        raise AssertionError("unknown blend method should raise ValueError")


def test_catboost_bagging_temperature_selects_bayesian_bootstrap() -> None:
    params = {
        "bootstrap_type": "Bernoulli",
        "subsample": 0.9,
        "bagging_temperature": 0.8,
    }

    normalize_catboost_bootstrap_params(
        params,
        explicit_params={"bagging_temperature": 0.8},
    )

    assert params["bootstrap_type"] == "Bayesian"
    assert params["bagging_temperature"] == 0.8
    assert "subsample" not in params


def test_catboost_explicit_bernoulli_drops_bagging_temperature() -> None:
    params = {
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "bagging_temperature": 0.8,
    }

    normalize_catboost_bootstrap_params(
        params,
        explicit_params={"bootstrap_type": "Bernoulli", "bagging_temperature": 0.8},
    )

    assert params["bootstrap_type"] == "Bernoulli"
    assert params["subsample"] == 0.8
    assert "bagging_temperature" not in params
