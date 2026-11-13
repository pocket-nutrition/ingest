"""Unit tests for the masked multi-output training helpers (no DB, no Redis)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.multioutput import MultiOutputRegressor

from pipeline.train import _fit_masked, _group_yield_factors, _masked_mae

_PARAMS = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.3}


def test_fit_masked_learns_each_target_from_observed_rows_only():
    # Rows with x=1 have target 100 when measured; most of them are unmeasured (NaN).
    # The observed median is 0 (ten rows at x=0), so median imputation would drag the
    # x=1 prediction towards 0. Masking must leave it at the measured 100.
    x = np.array([[0.0]] * 10 + [[1.0]] * 15)
    measured = [0.0] * 10 + [100.0] * 5 + [np.nan] * 10
    y = np.column_stack([np.full(len(x), 5.0), measured])
    weights = np.ones(len(x))

    model = _fit_masked(_PARAMS, x, y, weights)

    assert isinstance(model, MultiOutputRegressor)
    pred = model.predict(np.array([[1.0]]))
    assert pred.shape == (1, 2)
    assert pred[0, 0] == pytest.approx(5.0, abs=0.5)
    assert pred[0, 1] > 90.0


def test_fit_masked_rejects_a_target_without_any_observation():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.column_stack([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])

    with pytest.raises(ValueError, match="target 1"):
        _fit_masked(_PARAMS, x, y, np.ones(3))


def test_masked_mae_ignores_missing_truth():
    y_true = np.array([[1.0, np.nan], [3.0, 4.0]])
    y_pred = np.array([[2.0, 0.0], [3.0, 6.0]])

    assert _masked_mae(y_true, y_pred) == pytest.approx([0.5, 2.0])


def test_masked_mae_is_nan_for_a_target_without_truth():
    y_true = np.array([[1.0, np.nan]])
    y_pred = np.array([[1.0, 9.0]])

    mae = _masked_mae(y_true, y_pred)
    assert mae[0] == 0.0
    assert np.isnan(mae[1])


def _pair(group, method, yield_f, theoretical=False):
    return SimpleNamespace(food_group=group, cooking_method=method, yield_f=yield_f, is_theoretical=theoretical)


def test_group_yield_factors_is_the_median_of_known_yields_per_group_and_method():
    pairs = [
        _pair("meat", "grilled", 0.70), _pair("meat", "grilled", 0.75), _pair("meat", "grilled", 0.80),
        _pair("cereal", "boiled", 2.5), _pair("cereal", "boiled", 2.3), _pair("cereal", "boiled", 2.4),
    ]
    assert _group_yield_factors(pairs) == {
        "meat": {"grilled": pytest.approx(0.75)},
        "cereal": {"boiled": pytest.approx(2.4)},
    }


def test_group_yield_factors_ignores_theoretical_pairs_and_thin_groups():
    # Theoretical pairs already use a group median: counting them would be circular.
    pairs = [
        _pair("fish", "steamed", 0.8), _pair("fish", "steamed", 0.9),        # only 2 known yields
        _pair("fish", "steamed", 0.5, theoretical=True),
        _pair(None, "boiled", 1.0), _pair("dairy", "boiled", None),
    ]
    assert _group_yield_factors(pairs) == {}
