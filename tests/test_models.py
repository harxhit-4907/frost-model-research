import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (  # noqa: E402
    catboost_model,
    lightgbm_model,
    random_forest_model,
    xgboost_model,
)


def _toy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "temp_min": rng.uniform(-5, 15, n),
            "humidity_pct": rng.uniform(40, 100, n),
        }
    )
    # A simple, learnable relationship: tomorrow's min tracks today's min
    # minus a humidity-dependent offset, plus a little noise.
    y = X["temp_min"] - 0.02 * X["humidity_pct"] + rng.normal(0, 0.1, n)
    return X, y


def test_random_forest_model_fits_and_predicts():
    X, y = _toy_data()
    model = random_forest_model(n_estimators=20).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert model.feature_columns_ == ["temp_min", "humidity_pct"]


def test_random_forest_feature_importances_sum_to_one_and_are_sorted():
    X, y = _toy_data()
    model = random_forest_model(n_estimators=20).fit(X, y)
    importances = model.feature_importances()
    assert abs(importances.sum() - 1.0) < 1e-6
    assert list(importances.index) == list(importances.sort_values(ascending=False).index)


def test_predict_uses_recorded_feature_columns_even_with_extra_columns_present():
    X, y = _toy_data()
    model = random_forest_model(n_estimators=20).fit(X, y)

    X_with_extra = X.copy()
    X_with_extra["unrelated_column"] = 999
    # Should not raise, and should ignore the extra column.
    preds = model.predict(X_with_extra)
    assert len(preds) == len(X_with_extra)


def test_predict_before_fit_raises():
    X, _ = _toy_data()
    model = random_forest_model()
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_xgboost_model_fits_and_predicts():
    pytest.importorskip("xgboost")
    X, y = _toy_data()
    model = xgboost_model(n_estimators=20).fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_lightgbm_model_fits_and_predicts():
    pytest.importorskip("lightgbm")
    X, y = _toy_data()
    model = lightgbm_model(n_estimators=20).fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_catboost_model_fits_and_predicts():
    pytest.importorskip("catboost")
    X, y = _toy_data()
    model = catboost_model(n_estimators=20).fit(X, y)
    assert len(model.predict(X)) == len(X)
