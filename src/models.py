"""
models.py — Phase 3 models: Random Forest, XGBoost, LightGBM, CatBoost.

All four are wrapped as REGRESSORS predicting continuous minimum
temperature, thresholded downstream to a frost/no-frost call. This
matches the project's core design decision (see baselines.py): direct
classification on the rare frost label loses recall relative to
regression-then-threshold, so that stays the primary approach here too,
not just for the Phase 1 baselines.

TreeModelWrapper gives all four the same fit/predict/feature_importances
interface, so 05_model_training.ipynb can loop over them uniformly
instead of hand-writing four slightly different training blocks.

Each library is imported defensively (falls back to None if not
installed) so importing this module doesn't hard-crash an environment
that only has some of the four installed -- the corresponding factory
function raises a clear ImportError only when actually called.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


class TreeModelWrapper:
    """
    Common fit/predict/feature_importances wrapper around any sklearn-API
    regressor. `feature_columns_` is recorded at fit time (mirrors
    PhysicalThresholdBaseline's pattern in baselines.py) so predict()
    always uses exactly the columns the model was trained on, in the
    same order, regardless of what extra columns the caller's DataFrame
    happens to carry.
    """

    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name
        self.feature_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeModelWrapper":
        self.feature_columns_ = list(X.columns)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_columns_ is None:
            raise RuntimeError("Call .fit() before .predict().")
        return self.estimator.predict(X[self.feature_columns_])

    def feature_importances(self) -> pd.Series:
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            raise AttributeError(f"{type(self.estimator).__name__} has no feature_importances_")
        return pd.Series(importances, index=self.feature_columns_).sort_values(ascending=False)


def random_forest_model(**kwargs) -> TreeModelWrapper:
    defaults = dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    defaults.update(kwargs)
    return TreeModelWrapper(RandomForestRegressor(**defaults), name="RandomForest")


def xgboost_model(**kwargs) -> TreeModelWrapper:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. pip install xgboost")
    defaults = dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    defaults.update(kwargs)
    return TreeModelWrapper(XGBRegressor(**defaults), name="XGBoost")


def lightgbm_model(**kwargs) -> TreeModelWrapper:
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed. pip install lightgbm")
    defaults = dict(
        n_estimators=300, max_depth=-1, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    defaults.update(kwargs)
    return TreeModelWrapper(LGBMRegressor(**defaults), name="LightGBM")


def catboost_model(**kwargs) -> TreeModelWrapper:
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed. pip install catboost")
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=False)
    defaults.update(kwargs)
    return TreeModelWrapper(CatBoostRegressor(**defaults), name="CatBoost")
