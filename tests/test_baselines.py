import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baselines import (  # noqa: E402
    RUNS_CSV_COLUMNS,
    PhysicalThresholdBaseline,
    log_run,
    persistence_predict,
)


def test_persistence_predict_is_identity():
    series = pd.Series([1.0, 2.0, 3.0])
    result = persistence_predict(series)
    assert list(result) == [1.0, 2.0, 3.0]
    # Should be a copy, not the same object, so callers can't mutate the input by accident.
    assert result is not series


def test_log_run_writes_header_once_and_matches_schema(tmp_path):
    csv_path = tmp_path / "runs.csv"

    log_run(
        str(csv_path),
        model="Persistence",
        dataset="Basel_Tier1",
        metrics={"precision": 0.837, "recall": 0.837, "f1": 0.837,
                 "peirce_skill": 0.768, "rmse": 2.352, "mae": 1.799},
    )
    log_run(
        str(csv_path),
        model="LogisticRegression",
        dataset="Basel_Tier1",
        metrics={"precision": 0.6, "recall": 0.5, "f1": 0.55, "peirce_skill": 0.4},
    )

    df = pd.read_csv(csv_path)
    assert list(df.columns) == RUNS_CSV_COLUMNS
    assert len(df) == 2
    # Second run had no rmse/mae -> should be blank, not a fabricated 0.
    assert pd.isna(df.loc[1, "rmse"])


def test_physical_threshold_baseline_drops_partially_populated_columns():
    """
    Regression test: a multi-city table where 'wind_speed' exists for one
    city but is NaN for another (exactly what pd.concat produces when
    cities have different available sensors) used to crash LinearRegression
    with "Input X contains NaN". It should instead silently fall back to
    the columns that ARE fully populated, not raise.
    """
    df = pd.DataFrame(
        {
            "temp_min": [5.0, 2.0, -1.0, 8.0, 3.0, 1.0],
            "dew_point_depression_c": [1.0, 2.0, 0.5, 1.5, 2.0, 0.8],
            "wind_speed": [3.0, 4.0, np.nan, np.nan, np.nan, np.nan],  # only city A has wind
        }
    )
    target = pd.Series([2.0, -1.0, 8.0, 3.0, 1.0, 0.0])

    model = PhysicalThresholdBaseline().fit(df, target)

    assert "wind_speed" not in model.feature_columns_
    assert "temp_min" in model.feature_columns_
    assert "dew_point_depression_c" in model.feature_columns_

    preds = model.predict(df)
    assert len(preds) == len(df)
