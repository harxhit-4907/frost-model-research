"""
baselines.py — Phase 1 baselines: persistence, a physically-motivated
threshold model, and logistic regression. Every model in Phase 3 has to
beat these to justify its complexity (see the project's definition of
done), so they need to actually be run, not stubbed.

Three baselines, three different jobs:

  1. `persistence_predict` — the naive "tomorrow = today" forecast.
     Cheapest possible baseline; if a model can't beat this it isn't
     learning anything.

  2. `PhysicalThresholdBaseline` — an interpretable, physically-motivated
     baseline using ONLY the small set of variables the radiation-frost
     mechanism actually points to: dew-point depression, wind speed
     (where available), and cloud cover, plus the current day's Tmin.
     This is this project's stand-in for the literature's "Franklin/
     Young-style" empirical minimum-temperature formulas. We deliberately
     do NOT hardcode published coefficients (e.g. Young 1920's Oregon/
     California regression, or Kalma's nocturnal-cooling-curve method) —
     those were fit on specific stations and eras, and reproducing them
     from a citation alone without the original calibration data would be
     guessing, not baselining. Instead this fits a small linear model
     restricted to just those physically-motivated inputs, which keeps
     the "physical, interpretable, cheap" character the brief asks for
     while being honest about what's actually validated versus assumed.

  3. `LogisticRegressionBaseline` — plain logistic regression on whatever
     engineered feature set is handed to it (i.e. NOT restricted to the
     physical subset). This is the first baseline that gets to see
     everything features.py produces, and the last one before Phase 3's
     tree ensembles.

Per the brief's core design decision: primary approach for all real models
is regression-then-threshold on continuous Tmin, not direct classification.
`LogisticRegressionBaseline` is deliberately a classifier — it's the
documented class-weighted-classifier comparison point, not the headline
approach. Omazić et al. (2024) found exactly this trade-off with a
class-weighted XGBoost classifier (POD 0.81) versus a plain threshold rule
on continuous temperature (POD 0.98) — expect (and report) something
similar here, it isn't a bug in the pipeline.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


def persistence_predict(current_min_temp: pd.Series) -> pd.Series:
    """Predicted future min temp = current min temp. No fitting involved."""
    return current_min_temp.copy()


class PhysicalThresholdBaseline:
    """
    Linear regression restricted to physically-motivated inputs:
    current Tmin, dew-point depression, cloud cover, and wind speed (if
    supplied). Predicts continuous future Tmin; threshold afterward with
    label_rules for a frost/no-frost call, consistent with this project's
    regression-then-threshold design decision.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.feature_columns_: Optional[List[str]] = None

    @staticmethod
    def _select_columns(df: pd.DataFrame) -> List[str]:
        """
        A candidate column is used only if it's both present AND fully
        populated (no NaN) in `df`. The "present" check alone isn't enough
        once `df` can span multiple cities: pd.concat-ing cities with
        different available columns (e.g. Basel has no wind_speed) fills
        the gaps with NaN rather than dropping the column, and
        LinearRegression raises on NaN input rather than silently ignoring
        it. Checking "no NaN" here means a multi-city table automatically
        (and correctly) falls back to whatever subset of physical features
        every included city actually has, instead of crashing.
        """
        candidates = ["temp_min", "dew_point_depression_c", "cloud_cover", "wind_speed"]
        cols = [c for c in candidates if c in df.columns and df[c].notna().all()]
        if "temp_min" not in cols:
            raise ValueError(
                "PhysicalThresholdBaseline requires a fully-populated 'temp_min' column."
            )
        return cols

    def fit(self, df: pd.DataFrame, target: pd.Series) -> "PhysicalThresholdBaseline":
        self.feature_columns_ = self._select_columns(df)
        X = df[self.feature_columns_]
        self.model.fit(X, target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_columns_ is None:
            raise RuntimeError("Call .fit() before .predict().")
        X = df[self.feature_columns_]
        return self.model.predict(X)


class LogisticRegressionBaseline:
    """
    Plain logistic regression classifier on an arbitrary feature matrix.
    `class_weight="balanced"` by default, mirroring the class-weighted
    comparison Omazić et al. (2024) ran for their own classifier ablation
    — report this baseline's recall against the regression-then-threshold
    approach's recall explicitly, the gap is the point.
    """

    def __init__(self, class_weight: str = "balanced", max_iter: int = 1000):
        self.model = LogisticRegression(class_weight=class_weight, max_iter=max_iter)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


RUNS_CSV_COLUMNS = [
    "timestamp",
    "model",
    "params",
    "dataset",
    "precision",
    "recall",
    "f1",
    "peirce_skill",
    "rmse",
    "mae",
]


def log_run(
    csv_path: str,
    model: str,
    dataset: str,
    metrics: Dict[str, float],
    params: str = "None",
) -> None:
    """
    Append one row to experiments/runs.csv in the project's established
    schema. Centralizing this here (instead of re-writing the same csv
    logic inline in every notebook, which is how the first baseline was
    logged) means every run — baseline or model — is logged identically.

    `metrics` should contain whichever of {precision, recall, f1,
    peirce_skill, rmse, mae} are available; missing keys are logged blank
    rather than raising, since not every model produces both continuous
    and classification metrics.
    """
    path = Path(csv_path)
    is_new = not path.exists()

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model,
        "params": params,
        "dataset": dataset,
    }
    for key in ("precision", "recall", "f1", "peirce_skill", "rmse", "mae"):
        val = metrics.get(key)
        row[key] = round(val, 3) if val is not None else ""

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
