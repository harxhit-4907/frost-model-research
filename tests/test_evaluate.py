import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import get_classification_metrics, get_continuous_metrics  # noqa: E402


def test_get_continuous_metrics_hand_computed():
    # Errors are [1, 0, 1, 0] -> MAE = 0.5, RMSE = sqrt((1+0+1+0)/4) = sqrt(0.5)
    y_true = [0.0, 2.0, -1.0, 5.0]
    y_pred = [1.0, 2.0, 0.0, 5.0]
    metrics = get_continuous_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.5
    assert abs(metrics["rmse"] - 0.5 ** 0.5) < 1e-9


def test_get_classification_metrics_hand_computed():
    # Confusion matrix by hand:
    #   idx: 0  1  2  3  4  5  6  7
    #   y_true: 1  0  1  1  0  0  1  0
    #   y_pred: 1  0  0  1  0  1  1  0
    # TP=3 (idx 0,3,6), FN=1 (idx 2), FP=1 (idx 5), TN=3 (idx 1,4,7)
    # precision = TP/(TP+FP) = 3/4 = 0.75
    # recall    = TP/(TP+FN) = 3/4 = 0.75
    # f1        = 0.75
    # TPR = 0.75, FPR = FP/(FP+TN) = 1/4 = 0.25 -> Peirce = 0.75 - 0.25 = 0.5
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
    metrics = get_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1"] == 0.75
    assert abs(metrics["peirce_skill"] - 0.5) < 1e-9


def test_get_classification_metrics_perfect_predictions():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    metrics = get_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["peirce_skill"] == 1.0


def test_get_classification_metrics_handles_all_negative_batch():
    # Regression test: confusion_matrix(y_true, y_pred) without labels=[0,1]
    # returns a 1x1 matrix when only one class appears in the combined
    # y_true/y_pred, and .ravel() then crashes trying to unpack 4 values.
    # This is a real case, not a hypothetical one -- a small fold or a
    # single city/date slice can easily have zero frost days.
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    metrics = get_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["peirce_skill"] == 0.0


def test_get_classification_metrics_handles_all_positive_batch():
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    metrics = get_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["peirce_skill"] == 1.0


def test_get_classification_metrics_handles_no_positive_predictions():
    # zero_division=0 should keep this from raising, and precision/recall
    # of an always-negative classifier on some positives should be 0.
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    metrics = get_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
