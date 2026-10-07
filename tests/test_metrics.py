import numpy as np

from src.metrics import binary_metrics, expected_calibration_error


def test_ece_perfect_predictions_are_zero():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0


def test_binary_metrics_contains_core_metrics():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    result = binary_metrics(y, p)
    assert result["accuracy"] == 1.0
    assert result["auroc"] == 1.0
    assert result["brier"] < 0.05
