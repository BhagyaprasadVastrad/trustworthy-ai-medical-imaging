"""Metrics used by the reproducible reliability analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return discrimination and probability-quality metrics for binary predictions."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have the same shape")
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece": float(expected_calibration_error(y_true, probabilities)),
    }


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    """Compute equal-width expected calibration error."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have the same shape")
    if len(y_true) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        mask = (probabilities >= left) & (
            probabilities <= right if i == n_bins - 1 else probabilities < right
        )
        if np.any(mask):
            ece += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(ece)


def bootstrap_metric_ci(y_true: np.ndarray, probabilities: np.ndarray, metric: str = "auroc",
                        n_bootstrap: int = 2000, seed: int = 42,
                        confidence: float = 0.95) -> tuple[float, float, float]:
    """Return point estimate and percentile bootstrap confidence interval."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities) or len(y_true) < 2:
        raise ValueError("inputs must have equal length >= 2")

    metric_fn = {
        "auroc": lambda y, p: roc_auc_score(y, p),
        "average_precision": lambda y, p: average_precision_score(y, p),
        "brier": lambda y, p: brier_score_loss(y, p),
        "log_loss": lambda y, p: log_loss(y, p, labels=[0, 1]),
        "ece": lambda y, p: expected_calibration_error(y, p),
    }.get(metric)
    if metric_fn is None:
        raise ValueError(f"Unsupported metric: {metric}")

    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, probabilities))
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        y = y_true[idx]
        if metric in {"auroc", "average_precision"} and np.unique(y).size < 2:
            continue
        values.append(float(metric_fn(y, probabilities[idx])))

    if not values:
        raise RuntimeError("No valid bootstrap resamples were produced")
    alpha = 1.0 - confidence
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
