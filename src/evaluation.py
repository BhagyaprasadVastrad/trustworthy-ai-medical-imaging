"""Shared evaluation metrics for the medical-imaging experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1).astype(int)
    p = np.asarray(y_prob).reshape(-1).astype(float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError('y_true and y_prob must have equal non-zero length')
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError('y_prob must be in [0, 1]')
    return y, p


def expected_calibration_error(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """ECE using the project's historical predicted-class-confidence definition."""
    y, p = _validate(y_true, y_prob)
    pred = (p >= 0.5).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & ((confidence <= hi) if i == n_bins - 1 else (confidence < hi))
        if np.any(mask):
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def maximum_calibration_error(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """Maximum absolute confidence/accuracy gap over non-empty bins."""
    y, p = _validate(y_true, y_prob)
    pred = (p >= 0.5).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & ((confidence <= hi) if i == n_bins - 1 else (confidence < hi))
        if np.any(mask):
            gaps.append(abs(confidence[mask].mean() - correct[mask].mean()))
    return float(max(gaps)) if gaps else float('nan')


def evaluate_binary_classifier(y_true: Any, y_prob: Any, threshold: float = 0.5, ece_bins: int = 10) -> dict[str, float]:
    """Return the common binary-classification metric set used by future experiments."""
    y, p = _validate(y_true, y_prob)
    pred = (p >= threshold).astype(int)
    result = {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, np.column_stack((1.0 - p, p)), labels=[0, 1])),
        'ece': expected_calibration_error(y, p, ece_bins),
        'mce': maximum_calibration_error(y, p, ece_bins),
        'auroc': float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float('nan'),
        'auprc': float(average_precision_score(y, p)) if np.unique(y).size == 2 else float('nan'),
    }
    return result


__all__ = ['evaluate_binary_classifier', 'expected_calibration_error', 'maximum_calibration_error']