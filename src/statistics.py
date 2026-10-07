"""Reproducible bootstrap confidence intervals for evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def bootstrap_ci(
    y_true: Any,
    y_score: Any,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Return a percentile bootstrap CI for a metric."""
    y = np.asarray(y_true).reshape(-1)
    score = np.asarray(y_score).reshape(-1)
    if len(y) == 0 or len(y) != len(score):
        raise ValueError('Inputs must have equal non-zero length')
    if n_bootstrap < 100:
        raise ValueError('n_bootstrap must be at least 100')
    observed = float(metric_fn(y, score))
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            value = float(metric_fn(y[idx], score[idx]))
        except (ValueError, RuntimeError):
            continue
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        raise RuntimeError('No valid bootstrap replicates were produced')
    alpha = 1.0 - confidence_level
    return {
        'estimate': observed,
        'lower': float(np.quantile(estimates, alpha / 2.0)),
        'upper': float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        'n_bootstrap': n_bootstrap,
        'n_valid': len(estimates),
        'seed': seed,
    }


__all__ = ['bootstrap_ci']