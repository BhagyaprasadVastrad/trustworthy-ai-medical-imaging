"""Post-hoc temperature scaling for binary classifiers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply binary temperature scaling to probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = _logit(probabilities) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def fit_temperature(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit a single positive temperature by minimising binary log loss."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or len(y) == 0:
        raise ValueError("y_true and probabilities must be non-empty and have equal shape")
    if np.unique(y).size < 2:
        raise ValueError("temperature fitting requires both classes")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        q = temperature_scale(p, temperature)
        eps = 1e-7
        return float(-np.mean(
            y * np.log(np.clip(q, eps, 1 - eps))
            + (1 - y) * np.log(np.clip(1 - q, eps, 1 - eps))
        ))

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimisation failed: {result.message}")
    return float(np.exp(result.x))
