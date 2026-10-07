import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, log_loss

def ece(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        mask = ((probabilities >= edges[i]) & (probabilities <= edges[i+1])) if i == n_bins-1 else ((probabilities >= edges[i]) & (probabilities < edges[i+1]))
        if np.any(mask):
            value += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(value)

def binary_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece": ece(y_true, probabilities),
    }

def confidence(probabilities):
    p = np.asarray(probabilities, dtype=float)
    return np.maximum(p, 1.0 - p)

def predictive_entropy(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0-1e-12)
    return -(p*np.log(p) + (1.0-p)*np.log(1.0-p))

def bootstrap_auroc_ci(y_true, score, n_bootstrap=2000, seed=123):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if np.unique(y_true[idx]).size < 2:
            continue
        values.append(roc_auc_score(y_true[idx], score[idx]))
    if not values:
        return [float("nan")]*3
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(roc_auc_score(y_true, score)), float(lo), float(hi)]

def bootstrap_error_rate_ci(error_indicator, n_bootstrap=2000, seed=123):
    error_indicator = np.asarray(error_indicator).astype(float)
    rng = np.random.default_rng(seed)
    n = len(error_indicator)
    values = [error_indicator[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(error_indicator.mean()), float(lo), float(hi)]
