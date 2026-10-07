"""Stage 5 uncertainty analysis from identifier-aligned ensemble predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .alignment import align_by_study_id
from .metrics import binary_metrics, bootstrap_metric_ci


def analyse_ensemble(prediction_files: list[Path], labels_file: Path) -> dict[str, object]:
    """Compare ensemble uncertainty signals against observed errors."""
    if len(prediction_files) < 2:
        raise ValueError("At least two prediction files are required")

    labels = pd.read_csv(labels_file)
    merged = labels[["StudyInstanceUID", "label"]].copy()

    for path in prediction_files:
        frame = pd.read_csv(path)
        aligned = align_by_study_id(frame, labels)
        aligned = aligned[["StudyInstanceUID", "probability"]].rename(
            columns={"probability": path.stem}
        )
        merged = merged.merge(
            aligned, on="StudyInstanceUID", how="inner", validate="one_to_one"
        )

    prediction_cols = [c for c in merged.columns if c not in {"StudyInstanceUID", "label"}]
    matrix = merged[prediction_cols].to_numpy(dtype=float)
    y = merged["label"].to_numpy(dtype=int)

    mean_probability = matrix.mean(axis=1)
    disagreement = matrix.std(axis=1, ddof=0)
    max_probability = matrix.max(axis=1)

    p = np.clip(mean_probability, 1e-7, 1 - 1e-7)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    ensemble_metrics = binary_metrics(y, mean_probability)
    predicted = (mean_probability >= 0.5).astype(int)
    error = (predicted != y).astype(int)

    scores = {
        "ensemble_disagreement": disagreement,
        "one_minus_max_probability": 1.0 - max_probability,
        "predictive_entropy": entropy,
    }
    uncertainty_comparison = {}
    for name, score in scores.items():
        uncertainty_comparison[name] = {
            "error_detection_auroc": float(roc_auc_score(error, score)),
            "bootstrap_95_ci": list(bootstrap_metric_ci(error, score, metric="auroc")[1:]),
        }

    threshold = float(np.quantile(disagreement, 0.90))
    high = disagreement >= threshold

    return {
        "n_models": len(prediction_files),
        "n_studies": len(merged),
        "ensemble_metrics": ensemble_metrics,
        "overall_error_rate": float(error.mean()),
        "uncertainty_comparison": uncertainty_comparison,
        "top_10_percent_disagreement": {
            "threshold": threshold,
            "count": int(high.sum()),
            "error_rate": float(error[high].mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse identifier-aligned ensemble uncertainty.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyse_ensemble(args.predictions, args.labels)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
