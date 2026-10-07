"""Identifier-based alignment utilities for external medical-imaging predictions."""

from __future__ import annotations

import pandas as pd


def align_by_study_id(predictions: pd.DataFrame, labels: pd.DataFrame,
                      id_col: str = "StudyInstanceUID",
                      prediction_col: str = "probability",
                      label_col: str = "label") -> pd.DataFrame:
    """Align predictions and labels by study identifier and reject ambiguous inputs."""
    required_pred = {id_col, prediction_col}
    required_label = {id_col, label_col}
    if not required_pred.issubset(predictions.columns):
        raise ValueError(f"Predictions must contain {sorted(required_pred)}")
    if not required_label.issubset(labels.columns):
        raise ValueError(f"Labels must contain {sorted(required_label)}")
    if predictions[id_col].duplicated().any():
        raise ValueError("Predictions contain duplicate study identifiers")
    if labels[id_col].duplicated().any():
        raise ValueError("Labels contain duplicate study identifiers")

    merged = predictions[[id_col, prediction_col]].merge(
        labels[[id_col, label_col]], on=id_col, how="inner", validate="one_to_one"
    )
    if len(merged) != len(predictions) or len(merged) != len(labels):
        missing_predictions = set(labels[id_col]) - set(predictions[id_col])
        missing_labels = set(predictions[id_col]) - set(labels[id_col])
        raise ValueError(
            "Prediction/label coverage mismatch: "
            f"missing predictions={len(missing_predictions)}, missing labels={len(missing_labels)}"
        )
    merged[label_col] = merged[label_col].astype(int)
    return merged
