# Evaluation framework

## Purpose

This is the first reusable layer added after the initial project audit. It standardises the metrics used by future experiments without modifying the five existing experimental notebooks.

## Metrics

`src/evaluation.py` provides:

- accuracy
- precision
- recall
- F1
- AUROC
- AUPRC
- Brier score
- log loss
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)

AUPRC is included because the RSNA external evaluation is class-imbalanced.

## Calibration definition

The existing Stage 1-4 results use ECE based on confidence in the predicted class, with ten bins from 0.50 to 1.00. That definition is retained for historical comparability rather than silently changing previously reported values.

Future calibration experiments can report an additional probability-based calibration definition if scientifically justified.

## Bootstrap confidence intervals

`src/statistics.py` provides percentile bootstrap intervals with a default of 2,000 resamples and seed 42. Metrics that are undefined on a one-class resample, such as AUROC, are skipped for that resample.

## Scope

This framework does not change any historical result. The next step is to apply it to stored prediction arrays where they are available, verify metric agreement with the existing notebooks, and then use it for new experiments.