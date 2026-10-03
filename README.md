# Trustworthy AI for Medical Imaging

Research preparation project exploring calibration and distribution shift in medical image classification.

## Aim

The project investigates a practical question:

> When a medical imaging model reports a confidence score, does that confidence remain meaningful when the input data change?

The work is being developed as a small, reproducible study of:

- medical image classification
- model calibration
- uncertainty estimation
- distribution shift
- evaluation under controlled changes to image data

## Current stage

**Stage 1: Baseline calibration study**

The first experiment uses PneumoniaMNIST as a compact, reproducible dataset for binary medical image classification.

The planned workflow is:

1. Train a small convolutional neural network.
2. Evaluate predictions on an untouched test set.
3. Measure accuracy and probability-based metrics.
4. Examine calibration with reliability diagrams and Expected Calibration Error.
5. Introduce a controlled image contrast shift to the test images.
6. Compare predictive performance and calibration before and after the shift.
7. Fit temperature scaling using validation data only.
8. Evaluate whether calibration improves without using the test set for fitting.

This controlled contrast change is a stress test. It is **not** being treated as a real hospital-to-hospital shift.

## Why this project

The project is preparation for deeper research into trustworthy AI for healthcare, particularly questions around confidence, uncertainty and robustness when medical data differ from the conditions used to train a model.

It is deliberately being developed incrementally. Results will be added only after experiments are actually run and checked.

## Repository structure

```text
notebooks/       Experimental notebooks
research_notes/  Research observations and decisions
results/         Generated figures and result summaries
src/             Reusable experiment code
```

## Data

The repository does not redistribute the dataset. The notebook downloads the dataset through the MedMNIST package.

Dataset:
[MedMNIST](https://medmnist.com/)

The dataset and its terms should be reviewed before reuse or redistribution.

## Status

This is an active research preparation project. Early experiments are intended for learning, methodological practice and discussion rather than as a clinical system or validated medical model.
