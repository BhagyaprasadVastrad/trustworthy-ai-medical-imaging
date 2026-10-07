# Trustworthy AI for Medical Imaging

Research preparation project exploring calibration, confidence and uncertainty under distribution shift in medical image classification.

## Key results

| External evaluation | Before calibration | After temperature scaling |
|---|---:|---:|
| **AUROC** | **0.5641** | **0.5641** |
| **Expected Calibration Error (ECE)** | **0.4175** | **0.3566** |
| **Brier score** | 0.3590 | 0.3009 |
| **Log loss** | 0.9685 | 0.8016 |

The external evaluation covered **25,684 RSNA studies**. Temperature scaling improved probability quality while leaving discrimination unchanged.

![External evaluation vs baseline](results/stage2_external_vs_stage1.svg)

![Temperature scaling results](results/stage3_temperature_scaling.svg)

## Aim

The project investigates a practical question:

> When a medical imaging model reports a confidence score, does that confidence remain meaningful when the input data change?

The work is developed as a small, reproducible sequence of experiments covering:

- medical image classification
- controlled image shift
- external-dataset evaluation
- probability calibration
- confidence and error analysis
- model-disagreement-based uncertainty

The emphasis is on understanding model behaviour rather than presenting a benchmark model as a clinical system.

## Research pipeline

```text
Stage 1
PneumoniaMNIST baseline
        ↓
Controlled reduced-contrast shift
        ↓
Stage 2
External evaluation on RSNA
        ↓
Stage 3
Temperature scaling
        ↓
Stage 4
Confidence and error analysis
        ↓
Stage 5
Deep-ensemble uncertainty
```

## Current status

Stages 1–5 have been run and documented.

### Stage 1

A small convolutional neural network was trained on PneumoniaMNIST. A controlled reduction in image contrast reduced discrimination and worsened calibration.

### Stage 2

The Stage 1 model was evaluated without retraining on 25,684 RSNA studies. External performance and calibration degraded substantially.

### Stage 3

Temperature scaling was fitted on one stratified portion of the RSNA predictions and evaluated on a held-out portion. Probability quality improved, while AUROC remained unchanged.

### Stage 4

The analysis showed that the model could be more confident on incorrect predictions than on correct predictions. High-confidence errors were present in the external evaluation.

### Stage 5

Five independently trained copies of the same small CNN were used to measure model disagreement. Disagreement showed a limited association with errors in the highest-disagreement subset, but it was not a reliable standalone uncertainty signal across the full dataset.

## Main findings

The experiments support several practical observations:

1. A controlled change in image appearance can affect both discrimination and confidence quality.
2. Performance and calibration can degrade when a model is evaluated on an external dataset.
3. Calibration can improve probability quality without improving discrimination.
4. A model can be confidently wrong under external distribution shift.
5. Model disagreement does not automatically provide a reliable uncertainty measure.

The negative result in Stage 5 is intentionally retained. The purpose of the project is to investigate reliability, not to assume that every uncertainty method will work.

## Important scope and limitations

This is a research preparation project, not a clinical validation study.

The baseline model was trained on PneumoniaMNIST, while the external RSNA evaluation uses study-level lung-opacity labels. These targets are related but not identical.

The controlled contrast experiment is an artificial stress test and is not evidence of real hospital-to-hospital shift.

The deep ensemble contains five small CNNs, and prediction standard deviation is used as a model-disagreement proxy rather than a complete uncertainty decomposition.

The experiments are intended to support methodological learning and research discussion around trustworthy medical AI.

## Repository structure

```text
notebooks/       Experimental notebooks
research_notes/  Research observations and stage results
results/         Generated figures and result summaries
```

## Data

The repository does not redistribute medical datasets.

Stage 1 uses PneumoniaMNIST through the MedMNIST package. The RSNA data used for external evaluation are expected to be obtained separately and are not stored in this repository.

Dataset resource:
[MedMNIST](https://medmnist.com/)

## Setup

See [SETUP.md](SETUP.md) for the current environment and dataset notes.

## Status

This is an active research preparation project. Results are documented only after the corresponding experiments have been run and checked.
