# Research Log

## 2026-10-03: Project initialisation

### Question

Can a medical image classifier retain useful confidence estimates when the input images change from the conditions represented in its training data?

### Stage 1 hypothesis

A model may retain useful discrimination while its confidence becomes less reliable after a controlled change to the input images.

### Planned experiment

1. Train a small binary classifier on PneumoniaMNIST.
2. Evaluate the untouched test set.
3. Measure discrimination and calibration.
4. Apply a controlled contrast shift to test images.
5. Compare performance and calibration before and after the shift.
6. Fit temperature scaling on validation logits only.
7. Re-evaluate the shifted test set.

### Important limitation

The contrast transformation is a controlled stress test. It is not a substitute for a genuine cross-hospital or cross-dataset shift.

### Status

Experiment not yet run.

### Questions for later stages

- Which calibration metrics are most informative for medical imaging?
- How should calibration be reported under distribution shift?
- How do temperature scaling, ensembles and conformal methods compare?
- What evidence would justify moving from a benchmark experiment to an external-domain study?
