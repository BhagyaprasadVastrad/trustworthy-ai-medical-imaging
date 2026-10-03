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


## Stage 1 Controlled Contrast Shift Result

The same trained baseline model was evaluated on the original test images and on a controlled reduced-contrast version of the same 624 test images. The model was not retrained after the image transformation.

### Comparison

| Metric | Original | Reduced contrast | Change |
|---|---:|---:|---:|
| Accuracy | 0.6538 | 0.6170 | -0.0369 |
| Precision | 0.6505 | 0.6240 | -0.0265 |
| Recall | 0.9641 | 0.9744 | +0.0103 |
| F1 score | 0.7769 | 0.7608 | -0.0161 |
| AUROC | 0.7861 | 0.6909 | -0.0951 |
| Brier score | 0.2071 | 0.2828 | +0.0757 |
| ECE | 0.1114 | 0.2534 | +0.1420 |

### Initial observation

After the contrast reduction, accuracy, precision, F1 and AUROC decreased. Recall increased slightly. At the same time, the Brier score and Expected Calibration Error increased.

The AUROC decrease from 0.7861 to 0.6909 indicates a reduction in the model's ability to distinguish the two classes under this controlled image transformation. The increase in ECE from 0.1114 to 0.2534 indicates that the model's confidence was less aligned with observed accuracy after the shift.

### Important interpretation

This experiment provides a controlled example of how a change in image appearance can affect both predictive performance and confidence quality.

The transformation is artificial and should not be described as a real hospital-to-hospital distribution shift. It is a methodological stress test that prepares the project for later evaluation using more realistic external-domain data.

### Next step

The next experiment will apply temperature scaling using the validation set only. The test set will remain untouched for final evaluation.
