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


## Stage 1 Temperature Scaling Result

Temperature scaling was fitted using the validation set only. The learned temperature was **0.9243**.

### Test-set calibration comparison

| Dataset | ECE before | ECE after | Change |
|---|---:|---:|---:|
| Original test | 0.1114 | 0.1288 | +0.0174 |
| Reduced-contrast test | 0.2534 | 0.2660 | +0.0126 |

### Initial observation

For this baseline experiment, temperature scaling did not improve Expected Calibration Error on the untouched test set. ECE increased from 0.1114 to 0.1288. It also did not improve ECE on the reduced-contrast test set, where ECE increased from 0.2534 to 0.2660.

The learned temperature was below 1, meaning the fitted calibration procedure increased the magnitude of the model's logits rather than reducing them.

### Interpretation

This result should not be treated as evidence that temperature scaling is generally ineffective. It shows that, for this particular small model, dataset split and experimental setup, fitting one temperature on the validation set did not improve the ECE measured on these test sets.

The result is useful because it demonstrates why calibration methods need to be evaluated on held-out data rather than assumed to improve confidence automatically.

### Limitation

This is a small benchmark experiment. ECE is sensitive to binning and sample size, and the validation and test distributions are from the same benchmark rather than independent hospitals. Further analysis should include validation-set calibration behaviour, Brier score and negative log-likelihood, and should compare reliability diagrams before and after scaling.

### Next step

Before moving to a more complex calibration method, inspect the validation-set effect and compare Brier score and negative log-likelihood before and after temperature scaling. Then decide whether the baseline experiment needs refinement.


## Stage 1 Temperature Scaling Evaluation

Temperature scaling was fitted on the validation set only, producing a learned temperature of **0.9243**.

On the held-out test sets, temperature scaling did not improve the probability-quality metrics:

| Dataset | Brier before | Brier after | Log loss before | Log loss after |
|---|---:|---:|---:|---:|
| Original test | 0.2071 | 0.2102 | 0.5948 | 0.6035 |
| Reduced-contrast test | 0.2828 | 0.2901 | 0.8427 | 0.8845 |

ECE also increased:

| Dataset | ECE before | ECE after |
|---|---:|---:|
| Original test | 0.1114 | 0.1288 |
| Reduced-contrast test | 0.2534 | 0.2660 |

The reliability diagram likewise does not show a clear improvement after scaling.

### Interpretation

For this experiment, temperature scaling did not improve the held-out test-set calibration or probability-quality measures. This is a result to investigate rather than evidence that temperature scaling is ineffective in general.

Before drawing a stronger conclusion, the validation-set objective should be checked directly to confirm that the learned temperature improved the metric on the data used for fitting. This will also help distinguish a calibration-method limitation from a validation-to-test mismatch.

### Current Stage 1 conclusion

The baseline experiment has now shown two useful behaviours:

1. The model has measurable discriminative ability but imperfect calibration.
2. A controlled reduction in image contrast worsened discrimination and confidence quality.
3. Temperature scaling fitted on the validation set did not transfer into improved held-out calibration for this experiment.

The next analysis will inspect validation-set changes before deciding whether to refine temperature scaling or move to another calibration method.
