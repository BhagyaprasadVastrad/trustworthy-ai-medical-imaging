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


## Validation Check for Temperature Scaling

The learned temperature (0.9243) was fitted by minimising binary cross-entropy / negative log-likelihood on the validation set.

Validation results before versus after scaling:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Brier score | 0.1489 | 0.1493 | +0.0004 |
| Log loss | 0.4497 | 0.4481 | -0.0016 |
| ECE | 0.0690 | 0.0715 | +0.0025 |

### Interpretation

The temperature optimisation did achieve a small improvement in the validation-set log loss, which is the objective used by the temperature-fitting procedure. It did not improve Brier score or ECE on the validation set.

The held-out test results then showed increases in Brier score, log loss and ECE after scaling. Therefore, the learned temperature provided only a very small validation log-loss improvement and did not generalise as an improvement across the other probability-quality measures on the test data.

This is not evidence that temperature scaling is generally ineffective. It is a result from this particular small baseline experiment and demonstrates the importance of evaluating calibration methods with held-out data and more than one metric.


## Stage 1 Error Analysis

The baseline test-set confusion matrix was:

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 32 | 202 |
| Actual Pneumonia | 14 | 376 |

This corresponds to:

- True negatives: 32
- False positives: 202
- False negatives: 14
- True positives: 376
- False-positive rate: 0.8632
- False-negative rate: 0.0359

### Interpretation

The high recall of 0.9641 is explained by the model identifying 376 of the 390 pneumonia cases, with only 14 false negatives.

However, the model also incorrectly classified 202 of the 234 normal images as pneumonia. This produces a high false-positive rate of 0.8632 and explains why precision and overall accuracy are substantially lower than recall.

Under the reduced-contrast transformation, false positives increased from 202 to 229, while false negatives decreased from 14 to 10. This is consistent with the earlier observation that the contrast shift made the model more likely to predict pneumonia.

The precision-recall curve shows the expected trade-off between recall and precision as the classification threshold changes. The curve reaches high precision at lower recall values and declines as recall approaches 1.0.

### Important caution

The model should not be interpreted as clinically useful. The high false-positive rate on this benchmark and the small, simplified CNN are important limitations. The experiment is being used to study model behaviour, calibration and sensitivity to controlled input changes.


## Stage 2 External-Dataset Evaluation Result

The Stage 1 model was evaluated without retraining or fine-tuning on the RSNA Pneumonia Detection Challenge dataset.

### Dataset and matching

The extracted dataset contained 25,684 DICOM images representing 25,684 unique studies. Study-level labels were constructed from the RSNA annotation JSON:

- No Lung Opacity: 20,025 studies
- Lung Opacity: 5,659 studies

All 25,684 DICOM images were successfully matched to a study-level label. No images were missing labels.

### External evaluation

The trained Stage 1 model generated predictions for all 25,684 studies.

| Metric | PneumoniaMNIST Test | RSNA External | Change |
|---|---:|---:|---:|
| Accuracy | 0.6538 | 0.3639 | -0.2899 |
| Precision | 0.6505 | 0.2373 | -0.4132 |
| Recall | 0.9641 | 0.8521 | -0.1120 |
| F1 score | 0.7769 | 0.3712 | -0.4057 |
| AUROC | 0.7861 | 0.5640 | -0.2221 |
| Brier score | 0.2071 | 0.3579 | +0.1508 |
| ECE | 0.1114 | 0.4164 | +0.3050 |

### Calibration observation

The RSNA reliability diagram showed systematic overconfidence across all populated confidence bins.

For example, the highest populated confidence bin had a mean predicted probability of approximately 0.92 but an observed positive frequency of approximately 0.29.

The ECE increased from 0.1114 on PneumoniaMNIST to 0.4164 on RSNA, while the Brier score increased from 0.2071 to 0.3579.

### Interpretation

The model showed substantial external degradation in both discrimination and probability calibration. Recall remained relatively high, while precision fell sharply, indicating many false-positive predictions.

This should be described as observed external performance degradation and miscalibration. The current experiment does not establish a single causal explanation. Dataset characteristics, image characteristics, prevalence, labeling procedures and preprocessing differences may all contribute.

### Stage 2 conclusion

The experiment supports the observation that a model's predictive and calibration behaviour on its original benchmark does not necessarily transfer reliably to an external dataset.

This strengthens the motivation for studying calibration, uncertainty and robustness under dataset shift.

### Important terminology

The RSNA experiment is an **external-dataset evaluation**. It should not be described as the planned cross-hospital MIMIC-CXR to CheXpert experiment.

### Next step

Before moving to the planned cross-hospital experiment, investigate calibration and uncertainty methods that could better characterise or improve confidence reliability under distribution shift.


## Stage 3 Post-hoc Temperature Scaling Result

The Stage 2 RSNA predictions were split into two stratified subsets of 12,842 studies each. Temperature scaling was fitted only on the calibration subset and evaluated on the separate held-out subset.

The learned temperature was **1.5122**.

### Held-out evaluation

| Metric | Before calibration | After calibration | Change |
|---|---:|---:|---:|
| Brier score | 0.3590 | 0.3210 | -0.0380 |
| Log loss | 0.9685 | 0.8521 | -0.1164 |
| ECE | 0.4175 | 0.3794 | -0.0380 |
| AUROC | 0.5641 | 0.5641 | 0.0000 |

### Interpretation

Temperature scaling partially improved probability calibration on held-out RSNA data. Brier score, log loss and ECE all decreased, while AUROC remained unchanged.

The learned temperature greater than 1 indicates that the original external predictions benefited from probability compression. The reliability diagram showed the post-calibration curve moving closer to the perfect-calibration diagonal across most populated confidence bins.

Calibration remained substantially imperfect after scaling, with ECE still at 0.3794. The highest-confidence post-calibration bin contained only two observations and should not be interpreted independently.

### Important methodological note

This is an external-data calibration experiment, not the planned cross-hospital MIMIC-CXR to CheXpert experiment. The calibration parameter was learned on one portion of RSNA and evaluated on a held-out portion of the same external dataset.

### Stage 3 conclusion

Post-hoc temperature scaling partially reduced calibration error on held-out external data but did not improve the underlying discrimination. The result motivates comparison with additional uncertainty and calibration methods under distribution shift.

