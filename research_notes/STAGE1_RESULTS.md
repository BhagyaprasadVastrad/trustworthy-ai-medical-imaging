# Stage 1: Baseline Calibration and Controlled Image Shift

## 1. Research question

The first stage asks:

> **Does a medical image classifier's confidence remain meaningful when the appearance of its input images changes?**

The purpose of this stage is to establish a small, reproducible baseline before moving to more realistic external-domain evaluation.

## 2. Dataset

Stage 1 uses **PneumoniaMNIST**, loaded through the MedMNIST package.

The task is binary classification:

- **0 = normal**
- **1 = pneumonia**

| Split | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Training | 1,214 | 3,494 | 4,708 |
| Validation | 135 | 389 | 524 |
| Test | 234 | 390 | 624 |

The training data are substantially imbalanced toward pneumonia. This is important when interpreting accuracy and precision.

The dataset is downloaded at runtime and is not redistributed in this repository.

## 3. Baseline model

A small convolutional neural network (CNN) was used as the baseline.

The model contains:

- three convolutional layers
- ReLU activation
- max-pooling
- adaptive average pooling
- one final binary classification layer

The model was trained for **5 epochs** using:

- Binary cross-entropy with logits
- Adam optimiser
- Learning rate: 0.001
- Training batch size: 64

The test set was not used during model training.

## 4. Training behaviour

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.5903 | 0.5673 |
| 2 | 0.5669 | 0.5596 |
| 3 | 0.5509 | 0.5460 |
| 4 | 0.5207 | 0.4934 |
| 5 | 0.4845 | 0.4497 |

Training and validation loss both decreased during the five-epoch baseline run.

## 5. Baseline test performance

| Metric | Result |
|---|---:|
| Accuracy | 0.6538 |
| Precision | 0.6505 |
| Recall | 0.9641 |
| F1 score | 0.7769 |
| AUROC | 0.7861 |
| Brier score | 0.2071 |
| Expected Calibration Error (ECE) | 0.1114 |

The model achieved high recall but considerably lower precision and accuracy. It identified most pneumonia cases but also produced many false pneumonia predictions among normal images.

## 6. Baseline calibration

The reliability diagram showed noticeable deviation from the perfect-calibration diagonal, particularly across several intermediate-confidence ranges.

For example, one confidence group had approximately 0.73 average model confidence but only approximately 0.40 observed accuracy. This indicates substantial overconfidence in that group.

The overall ECE was **0.1114**.

The experiment demonstrates that classification performance and confidence quality are different properties.

## 7. Controlled image shift

The contrast of all 624 test images was reduced by a factor of 0.5.

The trained model was evaluated on these modified images without retraining.

This was deliberately designed as a **controlled stress test**. It is not treated as evidence of a genuine hospital-to-hospital distribution shift.

## 8. Effect of the controlled shift

| Metric | Original | Reduced contrast | Change |
|---|---:|---:|---:|
| Accuracy | 0.6538 | 0.6170 | -0.0369 |
| Precision | 0.6505 | 0.6240 | -0.0265 |
| Recall | 0.9641 | 0.9744 | +0.0103 |
| F1 score | 0.7769 | 0.7608 | -0.0161 |
| AUROC | 0.7861 | 0.6909 | -0.0951 |
| Brier score | 0.2071 | 0.2828 | +0.0757 |
| ECE | 0.1114 | 0.2534 | +0.1420 |

The controlled contrast change reduced classification performance and substantially worsened confidence quality.

## 9. Error analysis

Baseline confusion matrix:

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| **Actual Normal** | 32 | 202 |
| **Actual Pneumonia** | 14 | 376 |

Therefore:

- True negatives: **32**
- False positives: **202**
- False negatives: **14**
- True positives: **376**
- False-positive rate: **0.8632**
- False-negative rate: **0.0359**

After the contrast shift:

- False positives increased from **202 to 229**
- False negatives decreased from **14 to 10**

This is consistent with the observed increase in pneumonia probabilities after the contrast transformation.

## 10. Precision-recall behaviour

The precision-recall curve showed the expected trade-off between precision and recall as the classification threshold changes.

At very high recall, precision decreases.

No clinical decision threshold is proposed from this experiment.

## 11. Temperature scaling

Temperature scaling was introduced as a simple post-hoc calibration method.

The temperature parameter was fitted using the **validation set only**.

The learned temperature was **0.9243**.

The test set was not used to learn the temperature.

## 12. Exploratory temperature scaling results

### Expected Calibration Error

| Dataset | Before | After |
|---|---:|---:|
| Original test | 0.1114 | 0.1288 |
| Reduced-contrast test | 0.2534 | 0.2660 |

### Brier score and log loss

| Dataset | Brier before | Brier after | Log loss before | Log loss after |
|---|---:|---:|---:|---:|
| Original test | 0.2071 | 0.2102 | 0.5948 | 0.6035 |
| Reduced contrast | 0.2828 | 0.2901 | 0.8427 | 0.8845 |

All three probability-quality measures were worse on the held-out test data after temperature scaling.

## 13. Validation check

| Metric | Before | After |
|---|---:|---:|
| Brier score | 0.1489 | 0.1493 |
| Log loss | 0.4497 | 0.4481 |
| ECE | 0.0690 | 0.0715 |

The fitting objective was binary cross-entropy / negative log-likelihood, so the small reduction in validation log loss is consistent with the optimisation objective.

However, the improvement was very small, and Brier score and ECE did not improve on validation. More importantly, the small validation log-loss improvement did not generalise to the held-out test set.

## 14. What Stage 1 taught us

### Finding 1: Discrimination and calibration are different

The model showed measurable discriminatory ability but produced confidence estimates that were not consistently aligned with observed accuracy.

### Finding 2: A controlled input change affected both performance and confidence

Reducing image contrast caused lower AUROC, lower accuracy, lower precision, higher Brier score and higher ECE. The model also became more likely to predict pneumonia.

### Finding 3: Calibration cannot be assumed to generalise

Temperature scaling produced only a very small improvement in validation log loss and did not improve the held-out test calibration metrics.

This demonstrates why calibration methods should be evaluated on data that were not used to fit the calibration parameter.

## 15. Limitations

1. **PneumoniaMNIST is a benchmark dataset**, not a substitute for independent hospital data.
2. The contrast transformation is **artificial** and should not be described as genuine hospital-to-hospital distribution shift.
3. The CNN is intentionally small and is not a state-of-the-art medical imaging model.
4. The test set contains only 624 images.
5. ECE depends on the calibration-bin definition and should not be interpreted as a complete description of uncertainty.
6. The experiment does not establish clinical usefulness or clinical safety.
7. The model's high false-positive rate is a substantial limitation of this baseline.

## 16. Stage 1 conclusion

The baseline experiment supports the following research observation:

> **A medical image classifier can retain measurable classification ability while producing confidence estimates that are imperfectly calibrated, and a controlled change in image appearance can further degrade both discrimination and calibration.**

The experiment also shows that a simple post-hoc calibration method cannot automatically be assumed to solve the problem.

## 17. Next stage

**Stage 2: External-domain evaluation**

The next experiment will investigate whether a model trained on one medical-image distribution behaves differently when evaluated on data from another distribution.

The central question will become:

> **Does calibration learned on one dataset remain reliable when the model encounters a different data distribution?**

This is closer to the cross-hospital reliability question motivating the broader research direction.
