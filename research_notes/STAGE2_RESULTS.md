# Stage 2: External-Dataset Evaluation on RSNA Pneumonia Detection Challenge

## 1. Research question

Stage 2 asks:

> **Does calibration learned on one dataset remain reliable when the model encounters a different data distribution?**

The purpose of this stage is to evaluate whether the Stage 1 model maintains similar predictive and probability-calibration behaviour on an external chest X-ray dataset.

This stage is an **external-dataset evaluation**. It is not treated as a cross-hospital experiment.

## 2. External dataset

The RSNA Pneumonia Detection Challenge dataset was used for external evaluation.

The extracted dataset contained:

- **25,684 DICOM images**
- **25,684 unique studies**
- **20,025 No Lung Opacity studies**
- **5,659 Lung Opacity studies**

Study-level labels were constructed from the RSNA annotation JSON. Every extracted DICOM image was successfully matched to a study-level label:

- DICOM images: **25,684**
- Images without a label: **0**
- Unique studies: **25,684**

The dataset was not used for model training or fine-tuning.

## 3. Evaluation pipeline

The Stage 1 small convolutional neural network (CNN) was used without retraining.

For each RSNA image:

1. Load the DICOM image.
2. Extract the grayscale pixel array.
3. Resize the image from 1024 × 1024 to 64 × 64.
4. Convert the image using `ToTensor()`.
5. Pass the tensor through the trained Stage 1 model.
6. Convert the model output to a probability using the sigmoid function.
7. Compare the resulting probability with the corresponding RSNA study-level label.

A single-image inference test was completed successfully before full evaluation.

The complete dataset inference then produced:

- **25,684 predictions**
- **25,684 reference labels**
- Minimum predicted probability: **0.2125**
- Maximum predicted probability: **0.9746**

## 4. External-dataset performance

Using a probability threshold of 0.50:

| Metric | RSNA External |
|---|---:|
| Accuracy | 0.3639 |
| Precision | 0.2373 |
| Recall | 0.8521 |
| F1 score | 0.3712 |
| AUROC | 0.5640 |
| Brier score | 0.3579 |
| Expected Calibration Error (ECE) | 0.4164 |

The model retained relatively high recall but showed substantially lower precision, F1 score and AUROC than in the original PneumoniaMNIST test evaluation.

## 5. Reliability analysis

The RSNA reliability diagram showed systematic overconfidence across every populated confidence bin.

Examples include:

| Mean predicted probability | Observed positive frequency | Samples |
|---:|---:|---:|
| 0.2753 | 0.1897 | 116 |
| 0.3628 | 0.1561 | 1,294 |
| 0.4565 | 0.1552 | 3,951 |
| 0.5516 | 0.2013 | 5,892 |
| 0.6481 | 0.2412 | 5,759 |
| 0.7461 | 0.2709 | 3,854 |
| 0.8525 | 0.2371 | 3,593 |
| 0.9201 | 0.2865 | 1,225 |

The model's predicted confidence was consistently higher than the observed positive frequency.

For example, predictions with a mean confidence of approximately 0.92 corresponded to an observed positive frequency of approximately 0.29.

This systematic separation from the perfect-calibration diagonal is consistent with the high ECE of **0.4164** and Brier score of **0.3579**.

## 6. Comparison with Stage 1

| Metric | PneumoniaMNIST Test | RSNA External | Change |
|---|---:|---:|---:|
| Accuracy | 0.6538 | 0.3639 | -0.2899 |
| Precision | 0.6505 | 0.2373 | -0.4132 |
| Recall | 0.9641 | 0.8521 | -0.1120 |
| F1 score | 0.7769 | 0.3712 | -0.4057 |
| AUROC | 0.7861 | 0.5640 | -0.2221 |
| Brier score | 0.2071 | 0.3579 | +0.1508 |
| ECE | 0.1114 | 0.4164 | +0.3050 |

The external evaluation shows substantial degradation in both predictive performance and probability quality.

The largest calibration change was the increase in ECE from **0.1114** to **0.4164**.

AUROC also decreased from **0.7861** to **0.5640**, indicating substantially weaker discrimination on the external dataset.

Recall remained comparatively high, but precision decreased sharply. This indicates that the model continued to identify many positive cases while also producing many false-positive predictions.

## 7. Interpretation

The results provide empirical evidence that the Stage 1 model does not maintain the same predictive and calibration behaviour on the external RSNA dataset.

The reliability analysis indicates systematic overconfidence rather than a simple random degradation of confidence.

However, the experiment does **not** establish a single causal explanation for the observed degradation.

Potential contributors include:

- differences in dataset characteristics
- image characteristics and acquisition conditions
- class prevalence
- labeling procedures
- preprocessing and resolution changes
- differences between the benchmark and external datasets

Therefore, the current result should be described as **observed external performance degradation and miscalibration**, rather than attributing the effect to one specific source.

## 8. Stage 2 conclusion

The Stage 2 experiment supports the following research observation:

> **A medical image classifier that shows measurable performance and imperfect but substantially better calibration on its original benchmark can exhibit marked degradation in both discrimination and confidence reliability when evaluated on an external dataset.**

This provides a stronger empirical motivation for studying calibration, uncertainty and robustness under dataset shift.

## 9. Limitations

1. The RSNA evaluation is an external-dataset experiment and should not be described as a cross-hospital validation.
2. The Stage 1 model is intentionally small and was trained only on PneumoniaMNIST.
3. RSNA images required resizing from 1024 × 1024 to 64 × 64, introducing a preprocessing difference that may contribute to performance changes.
4. The two datasets differ in prevalence and labeling methodology.
5. ECE depends on the number and definition of calibration bins.
6. The experiment does not establish clinical usefulness or clinical safety.
7. The observed degradation cannot be attributed to a single causal mechanism from this experiment alone.

## 10. Next stage

The next research stage should investigate methods for improving or characterising reliability under distribution shift, with particular attention to:

- post-hoc calibration
- uncertainty estimation
- robustness to external-domain changes
- comparison of calibration methods
- eventual evaluation on the planned MIMIC-CXR to CheXpert cross-hospital setting
