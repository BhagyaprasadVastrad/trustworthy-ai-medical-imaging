# Research Log

## Project question

Can a medical image classifier retain useful confidence estimates when it is evaluated on data that differ from the conditions represented in its training data?

The project is being developed as a small, reproducible research preparation study around:

- distribution shift
- calibration
- confidence
- uncertainty
- external evaluation

The aim is to understand model behaviour rather than present the baseline model as a clinical system.

---

## Stage 1: Baseline and controlled image shift

A small convolutional neural network was trained on PneumoniaMNIST.

The same trained model was evaluated on the original 624-image test set and on a reduced-contrast version of the same images.

| Metric | Original | Reduced contrast | Change |
|---|---:|---:|---:|
| Accuracy | 0.6538 | 0.6170 | -0.0369 |
| Precision | 0.6505 | 0.6240 | -0.0265 |
| Recall | 0.9641 | 0.9744 | +0.0103 |
| F1 score | 0.7769 | 0.7608 | -0.0161 |
| AUROC | 0.7861 | 0.6909 | -0.0951 |
| Brier score | 0.2071 | 0.2828 | +0.0757 |
| ECE | 0.1114 | 0.2534 | +0.1420 |

The controlled contrast change reduced discrimination and worsened calibration.

The transformation is an artificial stress test and is not evidence of real hospital-to-hospital distribution shift.

### Stage 1 temperature-scaling observation

An earlier baseline temperature-scaling experiment produced T = 0.9243 and did not improve the held-out benchmark calibration metrics. This result was useful as an initial calibration-method observation.

The project then moved to a more relevant external-dataset evaluation rather than treating the small benchmark calibration result as definitive.

---

## Stage 2: External RSNA evaluation

The Stage 1 model was evaluated without retraining or fine-tuning on the RSNA Pneumonia Detection Challenge dataset.

The external evaluation contained:

- 25,684 DICOM images
- 25,684 unique studies
- 25,684 matched study-level labels
- 0 missing image-label matches

Study-level labels were:

- No Lung Opacity: 20,025
- Lung Opacity: 5,659

External results:

| Metric | PneumoniaMNIST Test | RSNA External |
|---|---:|---:|
| Accuracy | 0.6538 | 0.3639 |
| Precision | 0.6505 | 0.2373 |
| Recall | 0.9641 | 0.8521 |
| F1 score | 0.7769 | 0.3712 |
| AUROC | 0.7861 | 0.5640 |
| Brier score | 0.2071 | 0.3579 |
| ECE | 0.1114 | 0.4164 |

The external dataset exposed substantial degradation in both discrimination and calibration.

Important terminology: this is an **external-dataset evaluation**, not the planned cross-hospital MIMIC-CXR to CheXpert experiment.

Important label limitation: the original model was trained on PneumoniaMNIST for pneumonia classification, while the RSNA study-level labels used here represent lung opacity. These targets are related but not identical.

---

## Stage 3: Post-hoc temperature scaling

The 25,684 RSNA predictions were split into two stratified subsets of 12,842 studies.

Temperature scaling was fitted on the calibration subset only and evaluated on the held-out subset.

The latest verified run produced:

```text
T = 2.0350
```

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Brier score | 0.3590 | 0.3009 | -0.0581 |
| Log loss | 0.9685 | 0.8016 | -0.1669 |
| ECE | 0.4175 | 0.3566 | -0.0609 |
| AUROC | 0.5641 | 0.5641 | 0.0000 |

Temperature scaling improved probability-based metrics while leaving AUROC unchanged.

The conclusion is that calibration can improve probability quality without improving the classifier's underlying discrimination.

---

## Stage 4: Confidence and error analysis

Stage 4 reused the saved Stage 2 predictions rather than rerunning external inference.

The single-model external predictions contained:

- Correct predictions: 9,346
- Incorrect predictions: 16,338
- Accuracy: 0.3639

Confidence was higher for incorrect predictions:

| Group | Mean confidence | Median confidence |
|---|---:|---:|
| Correct | 0.6384 | 0.6041 |
| Incorrect | 0.6817 | 0.6588 |

There were:

```text
874 incorrect predictions with confidence >= 0.90
```

The most confident incorrect prediction was approximately 0.9698.

Applying T = 2.0350 reduced:

- mean confidence: 0.6659 → 0.5909
- median confidence: 0.6369 → 0.5686
- Brier score: 0.3579 → 0.3003
- log loss: 0.9652 → 0.8003

AUROC remained 0.5640.

The key observation is that the model could be confidently wrong under external evaluation.

---

## Stage 5: Deep ensemble uncertainty

Five independently initialized copies of the same SmallCNN were trained using seeds:

```text
42, 43, 44, 45, 46
```

The models were evaluated on the same 25,684 RSNA studies.

Prediction standard deviation was used as a simple model-disagreement proxy.

| Group | Count | Mean disagreement | Median disagreement |
|---|---:|---:|---:|
| Correct | 10,880 | 0.101513 | 0.104394 |
| Incorrect | 14,804 | 0.093461 | 0.091569 |

The full error-detection AUROC was:

```text
0.4225
```

The highest-disagreement 10% contained:

- 2,569 studies
- 62.36% error rate

The overall error rate was 57.64%, so the highest-disagreement group had a 4.72 percentage-point higher error rate.

However, incorrect predictions had lower average disagreement than correct predictions, and the full AUROC was below 0.50.

Therefore, deep-ensemble disagreement was not a reliable standalone uncertainty signal in this experiment.

---

## Overall research observation

The project has produced a useful sequence of results:

1. Controlled image changes can worsen both classification and calibration.
2. External evaluation can reveal much larger degradation than the original benchmark suggests.
3. Temperature scaling can improve probability quality without changing discrimination.
4. A model can be highly confident when it is wrong.
5. Model disagreement does not automatically provide a useful uncertainty signal.

The negative result from Stage 5 is retained deliberately. It is part of the research evidence rather than something to hide.

---

## Current status

Stages 1–5 have been run and documented.

The project remains a research preparation study. It is not a clinical validation system.

The next research direction is to compare additional calibration and uncertainty approaches under more realistic distribution shifts, eventually moving toward the broader multimodal and trustworthy-AI questions motivating the KCL research application.
