# Trustworthy AI for Medical Imaging: Research Summary

## Project question

> How reliable is a medical imaging model when it is used outside the data it was originally trained on?

The project started with a small chest X-ray classifier and developed into a sequence of experiments around distribution shift, calibration, confidence and uncertainty.

## Research pipeline

```text
Stage 1
Baseline model
      ↓
Controlled image shift
      ↓
Stage 2
External RSNA evaluation
      ↓
Stage 3
Temperature scaling
      ↓
Stage 4
Confidence and error analysis
      ↓
Stage 5
Deep ensemble uncertainty
```

## Stage 1: Baseline and controlled image shift

A small convolutional neural network was trained on PneumoniaMNIST.

The same trained model was then evaluated on the original test images and on a reduced-contrast version of the same 624 test images.

| Metric | Original | Reduced contrast |
|---|---:|---:|
| Accuracy | 0.6538 | 0.6170 |
| Precision | 0.6505 | 0.6240 |
| Recall | 0.9641 | 0.9744 |
| F1 | 0.7769 | 0.7608 |
| AUROC | 0.7861 | 0.6909 |
| Brier score | 0.2071 | 0.2828 |
| ECE | 0.1114 | 0.2534 |

The controlled image change reduced discrimination and worsened probability calibration.

This was deliberately treated as a stress test rather than a simulation of real hospital-to-hospital shift.

## Stage 2: External RSNA evaluation

The Stage 1 model was evaluated without retraining or fine-tuning on 25,684 RSNA studies.

All 25,684 DICOM images were matched to study-level labels.

| Metric | PneumoniaMNIST test | RSNA external |
|---|---:|---:|
| Accuracy | 0.6538 | 0.3639 |
| Precision | 0.6505 | 0.2373 |
| Recall | 0.9641 | 0.8521 |
| F1 | 0.7769 | 0.3712 |
| AUROC | 0.7861 | 0.5640 |
| Brier score | 0.2071 | 0.3579 |
| ECE | 0.1114 | 0.4164 |

The model's external performance and calibration degraded substantially.

An important limitation is that the targets are not identical: PneumoniaMNIST is a pneumonia classification task, while the RSNA study-level labels used here represent lung opacity.

Therefore, this is an external-dataset evaluation, not a direct clinical validation of pneumonia detection.

## Stage 3: Temperature scaling

The RSNA predictions were split into two stratified subsets of 12,842 studies.

Temperature scaling was fitted on one subset and evaluated on the held-out subset.

The latest verified run produced:

```text
T = 2.0350
```

| Metric | Before | After |
|---|---:|---:|
| Brier score | 0.3590 | 0.3009 |
| Log loss | 0.9685 | 0.8016 |
| ECE | 0.4175 | 0.3566 |
| AUROC | 0.5641 | 0.5641 |

The probability-based metrics improved while AUROC remained unchanged.

This means the probabilities became better behaved without changing the underlying ranking ability of the classifier.

## Stage 4: Confidence and error analysis

The single-model external predictions contained:

- 9,346 correct predictions
- 16,338 incorrect predictions
- Accuracy: 0.3639

Mean confidence was higher for incorrect predictions:

| Group | Mean confidence | Median confidence |
|---|---:|---:|
| Correct | 0.6384 | 0.6041 |
| Incorrect | 0.6817 | 0.6588 |

There were 874 incorrect predictions with confidence >= 0.90.

The most confident incorrect prediction had probability approximately 0.9698.

This is one of the clearest trustworthy-AI findings in the project: the model could be confidently wrong under external evaluation.

Applying the Stage 3 temperature reduced mean confidence from 0.6659 to 0.5909 and improved Brier score and log loss, while AUROC remained unchanged.

## Stage 5: Deep ensemble uncertainty

Five independently initialized copies of the same SmallCNN were trained using seeds 42–46.

The models were evaluated on the same 25,684 RSNA studies.

The experiment used prediction standard deviation as a model-disagreement proxy.

| Group | Count | Mean disagreement | Median disagreement |
|---|---:|---:|---:|
| Correct | 10,880 | 0.101513 | 0.104394 |
| Incorrect | 14,804 | 0.093461 | 0.091569 |

The full error-detection AUROC was 0.4225.

The highest-disagreement 10% contained 2,569 studies and had a 62.36% error rate, compared with 57.64% overall.

So the highest-disagreement group had somewhat more errors, but disagreement did not reliably rank incorrect predictions above correct predictions.

No studies were both >=0.90 confidence and in the top 10% disagreement group.

## Overall findings

The experiments produced five main observations:

1. **Controlled image changes can affect both performance and calibration.**
2. **External evaluation can expose substantial degradation that is not obvious from the original benchmark.**
3. **Calibration and discrimination are different properties.**
4. **A model can be highly confident when it is wrong.**
5. **Model disagreement is not automatically a reliable uncertainty measure.**

The project therefore did not end with a claim that one uncertainty method solved the problem.

Instead, the experiments showed why reliability needs to be measured from several angles.

## Reanalysis status

The quantitative results above are the recorded notebook results from the original single-seed experiment unless explicitly stated otherwise. A corrected `src/` workflow has now been added to rerun Stages 2–4 across seeds 42–46 and to recompute Stage 5 with identifier-based prediction/label alignment. It also adds simple uncertainty baselines and bootstrap confidence intervals.

Those corrected multi-seed runs have **not been executed in this environment** because the external RSNA DICOM data and local model checkpoints are not present here. Therefore no new multi-seed numerical result is presented as if it had been rerun.

## Limitations

This is a research preparation project, not a clinical validation study.

The main limitations are:

- The baseline model is intentionally small.
- PneumoniaMNIST and RSNA use related but non-identical targets.
- The controlled contrast shift is artificial.
- The ensemble contains only five models.
- Ensemble standard deviation is a model-disagreement proxy, not a complete uncertainty decomposition.
- The external evaluation uses one dataset.
- PneumoniaMNIST is derived from pediatric chest radiographs, while the RSNA dataset is primarily an adult chest-radiograph cohort. This age/distribution gap is an important confounder when interpreting the external performance drop.\n- The experiments focus on binary image-only classification.

## Future direction

The next stage of research can investigate how calibration, model disagreement and other uncertainty signals behave under more realistic distribution shifts.

Possible directions include:

- additional calibration methods
- alternative uncertainty estimation methods
- external validation across additional datasets
- structured clinical information alongside imaging
- multimodal medical-imaging models
- more realistic cross-site or cross-institution evaluation

## Conclusion

The main lesson from this project is:

> **Trustworthy medical AI requires looking beyond whether a model is correct. We also need to understand whether its confidence and uncertainty remain meaningful when the model encounters data that differ from its original training environment.**

The project provides a small, reproducible foundation for continuing that investigation toward more reliable and uncertainty-aware medical AI systems.
