# Stage 3: Post-hoc Temperature Scaling on Held-out RSNA Data

## 1. Research question

Stage 3 asks:

> **Can post-hoc temperature scaling improve probability calibration on held-out external-dataset cases without changing the model's underlying discrimination?**

The purpose of this stage is to test whether a calibration parameter learned from one portion of the external RSNA dataset improves probability reliability on a separate portion of the same external dataset.

This is an **external-data calibration experiment**. It is not the planned cross-hospital MIMIC-CXR to CheXpert experiment.

## 2. Experimental design

The complete RSNA prediction set from Stage 2 contained 25,684 studies.

The predictions and labels were split into two stratified subsets:

| Subset | Studies | No Lung Opacity | Lung Opacity |
|---|---:|---:|---:|
| Calibration subset | 12,842 | 10,012 | 2,830 |
| Held-out evaluation subset | 12,842 | 10,013 | 2,829 |
| Total | 25,684 | 20,025 | 5,659 |

Temperature scaling was fitted **only on the calibration subset**.

The held-out evaluation subset was not used to learn the temperature.

This separation is important because evaluating calibration on the same observations used to fit the temperature would give an overly optimistic estimate of generalisation.

## 3. Temperature scaling procedure

The Stage 2 model probabilities were converted back to logits using the inverse sigmoid transformation.

A single positive temperature parameter was then learned by minimising binary cross-entropy / negative log-likelihood on the calibration subset.

The learned temperature was:

**T = 1.5122**

The temperature was positive, so the fitted calibration transformation was valid.

Because T > 1, the transformation reduces the magnitude of the model logits and therefore softens the predicted probabilities.

No model weights were retrained or fine-tuned.

## 4. Held-out evaluation results

Temperature scaling was then applied only to the held-out RSNA evaluation subset.

| Metric | Before calibration | After calibration | Change |
|---|---:|---:|---:|
| Brier score | 0.3590 | 0.3210 | -0.0380 |
| Log loss | 0.9685 | 0.8521 | -0.1164 |
| ECE | 0.4175 | 0.3794 | -0.0380 |
| AUROC | 0.5641 | 0.5641 | 0.0000 |

The Brier score decreased by approximately 10.6% relative to its pre-calibration value.

Log loss decreased by approximately 12.0%.

ECE decreased from 0.4175 to 0.3794.

AUROC remained unchanged at 0.5641.

## 5. Reliability analysis

Before temperature scaling, the held-out reliability curve showed systematic overconfidence. The predicted probabilities were consistently higher than the corresponding observed positive frequencies across the populated confidence range.

Examples before calibration include:

| Mean predicted probability | Observed positive frequency | Count |
|---:|---:|---:|
| 0.2779 | 0.1864 | 59 |
| 0.3613 | 0.1625 | 646 |
| 0.4564 | 0.1559 | 1,943 |
| 0.5515 | 0.1949 | 2,997 |
| 0.6486 | 0.2428 | 2,801 |
| 0.7464 | 0.2772 | 1,952 |
| 0.8530 | 0.2330 | 1,824 |
| 0.9204 | 0.2903 | 620 |

After temperature scaling, the reliability curve moved closer to the perfect-calibration diagonal across most populated bins:

| Mean predicted probability | Observed positive frequency | Count |
|---:|---:|---:|
| 0.3754 | 0.1992 | 266 |
| 0.4613 | 0.1537 | 2,382 |
| 0.5499 | 0.2073 | 4,418 |
| 0.6437 | 0.2664 | 3,033 |
| 0.7505 | 0.2437 | 1,945 |
| 0.8284 | 0.2638 | 796 |
| 0.9103 | 1.0000 | 2 |

The final post-calibration bin contained only two observations. Its observed frequency is therefore unstable and should not be interpreted as evidence of reliable calibration at very high confidence.

## 6. Interpretation

Temperature scaling partially improved probability calibration on held-out external data.

The decrease in Brier score, log loss and ECE indicates that the calibrated probabilities were better aligned with the observed outcomes than the original probabilities on this held-out subset.

The learned temperature of 1.5122 indicates that the original external predictions benefited from probability compression rather than increased confidence.

However, calibration remained substantially imperfect after scaling. The ECE was still 0.3794 after calibration.

Therefore, the experiment should not be described as solving the calibration problem.

A more appropriate conclusion is:

> **Post-hoc temperature scaling partially reduced calibration error on held-out RSNA data, but substantial miscalibration remained.**

The unchanged AUROC is consistent with the expected behaviour of temperature scaling: it changes confidence magnitude without changing the ranking of predictions.

## 7. Relationship to Stage 2

Stage 2 showed that the model's performance and confidence reliability degraded substantially on the external RSNA dataset.

Stage 3 then tested whether a simple post-hoc calibration method could recover some of the probability reliability.

The combined observation is:

1. External evaluation produced substantial degradation in discrimination and calibration.
2. The external predictions were systematically overconfident.
3. Temperature scaling partially reduced this overconfidence on held-out external data.
4. Temperature scaling did not improve the underlying discrimination, as AUROC remained 0.5641.
5. Significant calibration error remained after scaling.

This supports further investigation of calibration and uncertainty methods under distribution shift.

## 8. Important limitations

1. The RSNA experiment is an external-dataset evaluation, not the planned cross-hospital MIMIC-CXR to CheXpert evaluation.
2. Temperature scaling was evaluated on a held-out subset of the same external dataset used to fit the calibration parameter. It therefore tests held-out external-data calibration rather than zero-shot calibration transfer to a completely different dataset.
3. The Stage 1 model is a small convolutional neural network trained on PneumoniaMNIST and was not retrained on RSNA.
4. RSNA images were resized from 1024 × 1024 to 64 × 64 before inference.
5. ECE depends on bin definitions and sample counts.
6. The highest-confidence post-calibration bin contained only two observations and is therefore unstable.
7. The experiment does not establish clinical usefulness, clinical safety or causal explanations for the external degradation.
8. The result from one temperature parameter should not be generalised to all medical imaging models or datasets.

## 9. Stage 3 conclusion

The Stage 3 experiment provides evidence that post-hoc temperature scaling can partially improve probability calibration on held-out external-dataset cases even when the underlying discrimination remains weak.

The result strengthens the motivation for comparing more robust approaches to uncertainty and calibration under distribution shift, including methods that can provide richer uncertainty information than a single global temperature parameter.

## 10. Next stage

The next stage should build on the observed external miscalibration and investigate additional uncertainty/calibration approaches before the planned MIMIC-CXR to CheXpert cross-hospital experiment.

Potential directions include:

- comparison of calibration methods
- uncertainty estimation
- calibration under controlled distribution shifts
- reliability analysis across subgroups or confidence ranges
- eventual evaluation in the planned MIMIC-CXR to CheXpert setting
