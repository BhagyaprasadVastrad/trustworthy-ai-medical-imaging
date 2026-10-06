# Stage 3: Post-hoc Temperature Scaling on Held-out RSNA Data

## Research question

> **Can post-hoc temperature scaling improve probability calibration on held-out external-dataset cases without changing the model's underlying discrimination?**

This is an external-data calibration experiment. It is not the planned cross-hospital MIMIC-CXR to CheXpert experiment.

## Experimental design

The Stage 2 RSNA prediction set contained 25,684 studies.

The predictions were split into two stratified subsets:

| Subset | Studies | No Lung Opacity | Lung Opacity |
|---|---:|---:|---:|
| Calibration subset | 12,842 | 10,012 | 2,830 |
| Held-out evaluation subset | 12,842 | 10,013 | 2,829 |
| Total | 25,684 | 20,025 | 5,659 |

Temperature scaling was fitted only on the calibration subset.

The held-out subset was used only for evaluation.

## Temperature scaling

The Stage 2 probabilities were converted to logits and a single positive temperature parameter was learned by minimising binary cross-entropy / negative log-likelihood on the calibration subset.

The latest verified run produced:

```text
T = 2.0350
```

No model weights were retrained or fine-tuned.

Because T > 1, the transformation compresses the logits and reduces extreme probabilities.

## Held-out results

| Metric | Before calibration | After calibration | Change |
|---|---:|---:|---:|
| Brier score | 0.3590 | 0.3009 | -0.0581 |
| Log loss | 0.9685 | 0.8016 | -0.1669 |
| ECE | 0.4175 | 0.3566 | -0.0609 |
| AUROC | 0.5641 | 0.5641 | 0.0000 |

The probability-based metrics improved while AUROC remained unchanged.

This is the expected qualitative behaviour of temperature scaling: confidence magnitude changes, but the ranking of predictions does not.

## Interpretation

The model was substantially overconfident under external evaluation.

Temperature scaling reduced this overconfidence and improved Brier score, log loss and ECE on the held-out RSNA subset.

However, calibration remained imperfect after scaling, with ECE still at 0.3566.

The appropriate conclusion is:

> **Post-hoc temperature scaling partially improved probability calibration on held-out RSNA data, but it did not improve the underlying discrimination and did not remove the calibration problem.**

## Important limitation

This is held-out calibration within the same external RSNA dataset. It should not be presented as evidence that calibration transfers across hospitals or across entirely different datasets.

The experiment also does not establish clinical usefulness or safety.

## Stage 3 conclusion

Temperature scaling provided a useful post-hoc correction to probability confidence, but it was not a solution to the broader uncertainty problem.

This motivated the next stage: directly examining confidence versus correctness and identifying high-confidence errors.
