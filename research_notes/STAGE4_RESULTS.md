# Stage 4: Confidence and Error Analysis

## Research question

> Can a model's confidence be trusted when it is evaluated on an external dataset?

Stage 4 uses the predictions already generated during Stage 2. No new DICOM inference or model retraining is performed.

## External predictions

The analysis contains:

- 25,684 logits
- 25,684 probabilities
- 25,684 labels

The predictions are the same Stage 2 external RSNA predictions used in the earlier calibration analysis.

## Correct versus incorrect predictions

The single model produced:

| Measure | Result |
|---|---:|
| Correct predictions | 9,346 |
| Incorrect predictions | 16,338 |
| Accuracy | 0.3639 |

Mean and median confidence were:

| Prediction group | Mean confidence | Median confidence |
|---|---:|---:|
| Correct | 0.6384 | 0.6041 |
| Incorrect | 0.6817 | 0.6588 |

The incorrect predictions were therefore more confident on average than the correct predictions.

## High-confidence errors

Using confidence >= 0.90:

- Incorrect predictions: **874**
- Share of all predictions: **3.40%**
- Most confident incorrect prediction: approximately **0.9698**

This is important for trustworthy AI because the concern is not only that the model makes errors, but that some errors are accompanied by very high confidence.

## Confidence bins

The reliability-style analysis showed that empirical accuracy remained below average confidence across the populated confidence bins.

This indicates substantial overconfidence under the external evaluation.

## Effect of Stage 3 temperature scaling

The Stage 3 temperature was:

```text
T = 2.0350
```

Applying it to the Stage 2 predictions reduced confidence:

| Measure | Before | After |
|---|---:|---:|
| Mean confidence | 0.6659 | 0.5909 |
| Median confidence | 0.6369 | 0.5686 |
| Brier score | 0.3579 | 0.3003 |
| Log loss | 0.9652 | 0.8003 |
| AUROC | 0.5640 | 0.5640 |

The maximum calibrated probability was approximately 0.8573, so there were no predictions >= 0.90 after calibration.

This does **not** mean that the errors disappeared. The probabilities became less extreme.

## Interpretation

Stage 4 provides a direct example of why confidence and correctness should be analysed separately.

The model could be wrong while being highly confident. Temperature scaling reduced the extremity of the probabilities and improved Brier score and log loss, but it did not change discrimination.

The conclusion is therefore:

> **Calibration can make confidence less extreme and improve probability quality without turning an inaccurate external classifier into a better classifier.**

## Important limitation

Confidence from a single deterministic model is not a complete uncertainty estimate. This stage studies predictive confidence and error behaviour.

Stage 5 therefore investigates whether disagreement between independently trained models provides an additional uncertainty signal.
