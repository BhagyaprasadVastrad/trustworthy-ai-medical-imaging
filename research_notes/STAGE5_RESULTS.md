# Stage 5: Deep Ensemble Uncertainty

## Research question

> Does disagreement between independently trained models provide a useful signal for identifying uncertain or incorrect predictions under external distribution shift?

Stage 5 trains five independently initialized copies of the same SmallCNN architecture used in Stage 1.

## Experimental setup

All five models use the same:

- PneumoniaMNIST training data
- image preprocessing
- SmallCNN architecture
- binary classification objective
- optimizer and learning rate
- five-epoch training procedure

The only intentional difference is the random seed:

```text
42, 43, 44, 45, 46
```

Each model is evaluated on the same 25,684 RSNA studies.

For each study, the experiment records:

- mean ensemble probability
- prediction variance
- prediction standard deviation

Standard deviation is used as a simple model-disagreement proxy.

## Evaluation-alignment note

The notebook originally contained an ensemble-performance block that compared the ensemble predictions with `rsna_labels["label"]` in annotation order. The ensemble predictions were produced in DICOM-file order, so the reported ensemble values from that block (including AUROC 0.4938 and Brier score 0.4282) are **not treated as valid ensemble performance metrics**.

The later uncertainty block uses the inference-order `all_labels` array and reports an aligned accuracy of 0.4236. The corrected `src/stage5.py` now enforces `StudyInstanceUID` alignment before calculating ensemble AUROC, accuracy and Brier score.

## Ensemble output

The ensemble produced predictions for all 25,684 studies.

| Measure | Range |
|---|---:|
| Mean probability | 0.116930 to 0.986835 |
| Variance | 0.000174 to 0.073641 |
| Standard deviation | 0.013174 to 0.271369 |

## Correct versus incorrect predictions

| Prediction group | Count | Mean disagreement | Median disagreement |
|---|---:|---:|---:|
| Correct | 10,880 | 0.101513 | 0.104394 |
| Incorrect | 14,804 | 0.093461 | 0.091569 |

Incorrect predictions actually had slightly lower disagreement than correct predictions.

## Error-detection test

Treating ensemble standard deviation as an error-detection score produced:

```text
AUROC = 0.4225
```

An AUROC of 0.50 would represent no useful ranking signal.

The result of 0.4225 therefore indicates that ensemble disagreement did not reliably rank incorrect predictions above correct predictions in this experiment.

## Highest-disagreement subset

The 90th-percentile disagreement threshold was:

```text
0.142279
```

This selected 2,569 studies.

Their error rate was:

```text
62.36%
```

The overall error rate was:

```text
57.64%
```

The highest-disagreement group therefore had an error rate 4.72 percentage points higher than the overall dataset.

This is a limited association, but it was not strong or consistent enough to make disagreement a reliable standalone uncertainty measure.

## High-confidence and high-disagreement cases

Using:

- confidence >= 0.90
- top 10% disagreement

the experiment found:

```text
0 high-confidence + high-disagreement cases
```

The ensemble therefore did not identify a group of cases where the models strongly disagreed while the ensemble remained highly confident.

## Corrected reanalysis status

The corrected multi-seed Stage 5 implementation has been added under `src/`. It also compares disagreement with one-minus-max-probability and predictive entropy and computes bootstrap confidence intervals.

The corrected ensemble AUROC and Brier score have **not yet been rerun in this environment**, because the original RSNA DICOM data and ensemble checkpoints are not available here. They should be taken from the output of `python -m src.stage5`, not inferred from the invalid earlier block.

## Interpretation

Stage 5 is best treated as a mixed or negative result.

There was a small increase in error rate among the highest-disagreement cases, but the full error-detection AUROC was only 0.4225 and incorrect predictions had lower average disagreement.

The conclusion is:

> **Deep-ensemble disagreement showed a limited association with errors in a high-disagreement subset, but it was not a reliable standalone uncertainty signal across the full external evaluation set.**

This negative result is retained because it is scientifically useful. An uncertainty method should not be considered successful merely because it produces a numerical disagreement score.

## Limitations

The ensemble contains only five small CNNs and evaluates one external dataset on a binary image-only task.

Prediction standard deviation is a model-disagreement proxy, not a complete decomposition of predictive uncertainty.

The RSNA target represents lung opacity, while the original model was trained on PneumoniaMNIST for pneumonia classification. These targets are related but not identical.

## Next direction

The results motivate comparing model disagreement with other approaches to uncertainty and reliability, including additional calibration methods and uncertainty-aware evaluation under more realistic distribution shifts.
