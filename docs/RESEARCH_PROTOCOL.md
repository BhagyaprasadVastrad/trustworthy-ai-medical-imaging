# Corrected research protocol

This document defines the analysis to use for the next rerun. It is intentionally written before the corrected numerical results are available so that the analysis is not changed to favour a preferred outcome.

## Primary question

Does predictive performance, probability calibration, confidence and uncertainty remain reliable when a medical-image classifier is evaluated outside its original training distribution?

## Primary evaluation dimensions

1. **Discrimination:** AUROC and average precision.
2. **Classification at a fixed threshold:** accuracy, precision, recall and F1 at 0.50.
3. **Probability quality:** Brier score, log loss and Expected Calibration Error using the repository's fixed binning implementation.
4. **Confidence/error relationship:** error-detection AUROC for confidence-derived scores and prevalence of high-confidence errors.
5. **Uncertainty:** compare ensemble disagreement against one minus maximum probability and predictive entropy.

## Repeated training

The corrected study is intended to use seeds:

\`42, 43, 44, 45, 46\`

The same architecture, preprocessing and evaluation protocol should be applied to every seed. Results should be reported per seed and as an aggregate summary. A single seed should not be presented as representative of the full experiment.

## External evaluation

The RSNA experiment must remain labelled as an **external-dataset evaluation**, not a hospital-to-hospital validation.

PneumoniaMNIST and RSNA have different target definitions and populations. The RSNA target represents lung opacity, whereas the model was trained on PneumoniaMNIST pneumonia labels. Any performance change therefore has multiple possible contributors and should not be attributed to a single causal mechanism.

## Calibration

Temperature scaling must be fitted only on a calibration subset and evaluated on a separate held-out subset.

The calibration data must not also be used to report the final held-out calibration metrics.

## Alignment rule

Predictions and labels must be joined using \`StudyInstanceUID\`.

Row order, DICOM directory order and annotation-file order must never be used as implicit alignment.

The pipeline should fail loudly on duplicate study identifiers, missing identifiers, extra identifiers, non-binary labels, and probabilities outside [0, 1].

## Uncertainty comparison

No uncertainty method should be declared successful from one metric alone.

For each uncertainty score, report:

- error-detection AUROC,
- bootstrap 95% confidence interval,
- error rate in the highest 10% uncertainty group,
- overall error rate,
- and comparison with simple baselines.

A negative result is retained as a result.

## Statistical reporting

Bootstrap confidence intervals should use a fixed random seed and be reported alongside point estimates.

The final report should distinguish:

- **recorded notebook results** from
- **corrected multi-seed rerun results**.

No new numerical result should be described as reproduced until the complete corrected workflow has actually executed.

## Claims boundary

This repository is a research-preparation study. It does not establish clinical safety, clinical usefulness, hospital-level generalisation, or a validated pneumonia detector.

The intended scientific contribution at this stage is methodological: documenting how calibration, confidence and uncertainty can fail under dataset change and establishing a reproducible framework for studying those failures more rigorously.
