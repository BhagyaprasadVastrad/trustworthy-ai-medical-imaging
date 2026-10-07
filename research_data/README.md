# Research data and prediction artifacts

The medical datasets and generated prediction arrays are intentionally not committed to this repository.

## Stage 2 artifact

After running `notebooks/02_rsna_external_evaluation.ipynb`, the local experiment produces:

```text
research_data/rsna_stage2_predictions.npz
```

The file contains:

- `logits`: raw model outputs
- `probabilities`: positive-class probabilities
- `labels`: study-level ground-truth labels constructed by the Stage 2 notebook

These arrays are the correct inputs for recalculating Stage 2 metrics, bootstrap confidence intervals, calibration analyses and later confidence/error analyses.

## Stage 5 artifact

The Stage 5 notebook constructs an in-memory array of five-member ensemble predictions and derives:

- ensemble mean probability
- ensemble variance
- ensemble standard deviation
- binary ensemble prediction

The current notebook records these derived results in its executed output but does not persist the ensemble prediction array as a repository artifact. This is a reproducibility gap and is intentionally documented rather than hidden.

Before treating Stage 5 as a reusable quantitative data source, the ensemble predictions should be exported from the notebook run to a local `.npz` artifact containing at least:

```text
probabilities  # shape: (5, N)
labels         # shape: (N,)
seeds           # [42, 43, 44, 45, 46]
```

## Data policy

Raw RSNA DICOM images and annotation files are not redistributed here. They must be obtained separately under the dataset's terms.

Generated prediction arrays may contain dataset-derived information and are therefore kept outside the public repository unless there is a clear reason and permission to distribute them.

## Reproducibility principle

Numbers in the README and research notes should only be promoted to final quantitative results when they can be traced either to an executed notebook output or to a persisted prediction artifact. Rounded summary values alone are not sufficient to reconstruct bootstrap confidence intervals.