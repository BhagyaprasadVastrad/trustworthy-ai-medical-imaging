# Reanalysis utilities

Reusable, testable components for the corrected reliability analysis.

## What is corrected

- Prediction/label alignment is performed by \`StudyInstanceUID\`, never by file or annotation order.
- Discrimination and probability-quality metrics are centralised in \`metrics.py\`.
- Bootstrap confidence intervals are available for AUROC, average precision, Brier score, log loss and Expected Calibration Error.
- Temperature scaling is implemented independently of the notebooks.
- Stage 5 compares ensemble disagreement with two simple baselines: one minus maximum probability and predictive entropy.

## Status

The corrected workflow is **implemented but not claimed as executed** until the required RSNA DICOM data and original model checkpoints are available locally.

Existing headline results remain the recorded notebook results. They are not silently replaced by unverified reruns.

## Stage 5 example

\`\`\`bash
python -m src.stage5 \
  --labels data/rsna/labels.csv \
  --predictions \
    data/predictions/seed42.csv \
    data/predictions/seed43.csv \
    data/predictions/seed44.csv \
    data/predictions/seed45.csv \
    data/predictions/seed46.csv
\`\`\`

Each prediction file must contain \`StudyInstanceUID\` and \`probability\`. The label file must contain \`StudyInstanceUID\` and \`label\`.
