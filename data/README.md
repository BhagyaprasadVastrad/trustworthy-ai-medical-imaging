# External data layout

Medical datasets are intentionally not committed to this repository.

## Expected local layout

\`\`\`text
data/
└── rsna/
    ├── labels.csv
    └── images/
        └── ... DICOM files ...
\`\`\`

\`labels.csv\` should contain one row per study:

\`\`\`text
StudyInstanceUID,label
...
\`\`\`

Prediction artifacts for reanalysis can be stored under \`data/predictions/\`:

\`\`\`text
StudyInstanceUID,probability
...
\`\`\`

Do not commit RSNA DICOM images, annotation files, model checkpoints, or other restricted/raw medical data.
