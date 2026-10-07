# Final execution guide

This is the only experiment you need to run to finish the corrected quantitative analysis.

## 1. Required local data

Keep the RSNA dataset outside GitHub. The expected structure is:

\`\`\`text
D:\0001_Kings_AI_Project\
├── datasets\
│   └── rsna_pneumonia\
│       ├── annotations\
│       │   └── pneumonia-challenge-annotations-original_2018.json
│       └── raw\
│           └── images\
│               └── ... 25,684 DICOM files ...
└── trustworthy-ai-medical-imaging\
\`\`\`

If your RSNA folder is somewhere else, that is fine. Pass its actual path to \`--rsna-root\`.

## 2. Open PowerShell

Go to the repository:

\`\`\`powershell
cd D:\0001_Kings_AI_Project\trustworthy-ai-medical-imaging
\`\`\`

Activate your virtual environment if you have one:

\`\`\`powershell
.venv\Scripts\Activate.ps1
\`\`\`

Install/update the dependencies:

\`\`\`python
python -m pip install -r requirements.txt
\`\`\`

Run the unit tests first:

\`\`\`powershell
pytest -q
\`\`\`

You should see the repository tests pass before starting the expensive run.

## 3. Run the complete corrected experiment

Run:

\`\`\`powershell
python scripts\final_reanalysis.py --rsna-root "D:\0001_Kings_AI_Project\datasets\rsna_pneumonia"
\`\`\`

**Do not change the seeds or epochs for the first final run.**

The script uses:

- seeds 42, 43, 44, 45, 46
- 5 training epochs per seed
- the original SmallCNN architecture
- Adam, learning rate 0.001
- batch size 64
- the same basic PneumoniaMNIST preprocessing
- identifier-based RSNA alignment using \`StudyInstanceUID\`
- separate calibration and held-out RSNA subsets
- temperature scaling
- confidence/error analysis
- five-model ensemble analysis
- bootstrap 95% confidence intervals
- uncertainty baselines

## 4. What the script produces

Everything is written to:

\`\`\`text
results/final_reanalysis/
\`\`\`

Important files:

\`\`\`text
checkpoints/
    smallcnn_seed_42.pt
    ...
    smallcnn_seed_46.pt

predictions/
    seed_42.csv
    ...
    seed_46.csv

training_history.csv
per_seed_external_metrics.csv
per_seed_calibration.csv
per_seed_confidence.csv
rsna_study_labels.csv
rsna_dicom_index.csv
ensemble_predictions.csv
ensemble_summary.json
\`\`\`

The most important files for me to review afterwards are:

1. \`per_seed_external_metrics.csv\`
2. \`per_seed_calibration.csv\`
3. \`per_seed_confidence.csv\`
4. \`ensemble_summary.json\`
5. \`ensemble_predictions.csv\`

## 5. Do NOT upload the following to GitHub

Do not commit:

- RSNA DICOM images
- RSNA annotation files
- model checkpoints
- \`rsna_dicom_index.csv\`
- \`ensemble_predictions.csv\`
- any other file containing medical-image-derived prediction records

The final repository should contain code, documentation and aggregate research results, not raw medical data or patient/study-level prediction artifacts.

## 6. When it finishes

Send me the contents of:

\`\`\`text
results/final_reanalysis/per_seed_external_metrics.csv
results/final_reanalysis/per_seed_calibration.csv
results/final_reanalysis/per_seed_confidence.csv
results/final_reanalysis/ensemble_summary.json
\`\`\`

You can simply paste the terminal output and those four files here.

I will then do the final scientific pass:

- check whether the run is valid,
- calculate the seed-level summary,
- compare against the original recorded results,
- interpret the uncertainty findings,
- update the research notes,
- update \`FINAL_RESEARCH_SUMMARY.md\`,
- update the README,
- generate the final tables/figures,
- and make the repository presentation-ready.

## Important

If the script stops with an error, **do not delete anything and do not start changing the code yourself**.

Copy the complete error message here. We will fix the exact problem and continue.
