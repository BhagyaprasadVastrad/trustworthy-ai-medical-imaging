# Project setup

## Software

Recommended Python versions: 3.11 to 3.13.

Install the dependencies with:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start Jupyter with:

```bash
jupyter notebook
```

## Datasets

### Stage 1: PneumoniaMNIST

Stage 1 uses **PneumoniaMNIST**, a binary chest X-ray classification benchmark from MedMNIST.

The repository does not store the dataset. The notebook downloads it through the MedMNIST package.

### Stage 2 onward: RSNA external evaluation

The RSNA Pneumonia Detection Challenge data are not redistributed in this repository.

The external evaluation requires:

- RSNA DICOM images
- the corresponding annotation JSON
- `pydicom` for DICOM reading

The local experiment used the RSNA data separately from the repository.

## Research split rules

The experiments keep model fitting separate from final evaluation wherever applicable.

For Stage 1:

- Training set: fit the model.
- Validation set: model selection during the baseline run.
- Test set: final evaluation and the controlled contrast-shift experiment.

For Stage 3, the Stage 2 RSNA predictions were divided into a calibration subset and a separate held-out evaluation subset. The temperature parameter was fitted on the calibration subset only and then evaluated on the held-out subset.

## Reproducibility

The repository now includes a `src/` package for the corrected reanalysis. The source workflow uses five seeds (42, 43, 44, 45, 46), validation-loss early stopping, basic training augmentation, identifier-based prediction/label alignment, uncertainty baselines, and bootstrap confidence intervals.

The RSNA data remain external and are not stored in the repository. See `data/README.md` and `src/README.md` for the expected local layout and commands.

The executed notebooks remain preserved as the original experimental record.

## Scope

The controlled contrast transformation is an artificial stress test. It is not evidence of real hospital-to-hospital distribution shift.

The RSNA experiment is an external-dataset evaluation. It should not be described as the planned cross-hospital MIMIC-CXR to CheXpert experiment.

The project is a research preparation study, not a clinical validation system.
