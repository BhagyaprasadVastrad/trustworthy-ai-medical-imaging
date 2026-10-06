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
- Validation set: model selection and temperature scaling.
- Test set: final evaluation.

For Stage 3, the RSNA predictions were divided into a calibration subset and a separate held-out evaluation subset. The temperature parameter was fitted on the calibration subset only.

## Reproducibility

The project uses a deliberately small convolutional neural network so the initial experiments can be reproduced on CPU hardware.

Stage 5 uses five independent random seeds:

```text
42, 43, 44, 45, 46
```

The Stage 2 prediction file is reused in Stage 4 so that confidence analysis is based on the same validated external predictions rather than rerunning inference.

## Scope

The controlled contrast transformation is an artificial stress test. It is not evidence of real hospital-to-hospital distribution shift.

The RSNA experiment is an external-dataset evaluation. It should not be described as the planned cross-hospital MIMIC-CXR to CheXpert experiment.

The project is a research preparation study, not a clinical validation system.
