# Stage 1 setup

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

## Stage 1 dataset

The first experiment uses **PneumoniaMNIST**, a binary chest X-ray classification benchmark from MedMNIST.

The choice is deliberate: a binary task makes the first calibration experiment easier to inspect while keeping the work within the medical-imaging setting of the broader research interest.

The repository does not store the dataset. The notebook downloads it through the MedMNIST package.

## Research split rule

The test set must remain untouched during model fitting and calibration.

- Training set: fit the model.
- Validation set: model selection and temperature scaling.
- Test set: final evaluation.

Temperature scaling must be fitted using validation logits only.

## Controlled shift

Stage 1 uses an artificial contrast change on test images as a simple stress test.

This should not be described as evidence of real hospital-to-hospital distribution shift. Later stages will use more realistic external-domain evaluation.
