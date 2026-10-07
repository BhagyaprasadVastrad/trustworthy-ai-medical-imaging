from pathlib import Path

def find_project_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "notebooks").exists():
            return candidate
    return here

PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
RSNA_ROOT = DATA_ROOT / "rsna_pneumonia"
MODEL_DIR = PROJECT_ROOT / "models"
RESEARCH_DATA_DIR = PROJECT_ROOT / "research_data"
RSNA_DICOM_ROOT = RSNA_ROOT / "raw" / "images"
RSNA_ANNOTATION_PATH = RSNA_ROOT / "annotations" / "pneumonia-challenge-annotations-original_2018.json"

def ensure_output_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
