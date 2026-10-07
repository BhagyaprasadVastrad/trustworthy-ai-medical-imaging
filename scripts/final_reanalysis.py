"""End-to-end corrected multi-seed reanalysis.

This script intentionally reproduces the original Stage 1 training protocol
(5 epochs, Adam 1e-3, batch size 64) across seeds 42-46, then performs
identifier-aligned RSNA inference, held-out temperature scaling, confidence
analysis, and ensemble uncertainty analysis.

It does NOT download or redistribute RSNA data.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import PneumoniaMNIST

from src.calibration import fit_temperature, temperature_scale
from src.metrics import bootstrap_metric_ci, expected_calibration_error


SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 64
RSNA_BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3


class SmallCNN(nn.Module):
    """Exact SmallCNN architecture used by the recorded Stage 1 experiment."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1)).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(expected_calibration_error(y, p)),
    }


def train_model(seed: int, device: torch.device, output_dir: Path, epochs: int) -> tuple[SmallCNN, list[dict]]:
    set_seed(seed)
    transform = transforms.ToTensor()
    train_ds = PneumoniaMNIST(split="train", download=True, transform=transform, size=64)
    val_ds = PneumoniaMNIST(split="val", download=True, transform=transform, size=64)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).reshape(-1)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * images.size(0)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).reshape(-1)
                loss = criterion(model(images), labels)
                val_total += loss.item() * images.size(0)

        row = {
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_total / len(train_loader.dataset),
            "val_loss": val_total / len(val_loader.dataset),
        }
        history.append(row)
        print(
            f"[seed {seed}] epoch {epoch + 1}/{epochs} "
            f"train={row['train_loss']:.4f} val={row['val_loss']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"smallcnn_seed_{seed}.pt"
    torch.save(model.state_dict(), checkpoint)
    print(f"[seed {seed}] checkpoint saved: {checkpoint}")
    return model, history


def build_rsna_labels(annotation_path: Path) -> pd.DataFrame:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = {
        item["id"]: item["name"]
        for item in data["labelGroups"][0]["labels"]
    }
    annotations = data["datasets"][0]["annotations"]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation["StudyInstanceUID"]].append(annotation)

    rows = []
    for study_uid, records in grouped.items():
        names = {label_map[r["labelId"]] for r in records}
        if "Lung Opacity" in names:
            label = 1
            name = "Lung Opacity"
        elif "No Lung Opacity" in names:
            label = 0
            name = "No Lung Opacity"
        else:
            continue
        rows.append(
            {
                "StudyInstanceUID": study_uid,
                "label": label,
                "label_name": name,
            }
        )

    labels = pd.DataFrame(rows)
    if labels.empty:
        raise RuntimeError("No RSNA study-level labels were created.")
    if labels["StudyInstanceUID"].duplicated().any():
        raise RuntimeError("Duplicate StudyInstanceUIDs in RSNA labels.")
    if set(labels["label"].unique()) - {0, 1}:
        raise RuntimeError("RSNA labels are not binary.")
    return labels


def discover_dicom_table(dicom_root: Path, labels: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(dicom_root.rglob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No .dcm files found under {dicom_root}")

    lookup = dict(zip(labels["StudyInstanceUID"], labels["label"]))
    rows = []
    for i, path in enumerate(paths, 1):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        uid = getattr(ds, "StudyInstanceUID", None)
        sop = getattr(ds, "SOPInstanceUID", None)
        if uid in lookup:
            rows.append(
                {
                    "dicom_path": str(path),
                    "StudyInstanceUID": uid,
                    "SOPInstanceUID": sop,
                    "label": int(lookup[uid]),
                }
            )
        if i % 2000 == 0 or i == len(paths):
            print(f"Read DICOM metadata: {i:,}/{len(paths):,}")

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No DICOM images matched the RSNA labels.")
    if table["StudyInstanceUID"].duplicated().any():
        counts = table["StudyInstanceUID"].value_counts()
        examples = counts[counts > 1].head().to_dict()
        raise RuntimeError(
            "More than one DICOM matched a StudyInstanceUID. "
            "The original experiment assumed one image per study. "
            f"Examples: {examples}"
        )
    if table["SOPInstanceUID"].duplicated().any():
        raise RuntimeError("Duplicate SOPInstanceUIDs detected.")
    return table.sort_values("StudyInstanceUID").reset_index(drop=True)


RSNA_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def load_rsna_image(path: str) -> torch.Tensor:
    ds = pydicom.dcmread(path)
    image = Image.fromarray(ds.pixel_array, mode="L")
    return RSNA_TRANSFORM(image)


class RSNADataset(Dataset):
    def __init__(self, table: pd.DataFrame):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        return (
            load_rsna_image(row["dicom_path"]),
            int(row["label"]),
            row["StudyInstanceUID"],
        )


def predict_rsna(model: nn.Module, table: pd.DataFrame, device: torch.device):
    dataset = RSNADataset(table)
    loader = DataLoader(
        dataset, batch_size=RSNA_BATCH_SIZE, shuffle=False, num_workers=0
    )
    probabilities = []
    labels = []
    uids = []

    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels, batch_uids in loader:
            logits = model(batch_images.to(device))
            p = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(p.tolist())
            labels.extend(batch_labels.numpy().astype(int).tolist())
            uids.extend(list(batch_uids))

    return (
        pd.DataFrame(
            {
                "StudyInstanceUID": uids,
                "label": labels,
                "probability": probabilities,
            }
        )
        .sort_values("StudyInstanceUID")
        .reset_index(drop=True)
    )


def calibration_analysis(df: pd.DataFrame, seed: int) -> dict[str, float]:
    train, heldout = train_test_split(
        df,
        test_size=0.5,
        stratify=df["label"],
        random_state=42,
    )
    temperature = fit_temperature(
        train["label"].to_numpy(),
        train["probability"].to_numpy(),
    )
    p_before = heldout["probability"].to_numpy()
    p_after = temperature_scale(p_before, temperature)
    y = heldout["label"].to_numpy()

    return {
        "seed": seed,
        "temperature": float(temperature),
        "heldout_brier_before": float(brier_score_loss(y, p_before)),
        "heldout_brier_after": float(brier_score_loss(y, p_after)),
        "heldout_log_loss_before": float(log_loss(y, p_before, labels=[0, 1])),
        "heldout_log_loss_after": float(log_loss(y, p_after, labels=[0, 1])),
        "heldout_ece_before": float(expected_calibration_error(y, p_before)),
        "heldout_ece_after": float(expected_calibration_error(y, p_after)),
        "heldout_auroc_before": float(roc_auc_score(y, p_before)),
        "heldout_auroc_after": float(roc_auc_score(y, p_after)),
    }


def confidence_analysis(df: pd.DataFrame, seed: int) -> dict[str, float]:
    y = df["label"].to_numpy()
    p = df["probability"].to_numpy()
    pred = (p >= 0.5).astype(int)
    correct = pred == y
    confidence = np.maximum(p, 1 - p)
    errors = ~correct
    high_error = errors & (confidence >= 0.90)

    return {
        "seed": seed,
        "n": int(len(df)),
        "accuracy": float(correct.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_incorrect": float(confidence[errors].mean()),
        "median_confidence_correct": float(np.median(confidence[correct])),
        "median_confidence_incorrect": float(np.median(confidence[errors])),
        "high_confidence_errors_ge_0_90": int(high_error.sum()),
        "high_confidence_error_rate_all": float(high_error.mean()),
    }


def bootstrap_error_auc(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    error = (score >= 0.5).astype(int)
    value, lo, hi = bootstrap_metric_ci(
        error, score, metric="auroc", n_bootstrap=2000, seed=42
    )
    return {"auroc": float(value), "ci_low": float(lo), "ci_high": float(hi)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsna-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/final_reanalysis"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    project_output = args.output.resolve()
    checkpoint_dir = project_output / "checkpoints"
    prediction_dir = project_output / "predictions"
    project_output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = args.rsna_root / "annotations" / "pneumonia-challenge-annotations-original_2018.json"
    dicom_root = args.rsna_root / "raw" / "images"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    if not dicom_root.exists():
        raise FileNotFoundError(f"Missing DICOM directory: {dicom_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("RSNA root:", args.rsna_root.resolve())

    labels = build_rsna_labels(annotation_path)
    labels.to_csv(project_output / "rsna_study_labels.csv", index=False)
    print(f"Study labels: {len(labels):,}")

    dicom_table = discover_dicom_table(dicom_root, labels)
    dicom_table.to_csv(project_output / "rsna_dicom_index.csv", index=False)
    print(f"Aligned DICOM studies: {len(dicom_table):,}")

    seed_metrics = []
    calibration_rows = []
    confidence_rows = []
    histories = []

    prediction_files = []

    for seed in args.seeds:
        model, history = train_model(seed, device, checkpoint_dir, args.epochs)
        histories.extend(history)

        pred_df = predict_rsna(model, dicom_table, device)
        pred_path = prediction_dir / f"seed_{seed}.csv"
        pred_df.to_csv(pred_path, index=False)
        prediction_files.append(pred_path)

        m = metrics(pred_df["label"].to_numpy(), pred_df["probability"].to_numpy())
        m["seed"] = seed
        seed_metrics.append(m)

        calibration_rows.append(calibration_analysis(pred_df, seed))
        confidence_rows.append(confidence_analysis(pred_df, seed))
        print(f"[seed {seed}] external metrics: {m}")

    pd.DataFrame(histories).to_csv(project_output / "training_history.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(project_output / "per_seed_external_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(project_output / "per_seed_calibration.csv", index=False)
    pd.DataFrame(confidence_rows).to_csv(project_output / "per_seed_confidence.csv", index=False)

    # Identifier-aligned ensemble.
    merged = labels[["StudyInstanceUID", "label"]].copy()
    for path in prediction_files:
        frame = pd.read_csv(path)
        if frame["StudyInstanceUID"].duplicated().any():
            raise RuntimeError(f"Duplicate IDs in {path}")
        merged = merged.merge(
            frame[["StudyInstanceUID", "probability"]].rename(
                columns={"probability": path.stem}
            ),
            on="StudyInstanceUID",
            how="inner",
            validate="one_to_one",
        )

    prediction_cols = [c for c in merged.columns if c not in {"StudyInstanceUID", "label"}]
    matrix = merged[prediction_cols].to_numpy(float)
    y = merged["label"].to_numpy(int)

    mean_p = matrix.mean(axis=1)
    disagreement = matrix.std(axis=1)
    max_p = matrix.max(axis=1)
    mean_p_clip = np.clip(mean_p, 1e-7, 1 - 1e-7)
    entropy = -(mean_p_clip * np.log(mean_p_clip) + (1 - mean_p_clip) * np.log(1 - mean_p_clip))
    error = ((mean_p >= 0.5).astype(int) != y).astype(int)

    uncertainty = {}
    for name, score in {
        "ensemble_disagreement": disagreement,
        "one_minus_max_probability": 1 - max_p,
        "predictive_entropy": entropy,
    }.items():
        uncertainty[name] = bootstrap_error_auc(error, score)

    threshold = float(np.quantile(disagreement, 0.90))
    top = disagreement >= threshold

    ensemble_summary = {
        "n_models": len(args.seeds),
        "n_studies": int(len(merged)),
        "ensemble_metrics": metrics(y, mean_p),
        "overall_error_rate": float(error.mean()),
        "uncertainty_error_detection": uncertainty,
        "top_10_percent_disagreement_threshold": threshold,
        "top_10_percent_disagreement_count": int(top.sum()),
        "top_10_percent_disagreement_error_rate": float(error[top].mean()),
    }

    pd.DataFrame(
        {
            "StudyInstanceUID": merged["StudyInstanceUID"],
            "label": y,
            "ensemble_probability": mean_p,
            "ensemble_disagreement": disagreement,
            "one_minus_max_probability": 1 - max_p,
            "predictive_entropy": entropy,
            "error": error,
        }
    ).to_csv(project_output / "ensemble_predictions.csv", index=False)

    (project_output / "ensemble_summary.json").write_text(
        json.dumps(ensemble_summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nFINAL REANALYSIS COMPLETE")
    print(json.dumps(ensemble_summary, indent=2))
    print(f"\nResults written to: {project_output}")


if __name__ == "__main__":
    main()
