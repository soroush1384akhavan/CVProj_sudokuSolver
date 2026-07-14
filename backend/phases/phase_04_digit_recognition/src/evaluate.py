# for run : python -m phases.phase_04_digit_recognition.src.evaluate --language en

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.config import settings
from .dataset import build_eval_only_datasets, get_phase4_config
from .metrics import ClassificationMetrics, compute_metrics
from .model import build_model
from .utils import latest_checkpoint_path, latest_run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate digit recognition model."
    )

    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "fa"],
        default="en",
        help="Language to evaluate: en or fa. Default: en",
    )

    return parser.parse_args()


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[list[int], list[int]]:
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, leave=False, desc="predict"):
        images = images.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(preds)

    return y_true, y_pred


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()

    images = images.to(device)
    logits = model(images)
    preds = torch.argmax(logits, dim=1).cpu()

    return preds


def evaluate_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_classes: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
) -> ClassificationMetrics:
    y_true, y_pred = predict_dataset(
        model,
        dataset,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return compute_metrics(
        y_true,
        y_pred,
        num_classes=num_classes,
    )


def evaluate_multiple_datasets(
    model: torch.nn.Module,
    datasets: dict[str, Dataset],
    device: torch.device,
    num_classes: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, ClassificationMetrics]:
    results: dict[str, ClassificationMetrics] = {}

    for name, dataset in datasets.items():
        print(f"\nEvaluating: {name}")
        print(f"Samples: {len(dataset)}")

        results[name] = evaluate_dataset(
            model,
            dataset,
            device,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        print(f"Accuracy: {results[name].accuracy:.4f}")

    return results


def print_confusion_matrix(
    cm: list[list[int]],
    labels: list[str] | None = None,
) -> None:
    num_classes = len(cm)
    labels = labels or [str(i) for i in range(num_classes)]

    header = "      " + " ".join(f"{label:>4}" for label in labels)
    print(header)

    for index, row in enumerate(cm):
        row_str = " ".join(f"{value:>4}" for value in row)
        print(f"{labels[index]:>4}: {row_str}")


def show_confusion_matrix(
    cm: list[list[int]],
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> None:
    matrix = torch.tensor(cm).cpu().numpy()

    num_classes = len(cm)
    labels = labels or [str(i) for i in range(num_classes)]

    figure, axis = plt.subplots(figsize=(8, 7))

    matrix_image = axis.imshow(
        matrix,
        interpolation="nearest",
    )

    figure.colorbar(matrix_image, ax=axis)

    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")

    axis.set_xticks(range(num_classes))
    axis.set_yticks(range(num_classes))

    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)

    max_value = matrix.max() if matrix.size > 0 else 0
    threshold = max_value / 2

    for row in range(num_classes):
        for column in range(num_classes):
            value = int(matrix[row, column])

            axis.text(
                column,
                row,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    plt.show()


def main():
    args = parse_args()
    language = args.language

    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    data_cfg = phase_cfg.get("data", {})

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Evaluation language: {language}")

    num_classes = int(
        model_cfg.get("num_classes", 10)
    )

    dropout = float(
        model_cfg.get("dropout", 0.5)
    )

    batch_size = int(
        phase_cfg.get("training", {}).get("batch_size", 64)
    )

    num_workers = int(
        data_cfg.get("num_workers", 0)
    )

    model = build_model(
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    configured_model_path = settings.get(
        f"digit_recognition.model_paths.{language}",
        None,
    )

    if configured_model_path:
        checkpoint_path = settings.resolve_path(
            str(configured_model_path)
        )
        run_dir = checkpoint_path.parent
    else:
        print(
            f"No configured model path found for language={language}. "
            "Using latest checkpoint."
        )

        run_dir = latest_run_dir()
        checkpoint_path = latest_checkpoint_path()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model not found for language={language}: "
            f"{checkpoint_path}"
        )

    print(f"Model directory: {run_dir}")
    print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    per_dataset_eval_sets = build_eval_only_datasets(
        languages=(language,),
    )

    if not per_dataset_eval_sets:
        raise ValueError(
            f"No evaluation datasets found for language={language}."
        )

    print("\nEvaluation datasets:")

    for name, dataset in per_dataset_eval_sets.items():
        print(f"  {name}: {len(dataset)}")

    all_metrics = evaluate_multiple_datasets(
        model,
        per_dataset_eval_sets,
        device,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    class_labels = [str(index) for index in range(num_classes)]

    for name, metrics in all_metrics.items():
        print(f"\n=== {name} ===")
        print(f"Accuracy: {metrics.accuracy:.4f}")

        print("\nConfusion Matrix:")
        print_confusion_matrix(
            metrics.confusion_matrix,
            labels=class_labels,
        )

        show_confusion_matrix(
            metrics.confusion_matrix,
            labels=class_labels,
            title=(
                f"Confusion Matrix - {name} - "
                f"language={language}"
            ),
        )


if __name__ == "__main__":
    main()