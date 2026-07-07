from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataset import build_eval_only_datasets, get_phase4_config
from .metrics import ClassificationMetrics, compute_metrics
from .model import build_model
from .utils import latest_checkpoint_path, latest_run_dir


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[list[int], list[int]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
    y_true, y_pred = predict_dataset(model, dataset, device, batch_size=batch_size, num_workers=num_workers)
    return compute_metrics(y_true, y_pred, num_classes=num_classes)


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
        results[name] = evaluate_dataset(
            model, dataset, device,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        print(f"  accuracy = {results[name].accuracy:.4f}")

    return results


def print_confusion_matrix(cm: list[list[int]], labels: list[str] | None = None) -> None:
    num_classes = len(cm)
    labels = labels or [str(i) for i in range(num_classes)]

    header = "      " + " ".join(f"{l:>4}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>4}" for v in row)
        print(f"{labels[i]:>4}: {row_str}")


def main():
    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = int(model_cfg.get("num_classes", 10))
    dropout = float(model_cfg.get("dropout", 0.5))

    model = build_model(num_classes=num_classes, dropout=dropout).to(device)

    run_dir = latest_run_dir()
    checkpoint_path = latest_checkpoint_path()

    print(f"Latest run: {run_dir}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    per_dataset_eval_sets = build_eval_only_datasets()

    all_metrics = evaluate_multiple_datasets(
        model, per_dataset_eval_sets, device, num_classes=num_classes
    )

    for name, metrics in all_metrics.items():
        print(f"\n=== {name} ===")
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print_confusion_matrix(metrics.confusion_matrix)


if __name__ == "__main__":
    main()