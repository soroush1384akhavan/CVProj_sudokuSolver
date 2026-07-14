from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from app.config import settings

from .dataset import DigitTransform, build_digit_dataloaders, get_phase4_config
from .datasets.fromsoduko.charls_soduku_dataloader import DatSudokuCellDataset
from .datasets.fromsoduko.generated_dataloader import GeneratedSudokuCellDataset
from .model import build_model
from .train import train_model, validate
from .utils import new_run_dir, save_json, save_training_plots, set_seed

from phases.phase_03_cell_extraction.cell_extraction import is_cell_empty

from torchvision.utils import save_image
from bisect import bisect_right


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _resolve_path(value: str | Path) -> Path:
    return settings.resolve_path(str(value))


def freeze_batchnorm(model: nn.Module) -> nn.Module:

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train_with_frozen_bn(self, mode: bool = True):
        nn.Module.train(self, mode)
        if mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    model.train = types.MethodType(train_with_frozen_bn, model)
    return model


# ---------------------------------------------------------------------------
# Easy-empty-cell filtering
#
# The "empty" class massively outnumbers digit classes in Sudoku cell data.
# Training on ALL of it makes the imbalance worse without teaching the model
# anything new, since `is_cell_empty` (the existing heuristic) already gets
# the easy majority of empty cells right on its own. What actually needs a
# learned model is the cases where the heuristic is WRONG — e.g. glare/noise
# that fools it into thinking an empty cell has a digit. So for training we
# keep only empty-labeled samples where `is_cell_empty` returns False (a
# miss); every other (non-empty) sample is kept untouched.
# ---------------------------------------------------------------------------

class FilteredCellDataset(Dataset):
    """
    Thin index-remapping wrapper around a base dataset that also exposes a
    `.samples` list (subset of the base dataset's), so helpers that expect
    direct `.samples` access (`_board_key`, `save_misclassified_samples`)
    keep working the same way they do for the unfiltered dataset classes.
    """

    def __init__(self, base_dataset: Dataset, keep_indices: list[int]) -> None:
        self.base_dataset = base_dataset
        self.keep_indices = keep_indices

        base_samples = getattr(base_dataset, "samples", None)
        if isinstance(base_samples, list):
            self.samples = [base_samples[i] for i in keep_indices]

    def __len__(self) -> int:
        return len(self.keep_indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.keep_indices[index]]


def _get_sample_label_and_path(dataset: Dataset, index: int) -> tuple[int, Path]:
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, list) or index >= len(samples):
        raise AttributeError(
            f"{type(dataset).__name__} has no indexable `.samples` list; "
            "cannot filter easy-empty cells without per-sample metadata. "
            "Adjust _get_sample_label_and_path to match your dataset's sample structure."
        )

    sample = samples[index]

    cell_path = getattr(sample, "cell_path", None)
    label = getattr(sample, "label", None)
    if label is None:
        label = getattr(sample, "digit", None)

    if isinstance(sample, (tuple, list)):
        if cell_path is None and len(sample) > 0:
            cell_path = sample[0]
        if label is None and len(sample) > 1:
            label = sample[1]

    if cell_path is None or label is None:
        raise ValueError(
            f"Could not resolve (cell_path, label) for sample index {index} "
            f"in {type(dataset).__name__}; adjust attribute names in "
            "_get_sample_label_and_path to match your dataset."
        )

    return int(label), Path(cell_path)


def _load_cell_image_for_empty_check(cell_path: Path) -> np.ndarray:
    image = cv2.imread(str(cell_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read cell image for empty-check: {cell_path}")
    return image


def filter_easy_empty_cells(
    dataset: Dataset,
    empty_label: int = 0,
) -> FilteredCellDataset:
    """
    Keep every digit sample.

    From ground-truth empty cells, keep only the hard-empty samples that
    `is_cell_empty` sees as non-empty. Easy empty cells already handled by
    the heuristic are removed from the fine-tuning dataset.

    Cached cell images are already in the same polarity expected by
    `is_cell_empty`: bright content on a dark background.
    """
    keep_indices: list[int] = []

    kept_digits = 0
    kept_hard_empty = 0
    dropped_easy_empty = 0

    for index in range(len(dataset)):
        label, cell_path = _get_sample_label_and_path(dataset, index)

        if label != empty_label:
            keep_indices.append(index)
            kept_digits += 1
            continue

        cell_image = _load_cell_image_for_empty_check(cell_path)
        heuristic_says_empty = is_cell_empty(inverted_gray=cell_image)

        if heuristic_says_empty:
            dropped_easy_empty += 1
            continue

        keep_indices.append(index)
        kept_hard_empty += 1

    print(
        f"[{type(dataset).__name__}] empty filtering:\n"
        f"  total input:        {len(dataset)}\n"
        f"  kept digits:        {kept_digits}\n"
        f"  kept hard empty:    {kept_hard_empty}\n"
        f"  dropped easy empty: {dropped_easy_empty}\n"
        f"  total kept:         {len(keep_indices)}"
    )

    return FilteredCellDataset(dataset, keep_indices)


def _build_sudoku_sources(
    transform: DigitTransform,
    sudoku_cfg: dict[str, Any],
    refresh_cache: bool,
    filter_easy_empty: bool = False,
) -> tuple[list[Dataset], dict[str, int]]:
    cache_root = _resolve_path(sudoku_cfg.get("cache_dir", "storage/sudoku/cell_cache"))
    include_empty_cells = bool(sudoku_cfg.get("include_empty_cells", False))
    empty_label = int(sudoku_cfg.get("empty_label", 0))

    if filter_easy_empty and not include_empty_cells:
        raise ValueError(
            "digit_recognition.data.sudoku_cells.include_empty_cells must be true "
            "to collect hard empty cells missed by is_cell_empty."
        )

    datasets: list[Dataset] = []
    summary: dict[str, int] = {}

    for root in _as_list(sudoku_cfg.get("generated_roots")):
        root_path = _resolve_path(root)

        if not root_path.is_dir():
            print(f"Warning: generated Sudoku dataset not found: {root_path}")
            continue

        dataset = GeneratedSudokuCellDataset(
            root_dir=root_path,
            cache_dir=cache_root / "generated",

            languages=("en",),
            strict_languages=True,

            include_empty_cells=include_empty_cells,
            transform=transform,
            refresh_cache=refresh_cache,

            skip_visually_empty_nonzero=True,
            min_content_area_ratio=0.006,
            min_component_area_ratio=0.0015,

            return_language=False,
            return_metadata=False,
        )

        print(
            f"Generated Sudoku dataset loaded: "
            f"language=en | samples={len(dataset)} | root={root_path}"
        )

        summary[f"generated:{root_path.name}:raw"] = len(dataset)

        if filter_easy_empty and include_empty_cells:
            dataset = filter_easy_empty_cells(dataset, empty_label=empty_label)
            summary[f"generated:{root_path.name}"] = len(dataset)
        else:
            summary[f"generated:{root_path.name}"] = len(dataset)

        datasets.append(dataset)

    for root in _as_list(sudoku_cfg.get("dat_roots")):
        root_path = _resolve_path(root)
        if not root_path.is_dir():
            print(f"Warning: DAT Sudoku dataset not found: {root_path}")
            continue

        dataset = DatSudokuCellDataset(
            root_dir=root_path,
            cache_dir=cache_root / "dat" / root_path.name,
            include_empty_cells=include_empty_cells,
            transform=transform,
            refresh_cache=refresh_cache,
        )

        summary[f"dat:{root_path.name}:raw"] = len(dataset)

        if filter_easy_empty and include_empty_cells:
            dataset = filter_easy_empty_cells(dataset, empty_label=empty_label)
            summary[f"dat:{root_path.name}"] = len(dataset)
        else:
            summary[f"dat:{root_path.name}"] = len(dataset)

        datasets.append(dataset)

    if not datasets:
        raise ValueError("No Sudoku fine-tuning datasets found. Check generated_roots/dat_roots in config.")

    return datasets, summary

def _board_key(
    dataset: Dataset,
    source_index: int,
    local_index: int,
) -> str:

    samples = getattr(dataset, "samples", None)

    if isinstance(samples, list) and local_index < len(samples):
        sample = samples[local_index]

        cell_path = getattr(sample, "cell_path", None)

        if cell_path is None and isinstance(sample, (tuple, list)) and sample:
            cell_path = sample[0]

        if cell_path is not None:
            board_dir = Path(cell_path).parent.parent
            return f"src{source_index}:{board_dir.resolve()}"

    return f"src{source_index}:sample{local_index}"


def _split_by_board(sources: list[Dataset], validation_split: float, seed: int) -> tuple[list[int], list[int]]:
    total = sum(len(d) for d in sources)
    if total < 2:
        raise ValueError("Sudoku fine-tuning dataset must contain at least 2 samples.")

    groups: dict[str, list[int]] = {}
    offset = 0
    for source_index, dataset in enumerate(sources):
        for local_index in range(len(dataset)):
            key = _board_key(dataset, source_index, local_index)
            groups.setdefault(key, []).append(offset + local_index)
        offset += len(dataset)

    target_val_size = max(1, min(total - 1, round(total * validation_split)))

    group_keys = list(groups.keys())
    generator = torch.Generator().manual_seed(seed)
    shuffled = [group_keys[i] for i in torch.randperm(len(group_keys), generator=generator).tolist()]

    val_indices: list[int] = []
    for key in shuffled:
        if len(val_indices) + len(groups[key]) >= total:
            continue
        val_indices.extend(groups[key])
        if len(val_indices) >= target_val_size:
            break

    if not val_indices:
        all_idx = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
        val_indices = all_idx[:target_val_size]

    val_set = set(val_indices)
    train_indices = [i for i in range(total) if i not in val_set]

    return sorted(train_indices), sorted(val_set)


def _build_replay_subset(n_new_samples: int, replay_ratio: float, seed: int) -> tuple[Subset, int]:
    if replay_ratio <= 0:
        raise ValueError("replay_ratio must be positive when replay is enabled.")

    original_loader, _, _ = build_digit_dataloaders(languages="en")
    original_dataset = original_loader.dataset

    n_replay = max(1, round(n_new_samples * replay_ratio))
    n_replay = min(n_replay, len(original_dataset))

    indices = torch.randperm(len(original_dataset), generator=torch.Generator().manual_seed(seed))[:n_replay].tolist()
    return Subset(original_dataset, indices), n_replay


def build_sudoku_finetune_loaders() -> tuple[DataLoader, DataLoader, dict[str, int]]:
    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    train_cfg = phase_cfg.get("training", {})
    data_cfg = phase_cfg.get("data", {})
    aug_cfg = phase_cfg.get("augmentation", {})
    sudoku_cfg = data_cfg.get("sudoku_cells", {})
    fine_tune_cfg = phase_cfg.get("sudoku_fine_tune", {})

    image_size = int(model_cfg.get("image_size", 28))
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 0))
    validation_split = float(sudoku_cfg.get("validation_split", 0.1))
    seed = int(train_cfg.get("seed", 42))
    augmentation_enabled = bool(aug_cfg.get("enabled", True))
    refresh_cache = bool(sudoku_cfg.get("refresh_cache", False))
    replay_enabled = bool(fine_tune_cfg.get("replay_enabled", True))
    replay_ratio = float(fine_tune_cfg.get("replay_ratio", 0.15))

    train_transform = DigitTransform(size=image_size, augment=augmentation_enabled, augment_config=aug_cfg)
    eval_transform = DigitTransform(size=image_size, augment=False, augment_config=aug_cfg)

    # Train and validation must use the same filtered sample indices.
    # Their only difference is augmentation: enabled for train, disabled for eval.
    train_sources, summary = _build_sudoku_sources(
        train_transform, sudoku_cfg, refresh_cache, filter_easy_empty=True
    )
    eval_sources, _ = _build_sudoku_sources(
        eval_transform, sudoku_cfg, refresh_cache=False, filter_easy_empty=True
    )

    sudoku_train_full = ConcatDataset(train_sources)
    sudoku_eval_full = ConcatDataset(eval_sources)

    train_indices, val_indices = _split_by_board(train_sources, validation_split, seed)

    sudoku_train_subset = Subset(sudoku_train_full, train_indices)
    sudoku_val_subset = Subset(sudoku_eval_full, val_indices)

    if replay_enabled:
        replay_subset, n_replay = _build_replay_subset(len(sudoku_train_subset), replay_ratio, seed)
        train_dataset: Dataset = ConcatDataset([sudoku_train_subset, replay_subset])
        summary["replay:original_datasets"] = n_replay
    else:
        train_dataset = sudoku_train_subset

    summary["split:sudoku_train"] = len(sudoku_train_subset)
    summary["split:sudoku_validation"] = len(sudoku_val_subset)
    summary["split:total_train_with_replay"] = len(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(sudoku_val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, summary


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dict or a dict containing one.")

    for key in ("model_state_dict", "state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break

    state_dict = {
        k.removeprefix("module."): v
        for k, v in checkpoint.items()
        if isinstance(v, torch.Tensor)
    }

    if not state_dict:
        raise ValueError("No tensors found in checkpoint.")

    return state_dict


def _load_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)


def _resolve_source_checkpoint(phase_cfg: dict[str, Any], fine_tune_cfg: dict[str, Any]) -> Path:
    override = fine_tune_cfg.get("checkpoint_path")
    if override:
        return _resolve_path(override)

    configured = phase_cfg.get("model_paths", {}).get("en")

    if not configured:
        raise KeyError(
            "English model path is not configured. "
            "Expected: digit_recognition.model_paths.en"
        )

    return _resolve_path(configured)

def resolve_dataset_index(
    dataset: Dataset,
    index: int,
) -> tuple[Dataset, int]:
    if isinstance(dataset, Subset):
        original_index = int(dataset.indices[index])

        return resolve_dataset_index(
            dataset.dataset,
            original_index,
        )

    if isinstance(dataset, ConcatDataset):
        source_index = bisect_right(
            dataset.cumulative_sizes,
            index,
        )

        previous_size = (
            0
            if source_index == 0
            else dataset.cumulative_sizes[source_index - 1]
        )

        local_index = index - previous_size

        return resolve_dataset_index(
            dataset.datasets[source_index],
            local_index,
        )

    if isinstance(dataset, FilteredCellDataset):
        original_index = dataset.keep_indices[index]

        return resolve_dataset_index(
            dataset.base_dataset,
            original_index,
        )

    return dataset, index

def save_misclassified_samples(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 100,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    mistakes: list[dict[str, Any]] = []
    saved_count = 0
    dataset_index = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            confidences, predictions = probabilities.max(dim=1)
            wrong_indices = torch.nonzero(
                predictions != labels,
                as_tuple=False,
            ).flatten()

            for local_index in wrong_indices.tolist():
                if saved_count >= max_samples:
                    break

                true_label = int(labels[local_index].item())
                predicted_label = int(predictions[local_index].item())
                confidence = float(confidences[local_index].item())

                top_k = min(3, probabilities.shape[1])
                top_probs, top_labels = probabilities[local_index].topk(top_k)
                
                validation_index = dataset_index + local_index

                source_dataset, source_index = resolve_dataset_index(
                    data_loader.dataset,
                    validation_index,
                )

                source_sample = source_dataset.samples[source_index]

                source_cell_path_value = getattr(source_sample, "cell_path", None)

                if (
                    source_cell_path_value is None
                    and isinstance(source_sample, (tuple, list))
                    and source_sample
                ):
                    source_cell_path_value = source_sample[0]

                if source_cell_path_value is None:
                    raise ValueError(
                        "Could not determine source cell path for "
                        f"dataset={type(source_dataset).__name__}, index={source_index}"
                    )

                source_cell_path = Path(source_cell_path_value)

                board_name = source_cell_path.parent.parent.name
                cell_name = source_cell_path.stem
                
                filename = (
                    f"{saved_count:03d}"
                    f"_board-{board_name}"
                    f"_{cell_name}"
                    f"_true-{true_label}"
                    f"_pred-{predicted_label}"
                    f"_confidence-{confidence:.4f}.png"
                )

                image_path = output_dir / filename

                save_image(
                    images[local_index].detach().cpu(),
                    image_path,
                    normalize=True,
                )

                mistakes.append(
                    {
                        "source_cell_path": str(source_cell_path),
                        "board_name": board_name,
                        "cell_name": cell_name,
                        "validation_index": dataset_index + local_index,
                        "image_path": str(image_path),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "top3": [
                            {
                                "digit": int(label.item()),
                                "probability": float(prob.item()),
                            }
                            for label, prob in zip(top_labels, top_probs)
                        ],
                    }
                )

                saved_count += 1

            dataset_index += images.shape[0]

            if saved_count >= max_samples:
                break

    save_json(
        output_dir / "mistakes.json",
        {
            "count": len(mistakes),
            "max_samples": max_samples,
            "samples": mistakes,
        },
    )

    print(
        f"Saved {len(mistakes)} misclassified validation samples "
        f"to: {output_dir}"
    )

    return mistakes

def main() -> None:
    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    train_cfg = phase_cfg.get("training", {})
    fine_tune_cfg = phase_cfg.get("sudoku_fine_tune", {})
    data_cfg = phase_cfg.get("data", {})
    aug_cfg = phase_cfg.get("augmentation", {})

    seed = set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, source_summary = build_sudoku_finetune_loaders()

    num_classes = int(model_cfg.get("num_classes", 10))
    dropout = float(model_cfg.get("dropout", 0.25))
    model = build_model(num_classes=num_classes, dropout=dropout).to(device)

    checkpoint_path = _resolve_source_checkpoint(phase_cfg, fine_tune_cfg)
    _load_weights(model, checkpoint_path, device)

    freeze_bn = bool(fine_tune_cfg.get("freeze_batchnorm", True))
    if freeze_bn:
        model = freeze_batchnorm(model)

    epochs = int(fine_tune_cfg.get("epochs", 5))
    learning_rate = float(fine_tune_cfg.get("learning_rate", 0.0001))
    weight_decay = float(fine_tune_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0001)))
    early_stopping_patience = int(fine_tune_cfg.get("early_stopping_patience", 3))
    lr_scheduler_patience = int(fine_tune_cfg.get("lr_scheduler_patience", 2))
    lr_scheduler_factor = float(fine_tune_cfg.get("lr_scheduler_factor", 0.5))

    run_dir = new_run_dir()
    checkpoint_name = str(train_cfg.get("checkpoint_name", "digit_cnn.pth"))
    output_checkpoint_path = run_dir / checkpoint_name

    used_config = {
        "language": "en",
        "model": model_cfg,
        "training": train_cfg,
        "sudoku_fine_tune": fine_tune_cfg,
        "data": data_cfg,
        "augmentation": aug_cfg,
        "source_checkpoint_path": str(checkpoint_path),
        "source_summary": source_summary,
        "freeze_batchnorm": freeze_bn,
        "seed": seed,
        "device": str(device),
    }
    save_json(run_dir / "config_used.json", used_config)

    print(f"Using device: {device}")
    print("Fine-tune language: en")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"BatchNorm frozen: {freeze_bn}")
    print(f"Run directory: {run_dir}")
    print("Sudoku fine-tune samples:")
    for name, count in source_summary.items():
        print(f"  {name}: {count}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        save_path=output_checkpoint_path,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
    )

    plot_paths = save_training_plots(history, run_dir)

    _load_weights(model, output_checkpoint_path, device)
    val_loss, val_accuracy = validate(model, val_loader, nn.CrossEntropyLoss(), device)
    
    misclassified_dir = run_dir / "misclassified_validation"

    misclassified_samples = save_misclassified_samples(
        model=model,
        data_loader=val_loader,
        device=device,
        output_dir=misclassified_dir,
        max_samples=100,
    )

    report = {
        "language": "en",
        "run_dir": str(run_dir),
        "source_checkpoint_path": str(checkpoint_path),
        "checkpoint_path": str(output_checkpoint_path),
        "seed": seed,
        "device": str(device),
        "epochs_ran": len(history["train_loss"]),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "freeze_batchnorm": freeze_bn,
        "source_summary": source_summary,
        "history": history,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "plot_paths": plot_paths,
        "misclassified_count": len(misclassified_samples),
        "misclassified_dir": str(misclassified_dir),
    }
    save_json(run_dir / "training_report.json", report)

    print(f"\nSudoku fine-tuning complete: val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.4f}")
    print(f"All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()