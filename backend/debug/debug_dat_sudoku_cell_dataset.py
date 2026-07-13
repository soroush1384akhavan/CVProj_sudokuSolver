# for run:
# python -m debug.debug_dat_sudoku_cell_dataset

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

# این import را مطابق ساختار پروژه خودت تغییر بده
from phases.phase_04_digit_recognition.src.datasets.fromsoduko.charls_soduku_dataloader import DatSudokuCellDataset


def image_to_numpy(image) -> np.ndarray:
    """
    Convert PIL image, NumPy array, or torch Tensor to a displayable NumPy array.
    """
    if torch.is_tensor(image):
        image = image.detach().cpu()

        # CHW -> HWC
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            else:
                image = image.permute(1, 2, 0)

        image = image.numpy()

    elif not isinstance(image, np.ndarray):
        image = np.asarray(image)

    return image


def print_dataset_summary(
    dataset: DatSudokuCellDataset,
    title: str = "Dataset summary",
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Dataset length: {len(dataset)}")
    print(f"Root directory: {dataset.root_dir}")
    print(f"Cache directory: {dataset.cache_dir}")
    print(f"Include empty cells: {dataset.include_empty_cells}")
    print(f"Safe augmentation: {dataset.apply_safe_augmentation}")

    labels = [label for _, label in dataset.samples]
    label_counts = Counter(labels)

    print("\nLabel distribution:")

    for label in sorted(label_counts):
        print(f"  Label {label}: {label_counts[label]} samples")

    print("=" * 80)


def inspect_sample(
    dataset: DatSudokuCellDataset,
    index: int,
) -> None:
    image, label = dataset[index]
    image_path, stored_label = dataset.samples[index]

    image_array = image_to_numpy(image)

    print("\n" + "-" * 80)
    print(f"Sample index: {index}")
    print(f"Image path: {image_path}")
    print(f"Returned label: {label}")
    print(f"Stored label: {stored_label}")
    print(f"Image type: {type(image)}")
    print(f"Image shape: {image_array.shape}")
    print(f"Image dtype: {image_array.dtype}")
    print(f"Minimum value: {image_array.min():.6f}")
    print(f"Maximum value: {image_array.max():.6f}")
    print(f"Mean value: {image_array.mean():.6f}")
    print(f"Standard deviation: {image_array.std():.6f}")
    print("-" * 80)


def debug_show_samples(
    dataset: DatSudokuCellDataset,
    num_samples: int = 20,
    seed: int | None = None,
    title: str = "Random Sudoku cells",
) -> None:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    random_generator = random.Random(seed)
    num_samples = min(num_samples, len(dataset))

    selected_indices = random_generator.sample(
        range(len(dataset)),
        k=num_samples,
    )

    print(f"\n{title}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Selected indices: {selected_indices}")

    cols = 5
    rows = (num_samples + cols - 1) // cols

    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, 3 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for plot_index, dataset_index in enumerate(selected_indices):
        image, label = dataset[dataset_index]
        image_path, stored_label = dataset.samples[dataset_index]

        image_array = image_to_numpy(image)
        axis = axes.flat[plot_index]

        axis.imshow(
            image_array,
            cmap="gray",
            interpolation="nearest",
        )

        # لیبل واضح در بالای تصویر
        axis.set_title(
            f"Label: {label}",
            fontsize=12,
            fontweight="bold",
            pad=10,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=0.25",
            },
        )

        # اطلاعات جانبی پایین تصویر
        axis.set_xlabel(
            f"Index: {dataset_index}\n{image_path.name}",
            fontsize=8,
            labelpad=6,
        )

        axis.set_xticks([])
        axis.set_yticks([])

        # بررسی اینکه لیبل برگشتی با لیبل ذخیره‌شده یکی باشد
        if label != stored_label:
            axis.set_xlabel(
                f"WARNING: returned={label}, stored={stored_label}\n"
                f"Index: {dataset_index}",
                fontsize=8,
                labelpad=6,
            )

    figure.suptitle(
        title,
        fontsize=16,
        y=0.99,
    )

    figure.subplots_adjust(
        top=0.90,
        bottom=0.08,
        hspace=0.85,
        wspace=0.30,
    )

    plt.show()

def debug_compare_augmentation(
    raw_dataset: DatSudokuCellDataset,
    augmented_dataset: DatSudokuCellDataset,
    num_samples: int = 10,
    seed: int | None = None,
) -> None:
    if len(raw_dataset) != len(augmented_dataset):
        raise ValueError(
            "Raw and augmented datasets have different lengths: "
            f"{len(raw_dataset)} != {len(augmented_dataset)}"
        )

    if raw_dataset.samples != augmented_dataset.samples:
        raise ValueError(
            "Raw and augmented datasets do not have the same sample order."
        )

    random_generator = random.Random(seed)
    num_samples = min(num_samples, len(raw_dataset))

    selected_indices = random_generator.sample(
        range(len(raw_dataset)),
        k=num_samples,
    )

    print("\nRaw versus augmented comparison")
    print(f"Selected indices: {selected_indices}")

    figure, axes = plt.subplots(
        num_samples,
        2,
        figsize=(8, 3.5 * num_samples),
        squeeze=False,
    )

    for row_index, dataset_index in enumerate(selected_indices):
        raw_image, raw_label = raw_dataset[dataset_index]
        augmented_image, augmented_label = augmented_dataset[dataset_index]

        image_path, _ = raw_dataset.samples[dataset_index]

        raw_array = image_to_numpy(raw_image)
        augmented_array = image_to_numpy(augmented_image)

        raw_axis = axes[row_index, 0]
        augmented_axis = axes[row_index, 1]

        raw_axis.imshow(
            raw_array,
            cmap="gray",
            interpolation="nearest",
        )

        raw_axis.set_title(
            "Raw",
            fontsize=11,
            pad=6,
        )

        raw_axis.set_xlabel(
            f"Index: {dataset_index} | Label: {raw_label}\n"
            f"{image_path.name}",
            fontsize=9,
            labelpad=8,
        )

        raw_axis.set_xticks([])
        raw_axis.set_yticks([])

        augmented_axis.imshow(
            augmented_array,
            cmap="gray",
            interpolation="nearest",
        )

        augmented_axis.set_title(
            "Augmented",
            fontsize=11,
            pad=6,
        )

        augmented_axis.set_xlabel(
            f"Index: {dataset_index} | Label: {augmented_label}\n"
            f"Shape: {augmented_array.shape}",
            fontsize=9,
            labelpad=8,
        )

        augmented_axis.set_xticks([])
        augmented_axis.set_yticks([])

    figure.suptitle(
        "Raw and safe augmentation comparison",
        fontsize=16,
        y=0.998,
    )

    figure.subplots_adjust(
        top=0.97,
        bottom=0.03,
        hspace=0.8,
        wspace=0.3,
    )

    plt.show()

def debug_same_sample_multiple_augmentations(
    dataset: DatSudokuCellDataset,
    index: int,
    num_versions: int = 10,
) -> None:
    """
    نمایش چند augmentation تصادفی از یک سلول ثابت.
    """
    if not dataset.apply_safe_augmentation:
        raise ValueError(
            "Dataset augmentation is disabled. "
            "Set apply_safe_augmentation=True."
        )

    if not 0 <= index < len(dataset):
        raise IndexError(
            f"Index {index} is outside dataset range 0..{len(dataset) - 1}"
        )

    image_path, stored_label = dataset.samples[index]

    cols = 5
    rows = (num_versions + cols - 1) // cols

    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 2.8 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for version_index in range(num_versions):
        image, label = dataset[index]
        image_array = image_to_numpy(image)

        axis = axes.flat[version_index]
        axis.imshow(image_array, cmap="gray")
        axis.set_title(
            f"Version {version_index + 1}\n"
            f"label={label}",
            fontsize=9,
        )
        axis.axis("off")

    figure.suptitle(
        f"Multiple augmentations\n"
        f"index={index}, label={stored_label}, file={image_path.name}",
        fontsize=14,
    )

    figure.tight_layout()
    plt.show()


def main() -> None:
    dat_root_dir = Path(
        r"C:\Users\sorou\OneDrive\Desktop\sudoku-cv-windows-pytorch-opencv"
        r"\backend\storage\sudoku\raw\v1_training\v1_training"
    )

    # سلول‌های استخراج‌شده در این پوشه ذخیره می‌شوند
    cache_dir = dat_root_dir / "_debug_extracted_cells"

    print(f"Dataset directory: {dat_root_dir}")
    print(f"Dataset exists: {dat_root_dir.exists()}")
    print(f"Cache directory: {cache_dir}")

    cell_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    raw_dataset = DatSudokuCellDataset(
        root_dir=dat_root_dir,
        cache_dir=cache_dir,
        include_empty_cells=False,
        transform=cell_transform,
        refresh_cache=False,
        apply_safe_augmentation=False,
    )

    augmented_dataset = DatSudokuCellDataset(
        root_dir=dat_root_dir,
        cache_dir=cache_dir,
        include_empty_cells=False,
        transform=cell_transform,
        refresh_cache=False,
        apply_safe_augmentation=True,
    )

    print_dataset_summary(
        dataset=raw_dataset,
        title="Raw DAT Sudoku dataset",
    )

    inspect_sample(
        dataset=raw_dataset,
        index=0,
    )

    debug_show_samples(
        dataset=raw_dataset,
        num_samples=20,
        seed=None,
        title="Random cells with labels",
    )

    debug_compare_augmentation(
        raw_dataset=raw_dataset,
        augmented_dataset=augmented_dataset,
        num_samples=10,
        seed=42,
    )

    debug_same_sample_multiple_augmentations(
        dataset=augmented_dataset,
        index=0,
        num_versions=10,
    )


if __name__ == "__main__":
    main()