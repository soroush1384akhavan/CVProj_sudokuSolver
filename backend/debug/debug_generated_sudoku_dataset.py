## for run: python -m debug.debug_generated_sudoku_dataset

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phases.phase_04_digit_recognition.src.datasets.fromsoduko.generated_dataloader import (
GeneratedSudokuCellDataset,
)



def tensor_or_image_to_numpy(image) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu()

        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        elif image.ndim == 3:
            image = image.permute(1, 2, 0)

        return image.numpy()

    if isinstance(image, Image.Image):
        return np.asarray(image)

    return np.asarray(image)


def print_dataset_summary(dataset: GeneratedSudokuCellDataset) -> None:
    language_counts = Counter(sample.language for sample in dataset.samples)
    label_counts = Counter(sample.label for sample in dataset.samples)
    kind_counts = Counter(sample.kind for sample in dataset.samples)
    source_counts = Counter(sample.source_image_path for sample in dataset.samples)

    source_language_counts = Counter(
        (sample.language, sample.source_image_path)
        for sample in dataset.samples
    )

    print("\nGenerated Sudoku dataset summary")
    print(f"Dataset length: {len(dataset)}")
    print(f"Unique source images: {len(source_counts)}")
    print(f"Languages: {dict(language_counts)}")
    print(f"Kinds: {dict(kind_counts)}")
    print(f"Labels: {dict(sorted(label_counts.items()))}")

    for language in sorted(language_counts):
        image_count = sum(
            1 for current_language, _path in source_language_counts
            if current_language == language
        )
        print(
            f"[{language}] cells={language_counts[language]} | "
            f"source images={image_count}"
        )


def choose_balanced_sample_indices(
    dataset: GeneratedSudokuCellDataset,
    num_samples_per_language: int,
    seed: int | None = None,
) -> list[int]:
    rng = random.Random(seed)
    indices_by_language: dict[str, list[int]] = defaultdict(list)

    for index, sample in enumerate(dataset.samples):
        indices_by_language[sample.language].append(index)

    selected_indices: list[int] = []

    for language in dataset.languages:
        language_indices = indices_by_language.get(language, [])
        count = min(num_samples_per_language, len(language_indices))

        if count == 0:
            print(f"No samples available for language: {language}")
            continue

        selected_indices.extend(rng.sample(language_indices, k=count))

    rng.shuffle(selected_indices)
    return selected_indices


def debug_show_random_cells(
    dataset: GeneratedSudokuCellDataset,
    num_samples_per_language: int = 15,
    seed: int | None = None,
    title: str = "Random Persian and English cells",
) -> None:
    indices = choose_balanced_sample_indices(
        dataset=dataset,
        num_samples_per_language=num_samples_per_language,
        seed=seed,
    )

    if not indices:
        raise ValueError("No dataset samples available for debug display.")

    print(f"\n{title}")
    print(f"Selected indices: {indices}")

    cols = 6
    rows = (len(indices) + cols - 1) // cols

    plt.figure(figsize=(2.3 * cols, 2.5 * rows))

    for plot_index, dataset_index in enumerate(indices):
        item = dataset[dataset_index]
        image = tensor_or_image_to_numpy(item["image"])
        language = item["language"]
        label = item["label"]
        cell_index = item["cell_index"]

        plt.subplot(rows, cols, plot_index + 1)
        plt.imshow(image, cmap="gray")
        plt.title(
            f"idx {dataset_index}\n{language} | label={label} | cell={cell_index}",
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def build_source_montage(
    dataset: GeneratedSudokuCellDataset,
    source_image_path: str | Path,
) -> np.ndarray:
    source_image_path = Path(source_image_path)

    source_samples = [
        sample
        for sample in dataset.samples
        if sample.source_image_path == source_image_path
    ]

    if len(source_samples) != 81:
        raise ValueError(
            "Montage requires include_empty_cells=True and exactly 81 cached cells. "
            f"Found {len(source_samples)} cells for {source_image_path}."
        )

    source_samples.sort(key=lambda sample: sample.cell_index)

    cell_images: list[np.ndarray] = []
    max_height = 0
    max_width = 0

    for sample in source_samples:
        with Image.open(sample.cell_path) as image_file:
            cell = np.asarray(image_file.convert("L"))

        cell_images.append(cell)
        max_height = max(max_height, cell.shape[0])
        max_width = max(max_width, cell.shape[1])

    montage = np.full(
        (9 * max_height, 9 * max_width),
        fill_value=255,
        dtype=np.uint8,
    )

    for cell_index, cell in enumerate(cell_images):
        row = cell_index // 9
        col = cell_index % 9
        y0 = row * max_height
        x0 = col * max_width
        montage[y0 : y0 + cell.shape[0], x0 : x0 + cell.shape[1]] = cell

    return montage


def debug_show_cell_montages(
    dataset: GeneratedSudokuCellDataset,
    num_images_per_language: int = 3,
    seed: int | None = None,
    title: str = "Extracted 9x9 cell montages",
) -> None:
    rng = random.Random(seed)
    sources_by_language: dict[str, list[Path]] = defaultdict(list)

    for sample in dataset.samples:
        if sample.source_image_path not in sources_by_language[sample.language]:
            sources_by_language[sample.language].append(sample.source_image_path)

    selected_sources: list[tuple[str, Path]] = []

    for language in dataset.languages:
        sources = sources_by_language.get(language, [])
        count = min(num_images_per_language, len(sources))

        if count == 0:
            continue

        selected_sources.extend(
            (language, path) for path in rng.sample(sources, k=count)
        )

    if not selected_sources:
        raise ValueError("No source images available for montage display.")

    cols = 3
    rows = (len(selected_sources) + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))

    for plot_index, (language, source_path) in enumerate(selected_sources):
        montage = build_source_montage(dataset, source_path)

        plt.subplot(rows, cols, plot_index + 1)
        plt.imshow(montage, cmap="gray")
        plt.title(
            f"{language} | {source_path.name}",
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    
    generated_run_root = Path(
    r"C:\Users\sorou\OneDrive\Desktop\sudoku-cv-windows-pytorch-opencv\backend\sudoku_runs\sudoku_medium_20260714_151427_563299_seed_1701254515")
    cache_dir = "storage/sudoku/debug_generated_dataset/extracted_cells"

    dataset = GeneratedSudokuCellDataset(
        root_dir=generated_run_root,
        cache_dir=cache_dir,
        languages=("en", "fa"),
        include_empty_cells=False,
        refresh_cache=False,
        label_field="puzzle",
        apply_safe_augmentation=False,
        return_metadata=True,
        strict_languages=True,
    )

    print_dataset_summary(dataset)

    debug_show_random_cells(
        dataset=dataset,
        num_samples_per_language=12,
        seed=None,
        title="Random cells from English and Persian generated datasets",
    )

    debug_show_cell_montages(
        dataset=dataset,
        num_images_per_language=2,
        seed=None,
        title="9x9 extracted-cell montages for English and Persian images",
    )


if __name__ == "__main__":
    main()