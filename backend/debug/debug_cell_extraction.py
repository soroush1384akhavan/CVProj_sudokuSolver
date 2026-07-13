## for run : python -m debug.debug_cell_extraction

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_and_extract_cells(folder: str | Path, debug_output_dir: str | Path) -> list:
    folder = Path(folder)
    debug_output_dir = Path(debug_output_dir)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    results = []
    grid_not_found_count = 0

    for path in image_paths:
        image_bgr = cv2.imread(str(path))

        if image_bgr is None:
            print(f"Failed to read image (skipped): {path}")
            continue

        image_output_dir = debug_output_dir / path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        preprocessed = preprocess_image(image_bgr, image_output_dir)

        grid_result = find_sudoku_grid(
            preprocessed_binary=preprocessed["threshold"],
            original_bgr=preprocessed["original"],
            output_dir=image_output_dir,
        )

        if not grid_result["found"]:
            grid_not_found_count += 1
            print(f"Grid not found (fallback used): {path.name}")

        cells_result = extract_cells(
            warped_bgr=grid_result["warped"],
            output_dir=image_output_dir,
        )

        cells_result["source_path"] = str(path)
        cells_result["grid_found"] = grid_result["found"]

        results.append(cells_result)

    print(f"\nTotal images: {len(results)} | Grid not found: {grid_not_found_count}")

    return results


def debug_show_montages(
    cell_results,
    num_samples: int = 20,
    seed: int | None = None,
    title: str = "Cell extraction montages",
):

    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(cell_results))

    random_indices = random.sample(
        range(len(cell_results)),
        k=num_samples,
    )

    print(f"\n{title}")
    print(f"Dataset length: {len(cell_results)}")
    print(f"Selected indices: {random_indices}")

    cols = 4
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, index in enumerate(random_indices):
        item = cell_results[index]

        montage_img = item["cells_montage"]
        montage_rgb = cv2.cvtColor(montage_img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(montage_rgb)

        found = item.get("grid_found")
        subtitle = "" if found else " (grid NOT found)"
        plt.title(f"idx {index}{subtitle}", fontsize=8, color="red" if not found else "black")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def debug_show_single_cells(
    cell_results,
    result_index: int = 0,
    num_samples: int = 20,
    seed: int | None = None,
    title: str = "Random single cells",
):

    if seed is not None:
        random.seed(seed)

    item = cell_results[result_index]
    clean_cells = item["clean_cells"]
    empty_flags = item["empty_flags"]

    num_samples = min(num_samples, len(clean_cells))
    random_indices = random.sample(range(len(clean_cells)), k=num_samples)

    print(f"\n{title} (from result index {result_index})")
    print(f"Selected cell indices: {random_indices}")

    cols = 5
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(10, 2 * rows))

    for i, cell_idx in enumerate(random_indices):
        cell = clean_cells[cell_idx]
        empty = empty_flags[cell_idx]

        if torch.is_tensor(cell):
            cell = cell.detach().cpu()
            if cell.ndim == 3:
                cell = cell.squeeze(0)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(cell, cmap="gray")
        plt.title(f"#{cell_idx}{' (empty)' if empty else ''}", fontsize=8, color="gray" if empty else "black")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    sudokus_folder = "storage/sudoku/raw"
    debug_output_dir = "storage/sudoku/debug_cell_extraction"

    cell_results = load_and_extract_cells(sudokus_folder, debug_output_dir)

    debug_show_montages(
        cell_results=cell_results,
        num_samples=8,
        seed=None,
        title="Cell extraction montages (per image)",
    )

    debug_show_single_cells(
        cell_results=cell_results,
        result_index=0,
        num_samples=20,
        seed=None,
        title="Random single cells from first image",
    )


if __name__ == "__main__":
    main()