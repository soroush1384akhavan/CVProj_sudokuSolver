## for run : python -m debug.debug_preprocessing

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from phases.phase_01_preprocessing.preprocess import preprocess_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_sudoku_images(folder: str | Path, debug_output_dir: str | Path) -> list:
    folder = Path(folder)
    debug_output_dir = Path(debug_output_dir)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    images = []
    for path in image_paths:
        image_bgr = cv2.imread(str(path))

        if image_bgr is None:
            print(f"Failed to read image (skipped): {path}")
            continue

        image_output_dir = debug_output_dir / path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        processed = preprocess_image(image_bgr, image_output_dir)
        images.append(processed)

    return images


def debug_show_samples(
    sudokus,
    num_samples: int = 20,
    seed: int | None = None,
    title: str = "Random samples",
    image_key: str = "threshold",
):
    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(sudokus))

    random_indices = random.sample(
        range(len(sudokus)),
        k=num_samples,
    )

    images = [sudokus[index] for index in random_indices]

    print(f"\n{title}")
    print(f"Dataset length: {len(sudokus)}")
    print(f"Selected indices: {random_indices}")

    cols = 5
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(10, 2 * rows))

    for i, image in enumerate(images):
        if isinstance(image, dict):
            image = image[image_key]

        if torch.is_tensor(image):
            image = image.detach().cpu()
            if image.ndim == 3:
                image = image.squeeze(0)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    sudokus_folder = "storage/sudoku/raw"
    debug_output_dir = "storage/sudoku/debug_preprocessing"

    sudokus = load_sudoku_images(sudokus_folder, debug_output_dir)

    debug_show_samples(
        sudokus=sudokus,
        num_samples=20,
        seed=None,
        title="Train loader random samples with augmentation",
        image_key="threshold",
    )


if __name__ == "__main__":
    main()