## for run:
## python -m debug.debug_grid_detection

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_sudoku_grids(folder: str | Path, debug_output_dir: str | Path) -> list[dict]:
    folder = Path(folder)
    debug_output_dir = Path(debug_output_dir)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    results: list[dict] = []
    not_found_count = 0
    failed_read_count = 0

    for index, path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(path))

        if image_bgr is None:
            failed_read_count += 1
            print(f"Failed to read image skipped: {path}")
            continue

        image_output_dir = debug_output_dir / path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(image_paths)}] Processing: {path.name}")

        preprocessed = preprocess_image(
            image_bgr,
            image_output_dir,
        )

        grid_result = find_sudoku_grid(
            preprocessed_binary=preprocessed["threshold"],
            output_dir=image_output_dir,
        )

        if not grid_result["found"]:
            not_found_count += 1
            print(f"Grid not found fallback used: {path.name}")

        # چندتا مقدار کمکی اضافه می‌کنیم تا موقع debug بتوانیم فایل‌ها را پیدا کنیم
        grid_result = dict(grid_result)
        grid_result["image_path"] = str(path)
        grid_result["image_name"] = path.name
        grid_result["output_dir"] = str(image_output_dir)

        results.append(grid_result)

    print(
        f"\nTotal readable images: {len(results)} | "
        f"Grid not found: {not_found_count} | "
        f"Failed reads: {failed_read_count}"
    )

    return results


def _load_debug_image_from_path(item: dict, image_key: str):
    value = item.get(image_key)

    if value is None:
        return None

    # اگر خود تصویر داخل dict باشد
    if not isinstance(value, (str, Path)):
        return value

    image_path = Path(value)

    # اگر مسیر فقط اسم فایل بود، نسبت به output_dir حسابش کن
    if not image_path.is_absolute():
        output_dir = item.get("output_dir")
        if output_dir is not None:
            image_path = Path(output_dir) / image_path

    if not image_path.exists():
        print(f"Debug image path does not exist: {image_path}")
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Could not read debug image: {image_path}")
        return None

    return image


def _prepare_image_for_matplotlib(image):
    if image is None:
        return None, None

    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

        if image.ndim == 3:
            image = image.squeeze(0)

    # BGRA
    if getattr(image, "ndim", None) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return image, None

    # BGR
    if getattr(image, "ndim", None) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, None

    # grayscale / binary
    return image, "gray"


def debug_show_samples(
    grid_results: list[dict],
    num_samples: int = 20,
    seed: int | None = None,
    title: str = "Random samples",
    image_key: str = "warped",
):
    if not grid_results:
        print("No grid results to show.")
        return

    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(grid_results))

    random_indices = random.sample(
        range(len(grid_results)),
        k=num_samples,
    )

    selected_items = [grid_results[index] for index in random_indices]

    print(f"\n{title}")
    print(f"Dataset length: {len(grid_results)}")
    print(f"Selected indices: {random_indices}")
    print(f"Image key: {image_key}")

    cols = 5
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(12, 2.4 * rows))

    for i, item in enumerate(selected_items):
        image = _load_debug_image_from_path(item, image_key)
        image, cmap = _prepare_image_for_matplotlib(image)

        plt.subplot(rows, cols, i + 1)

        if image is None:
            plt.text(
                0.5,
                0.5,
                "Missing image",
                ha="center",
                va="center",
                fontsize=8,
            )
            plt.axis("off")
            continue

        plt.imshow(image, cmap=cmap)

        found = item.get("found")
        image_name = item.get("image_name", "")

        if found is False:
            plt.title(f"NOT FOUND\n{image_name}", color="red", fontsize=8)
        else:
            plt.title(image_name, fontsize=8)

        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    sudokus_folder = "storage/sudoku/raw"
    debug_output_dir = "storage/sudoku/debug_grid_detection"

    grid_results = load_sudoku_grids(
        folder=sudokus_folder,
        debug_output_dir=debug_output_dir,
    )

    debug_show_samples(
        grid_results=grid_results,
        num_samples=20,
        seed=None,
        title="Warped sudoku boards after grid detection",
        image_key="warped_binary",
    )

    debug_show_samples(
        grid_results=grid_results,
        num_samples=20,
        seed=None,
        title="Detected contour and corners overlay",
        image_key="contour_debug_path",
    )


if __name__ == "__main__":
    main()