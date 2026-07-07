## for run : python -m debug.debug_overlay

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells
from phases.phase_04_digit_recognition.src.classifier import PyTorchDigitClassifier
from phases.phase_05_solver.solver import solve_board
from phases.phase_06_overlay.overlay import create_overlay_images

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_and_create_overlays(folder: str | Path, debug_output_dir: str | Path) -> list:

    folder = Path(folder)
    debug_output_dir = Path(debug_output_dir)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    classifier = PyTorchDigitClassifier()

    results = []
    skipped_count = 0

    for path in image_paths:
        image_bgr = cv2.imread(str(path))

        if image_bgr is None:
            print(f"⚠️  Failed to read image (skipped): {path}")
            continue

        image_output_dir = debug_output_dir / path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        preprocessed = preprocess_image(image_bgr, image_output_dir)

        grid_result = find_sudoku_grid(
            original_bgr=preprocessed["original"],
            threshold=preprocessed["threshold"],
            output_dir=image_output_dir,
        )

        cells_result = extract_cells(
            warped_bgr=grid_result["warped"],
            output_dir=image_output_dir,
        )

        original_board, confidence, low_confidence_cells = classifier.predict_board(
            clean_cells=cells_result["clean_cells"],
            empty_flags=cells_result["empty_flags"],
        )

        solved, solved_board, message = solve_board(original_board)

        if not solved:
            skipped_count += 1
            print(f"⚠️  Skipping overlay (solve failed): {path.name} -> {message}")
            continue

        overlay_files = create_overlay_images(
            run_dir=image_output_dir,
            original_bgr=preprocessed["original"],
            warped_bgr=grid_result["warped"],
            corners=grid_result["corners"],
            original_board=original_board,
            solved_board=solved_board,
            inverse_matrix=grid_result.get("inverse_matrix"),
        )

        results.append({
            "source_path": str(path),
            "image_output_dir": str(image_output_dir),
            "overlay_files": overlay_files,
            "grid_found": grid_result["found"],
        })

    print(f"\nTotal images processed: {len(results)} | Skipped (solve failed): {skipped_count}")

    return results


def debug_show_overlays(
    results,
    num_samples: int = 6,
    seed: int | None = None,
    title: str = "Solved overlay on original image",
) -> None:
    """
    برای چند نمونه‌ی رندوم، تصویر نهایی (ارقام حل‌شده روی عکس اصلی) رو نشون می‌ده.
    """
    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(results))
    random_indices = random.sample(range(len(results)), k=num_samples)

    cols = 3
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, idx in enumerate(random_indices):
        item = results[idx]
        image_output_dir = Path(item["image_output_dir"])

        overlay_entry = next(
            f for f in item["overlay_files"] if f["key"] == "solved_overlay"
        )
        overlay_path = image_output_dir / overlay_entry["filename"]
        overlay_img = cv2.imread(str(overlay_path))
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay_rgb)
        plt.title(f"idx {idx}", fontsize=9)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def debug_show_warped_vs_overlay(
    results,
    result_index: int = 0,
    title: str = "Warped solution vs Original overlay",
) -> None:
    """
    برای یک تصویر مشخص، دو خروجی overlay (روی تخته‌ی صاف، و روی عکس اصلی) رو کنار هم نشون می‌ده.
    """
    item = results[result_index]
    image_output_dir = Path(item["image_output_dir"])

    warped_entry = next(f for f in item["overlay_files"] if f["key"] == "solved_warped")
    overlay_entry = next(f for f in item["overlay_files"] if f["key"] == "solved_overlay")

    warped_img = cv2.imread(str(image_output_dir / warped_entry["filename"]))
    overlay_img = cv2.imread(str(image_output_dir / overlay_entry["filename"]))

    warped_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(warped_rgb)
    plt.title("Solved Warped Board")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_rgb)
    plt.title("Solved Original Overlay")
    plt.axis("off")

    plt.suptitle(f"{title} (result index {result_index})")
    plt.tight_layout()
    plt.show()


def main():
    sudokus_folder = "storage/sudoku/raw"
    debug_output_dir = "storage/sudoku/debug_overlay"

    results = load_and_create_overlays(sudokus_folder, debug_output_dir)

    if not results:
        print("No successful overlays to display.")
        return

    debug_show_overlays(results, num_samples=6, seed=None)

    debug_show_warped_vs_overlay(results, result_index=0)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()