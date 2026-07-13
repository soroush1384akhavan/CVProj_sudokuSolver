## for run:
## python -m debug.debug_row1_duplicate
## python -m debug.debug_row1_duplicate --row 1
## python -m debug.debug_row1_duplicate --row 4

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells
from phases.phase_04_digit_recognition.src.classifier import PyTorchDigitClassifier


IMAGE_PATH = "storage/sudoku/raw/v1_test/v1_test/image199.jpg"
DEBUG_DIR = Path("storage/sudoku/debug_row_check")

# ردیف دلخواه برای نمایش، عدد انسانی از 1 تا 9
ROW_NUMBER = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row",
        type=int,
        default=ROW_NUMBER,
        help="Row number to debug. Use 1 to 9.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_PATH,
        help="Path to sudoku image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    row_number = args.row

    if row_number < 1 or row_number > 9:
        raise ValueError(f"row must be between 1 and 9, got: {row_number}")

    row_idx = row_number - 1
    image_path = args.image

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    preprocessed = preprocess_image(image_bgr, DEBUG_DIR)

    grid_result = find_sudoku_grid(
        preprocessed_binary=preprocessed["threshold"],
        original_bgr=preprocessed["original"],
        output_dir=DEBUG_DIR,
    )

    cells_result = extract_cells(
        warped_bgr=grid_result["warped"],
        warped_binary=grid_result["warped_binary"],
        output_dir=DEBUG_DIR,
    )

    classifier = PyTorchDigitClassifier()

    board, confidence, low_conf = classifier.predict_board(
        clean_cells=cells_result["clean_cells"],
        empty_flags=cells_result["empty_flags"],
    )

    print("Grid found:", grid_result["found"])
    print(f"Detected board, row {row_number}:", board[row_idx])
    print(
        f"Confidence, row {row_number}:",
        [round(c, 3) for c in confidence[row_idx]],
    )

    if low_conf:
        print("Low confidence cells:")
        for item in low_conf:
            print(item)

    # نمایش ۹ خانه‌ی ردیف انتخاب‌شده
    plt.figure(figsize=(14, 2.5))

    for col in range(9):
        flat_idx = row_idx * 9 + col
        cell = cells_result["clean_cells"][flat_idx]

        pred = board[row_idx][col]
        conf = confidence[row_idx][col]
        is_empty = cells_result["empty_flags"][flat_idx]

        plt.subplot(1, 9, col + 1)
        plt.imshow(cell, cmap="gray")
        plt.title(
            f"r={row_number}, c={col + 1}\n"
            f"pred={pred}\n"
            f"conf={conf:.2f}\n"
            f"empty={is_empty}",
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle(f"Row {row_number} cells — {image_path}")
    plt.tight_layout()
    plt.show()

    # نمایش کل تخته‌ی warp شده
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_result["warped"], cmap="gray")
    plt.title("Warped binary board")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()