## for run : python -m debug.debug_solver

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells
from phases.phase_04_digit_recognition.src.classifier import PyTorchDigitClassifier
from phases.phase_05_solver.solver import solve_board

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_and_solve(
    folder: str | Path,
    debug_output_dir: str | Path,
    exclude_dirs: set[str] | None = None,
) -> list:

    folder = Path(folder)
    debug_output_dir = Path(debug_output_dir)
    exclude_dirs = exclude_dirs or {"mixed", "mixed 2"}

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(excluded in p.parts for excluded in exclude_dirs)
    )

    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    classifier = PyTorchDigitClassifier()

    results = []
    solve_failed_count = 0
    grid_not_found_count = 0

    for path in image_paths:
        image_bgr = cv2.imread(str(path))

        if image_bgr is None:
            print(f"⚠️  Failed to read image (skipped): {path}")
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
            print(f"⚠️  Grid not found (fallback used): {path.name}")

        cells_result = extract_cells(
            warped_bgr=grid_result["warped"],
            warped_binary=grid_result["warped_binary"],
            output_dir=image_output_dir,
        )

        original_board, confidence, low_confidence_cells = classifier.predict_board(
            clean_cells=cells_result["clean_cells"],
            empty_flags=cells_result["empty_flags"],
        )

        solved, solved_board, message = solve_board(original_board)

        if not solved:
            solve_failed_count += 1
            print(f"⚠️  Solve failed: {path.name} -> {message}")

        results.append({
            "source_path": str(path),
            "original_board": original_board,
            "solved_board": solved_board,
            "solved": solved,
            "message": message,
            "confidence": confidence,
            "low_confidence_cells": low_confidence_cells,
            "grid_found": grid_result["found"],
        })

    print(
        f"\nTotal images: {len(results)} | "
        f"Grid not found: {grid_not_found_count} | "
        f"Solve failed: {solve_failed_count}"
    )

    return results


def print_board(board: list[list[int]], title: str = "Board") -> None:
    print(f"\n{title}")

    for r in range(9):
        row_str = " ".join(
            str(board[r][c]) if board[r][c] != 0 else "."
            for c in range(9)
        )

        if r % 3 == 0 and r != 0:
            print("-" * 21)

        formatted = " | ".join(row_str[i:i + 6] for i in range(0, len(row_str), 6))
        print(formatted)


def debug_print_boards(results, num_samples: int = 5, seed: int | None = None) -> None:
    """
    برای چند نمونه‌ی رندوم، پازل اصلی تشخیص‌داده‌شده و حل‌شده رو در کنسول چاپ می‌کنه.
    """

    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(results))
    random_indices = random.sample(range(len(results)), k=num_samples)

    for idx in random_indices:
        item = results[idx]

        print(f"\n{'=' * 40}")
        print(f"Image: {item['source_path']}")
        print(
            f"Grid found: {item['grid_found']} | "
            f"Solved: {item['solved']} | "
            f"Message: {item['message']}"
        )

        print_board(item["original_board"], title="Detected original board")

        if item["solved_board"] is not None:
            print_board(item["solved_board"], title="Solved board")

        if item["low_confidence_cells"]:
            print(f"Low-confidence cells: {item['low_confidence_cells']}")


def debug_show_confidence_heatmap(
    results,
    num_samples: int = 4,
    seed: int | None = None,
    title: str = "Digit confidence heatmaps",
) -> None:
    """
    برای چند نمونه، heatmap میزان اطمینان مدل روی هر خانه رو نشون می‌ده.
    """

    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(results))
    random_indices = random.sample(range(len(results)), k=num_samples)

    cols = 2
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, idx in enumerate(random_indices):
        item = results[idx]
        confidence = item["confidence"]

        plt.subplot(rows, cols, i + 1)
        im = plt.imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        for r in range(9):
            for c in range(9):
                digit = item["original_board"][r][c]
                if digit != 0:
                    plt.text(
                        c,
                        r,
                        str(digit),
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

        plt.title(f"idx {idx} | solved={item['solved']}", fontsize=9)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    sudokus_folder = "storage/sudoku/raw/v1_test"
    debug_output_dir = "storage/sudoku/debug_solver"

    results = load_and_solve(
        sudokus_folder,
        debug_output_dir,
        exclude_dirs={"mixed", "mixed 2"},
    )

    debug_print_boards(results, num_samples=5, seed=None)

    debug_show_confidence_heatmap(results, num_samples=4, seed=None)


if __name__ == "__main__":
    main()