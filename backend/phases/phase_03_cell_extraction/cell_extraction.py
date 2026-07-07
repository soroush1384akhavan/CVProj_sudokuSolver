from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from app.config import settings
from common.images import save_image


def clean_cell(cell_binary: np.ndarray, margin_ratio: float | None = None, output_size: int | None = None) -> np.ndarray:
    """Crop borders, denoise, and resize an already-binary cell for digit recognition."""

    margin_ratio = float(settings.get("cell_extraction.margin_ratio", 0.14)) if margin_ratio is None else margin_ratio
    output_size = int(settings.get("cell_extraction.digit_input_size", 28)) if output_size is None else output_size

    h, w = cell_binary.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    cropped = cell_binary[my : h - my, mx : w - mx]

    if cropped.ndim == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


    denoised = cv2.medianBlur(binary, 3)

    resized = cv2.resize(
        denoised,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    result = cv2.erode(resized, kernel, iterations=1)

    _, final_binary = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    return final_binary


def is_cell_empty(clean_binary: np.ndarray, min_ink_ratio: float | None = None) -> bool:

    min_ink_ratio = float(settings.get("cell_extraction.empty_pixel_ratio_threshold", 0.035)) if min_ink_ratio is None else min_ink_ratio

    ratio = float(np.count_nonzero(clean_binary)) / float(clean_binary.size)
    return ratio < min_ink_ratio


def make_montage(clean_cells: list[np.ndarray], cell_size: int | None = None) -> np.ndarray:
    cell_size = int(settings.get("cell_extraction.montage_cell_size", 40)) if cell_size is None else cell_size

    montage = np.ones((9 * cell_size, 9 * cell_size), dtype=np.uint8) * 255

    for idx, cell in enumerate(clean_cells):
        r, c = divmod(idx, 9)

        resized = cv2.resize(
            cell,
            (cell_size, cell_size),
            interpolation=cv2.INTER_NEAREST,
        )

        visual = 255 - resized

        montage[
            r * cell_size : (r + 1) * cell_size,
            c * cell_size : (c + 1) * cell_size,
        ] = visual

    return cv2.cvtColor(montage, cv2.COLOR_GRAY2BGR)


def extract_cells(
    warped_binary: np.ndarray,
    output_dir: Path,
) -> dict[str, list[np.ndarray] | np.ndarray | str | list[bool] | list[str]]:

    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    h, w = warped_binary.shape[:2]
    cell_h = h // 9
    cell_w = w // 9

    raw_cells: list[np.ndarray] = []
    clean_cells: list[np.ndarray] = []
    empty_flags: list[bool] = []
    cell_filenames: list[str] = []

    for r in range(9):
        for c in range(9):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w

            raw = warped_binary[y1:y2, x1:x2]

            clean = clean_cell(raw)
            empty = is_cell_empty(clean)

            idx = r * 9 + c
            filename = f"cells/cell_{idx:02d}.png"

            save_image(output_dir / filename, clean)

            raw_cells.append(raw)
            clean_cells.append(clean)
            empty_flags.append(empty)
            cell_filenames.append(filename)

    montage = make_montage(clean_cells)
    save_image(output_dir / "07_cells_montage.png", montage)

    return {
        "raw_cells": raw_cells,
        "clean_cells": clean_cells,
        "empty_flags": empty_flags,
        "cell_filenames": cell_filenames,
        "cells_montage": montage,
        "cells_montage_path": "07_cells_montage.png",
    }