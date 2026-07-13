from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from common.images import save_image


def is_cell_empty(
    inverted_gray: np.ndarray,
    min_ink_ratio: float | None = None,
    margin_ratio: float | None = None,
) -> bool:
    """
    تشخیص خالی‌بودن سلول grayscale اینورت‌شده.

    تصویر تغییر داده یا binary نمی‌شود. فقط بررسی می‌کنیم چه نسبتی
    از پیکسل‌های ناحیه‌ی داخلی، شدت قابل‌توجهی دارند.

    در تصویر اینورت‌شده:
    - پس‌زمینه تقریباً سیاه است.
    - رقم روشن است.
    """
    if inverted_gray.ndim != 2:
        raise ValueError(
            "inverted_gray must be a single-channel image"
        )

    min_ink_ratio = (
        float(
            settings.get(
                "cell_extraction.empty_pixel_ratio_threshold",
                0.025,
            )
        )
        if min_ink_ratio is None
        else float(min_ink_ratio)
    )

    margin_ratio = (
        float(
            settings.get(
                "cell_extraction.margin_ratio",
                0.10,
            )
        )
        if margin_ratio is None
        else float(margin_ratio)
    )

    intensity_threshold = int(
        settings.get(
            "cell_extraction.empty_intensity_threshold",
            35,
        )
    )

    height, width = inverted_gray.shape

    margin_y = int(round(height * margin_ratio))
    margin_x = int(round(width * margin_ratio))

    y1 = margin_y
    y2 = height - margin_y
    x1 = margin_x
    x2 = width - margin_x

    if y2 <= y1 or x2 <= x1:
        return True

    inner = inverted_gray[y1:y2, x1:x2]

    if inner.size == 0:
        return True

    # فقط برای تشخیص خالی‌بودن یک mask منطقی می‌سازیم.
    # تصویر خروجی هیچ‌وقت binary نمی‌شود.
    significant_pixels = inner >= intensity_threshold

    ink_ratio = float(
        np.count_nonzero(significant_pixels)
        / inner.size
    )

    return ink_ratio < min_ink_ratio


def clean_cell(
    cell: np.ndarray,
    margin_ratio: float | None = None,
    output_size: int | None = None,
    min_area_ratio: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    سلول را فقط grayscale و invert می‌کند.

    هیچ‌کدام از عملیات زیر انجام نمی‌شوند:
    - adaptive threshold
    - binary threshold
    - connected components
    - morphology
    - crop
    - resize
    - center کردن رقم
    """
    del output_size
    del min_area_ratio

    if cell.ndim == 3:
        gray = cv2.cvtColor(
            cell,
            cv2.COLOR_BGR2GRAY,
        )
    elif cell.ndim == 2:
        gray = cell.copy()
    else:
        raise ValueError(
            f"Unsupported cell shape: {cell.shape}"
        )

    expected_size = int(
        settings.get(
            "cell_extraction.digit_input_size",
            28,
        )
    )

    if gray.shape != (
        expected_size,
        expected_size,
    ):
        raise ValueError(
            f"Expected cell size "
            f"{expected_size}x{expected_size}, "
            f"but received "
            f"{gray.shape[1]}x{gray.shape[0]}. "
            f"Make sure grid_detection.board_size is "
            f"{expected_size * 9}."
        )

    # فقط inversion؛ شدت‌های خاکستری حفظ می‌شوند.
    inverted_gray = cv2.bitwise_not(gray)

    empty = is_cell_empty(
        inverted_gray=inverted_gray,
        margin_ratio=margin_ratio,
    )

    if empty:
        return (
            np.zeros_like(
                inverted_gray,
                dtype=np.uint8,
            ),
            True,
        )

    return inverted_gray, False


def make_montage(
    cells: list[np.ndarray],
    cell_size: int | None = None,
) -> np.ndarray:
    del cell_size

    if len(cells) != 81:
        raise ValueError(
            f"Expected 81 cells, received {len(cells)}"
        )

    first_cell = cells[0]

    if first_cell.ndim != 2:
        raise ValueError(
            "cells must contain grayscale images"
        )

    cell_height, cell_width = first_cell.shape

    montage = np.zeros(
        (
            9 * cell_height,
            9 * cell_width,
        ),
        dtype=np.uint8,
    )

    for index, cell in enumerate(cells):
        if cell.ndim != 2:
            raise ValueError(
                f"Cell {index} is not grayscale"
            )

        if cell.shape != (
            cell_height,
            cell_width,
        ):
            raise ValueError(
                f"Cell {index} has shape {cell.shape}, "
                f"expected {(cell_height, cell_width)}"
            )

        row, column = divmod(index, 9)

        y1 = row * cell_height
        y2 = y1 + cell_height
        x1 = column * cell_width
        x2 = x1 + cell_width

        montage[
            y1:y2,
            x1:x2,
        ] = cell

    return cv2.cvtColor(
        montage,
        cv2.COLOR_GRAY2BGR,
    )


def extract_cells(
    warped_bgr: np.ndarray,
    output_dir: Path,
) -> dict[
    str,
    list[np.ndarray]
    | np.ndarray
    | str
    | list[bool]
    | list[str],
]:
    if (
        warped_bgr is None
        or warped_bgr.size == 0
    ):
        raise ValueError(
            "warped_bgr is empty"
        )

    cells_dir = output_dir / "cells"

    cells_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = warped_bgr.shape[:2]

    expected_cell_size = int(
        settings.get(
            "cell_extraction.digit_input_size",
            28,
        )
    )

    expected_board_size = (
        expected_cell_size * 9
    )

    if (
        height != expected_board_size
        or width != expected_board_size
    ):
        raise ValueError(
            f"Expected warped board size "
            f"{expected_board_size}x"
            f"{expected_board_size}, "
            f"but received {width}x{height}. "
            f"Set grid_detection.board_size to "
            f"{expected_board_size}."
        )

    cell_height = height // 9
    cell_width = width // 9

    raw_cells: list[np.ndarray] = []
    inverted_cells: list[np.ndarray] = []
    empty_flags: list[bool] = []
    cell_filenames: list[str] = []

    save_cells = bool(
        settings.get(
            "cell_extraction.save_cells",
            True,
        )
    )

    for row in range(9):
        for column in range(9):
            y1 = row * cell_height
            y2 = (row + 1) * cell_height

            x1 = column * cell_width
            x2 = (column + 1) * cell_width

            raw = warped_bgr[
                y1:y2,
                x1:x2,
            ]

            if raw.shape[:2] != (
                expected_cell_size,
                expected_cell_size,
            ):
                raise ValueError(
                    f"Cell ({row}, {column}) "
                    f"has shape {raw.shape[:2]}, "
                    f"expected "
                    f"{expected_cell_size}x"
                    f"{expected_cell_size}"
                )

            if raw.ndim == 3:
                raw_gray = cv2.cvtColor(
                    raw,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                raw_gray = raw.copy()

            inverted, empty = clean_cell(
                raw_gray
            )

            index = row * 9 + column

            filename = (
                f"cells/cell_{index:02d}.png"
            )

            if save_cells:
                save_image(
                    output_dir / filename,
                    inverted,
                )

            raw_cells.append(
                raw_gray
            )

            inverted_cells.append(
                inverted
            )

            empty_flags.append(
                empty
            )

            cell_filenames.append(
                filename
            )

    raw_montage = make_montage(
        raw_cells
    )

    save_image(
        output_dir
        / "06_raw_cells_montage.png",
        raw_montage,
    )

    inverted_montage = make_montage(
        inverted_cells
    )

    montage_path = (
        "07_inverted_grayscale_montage.png"
    )

    save_image(
        output_dir / montage_path,
        inverted_montage,
    )

    return {
        "raw_cells": raw_cells,
        # برای حفظ سازگاری با کدهای فعلی، نام کلید را نگه می‌داریم.
        "clean_cells": inverted_cells,
        "inverted_cells": inverted_cells,
        "empty_flags": empty_flags,
        "cell_filenames": cell_filenames,
        "cells_montage": inverted_montage,
        "cells_montage_path": montage_path,
    }