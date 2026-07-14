from pathlib import Path
import cv2
import numpy as np

from app.config import settings
from common.images import save_image


def normalize_binary_mask(
    binary: np.ndarray,
    invert: bool | None = None,
) -> np.ndarray:
    """
    Normalize polarity so ink/digit pixels are 255 and background is 0.

    IMPORTANT: pass `invert` explicitly (decided ONCE for the whole board)
    whenever possible. Deciding per-cell from a local white-pixel ratio is
    fragile: uneven lighting/glare on a single cell can push that cell's
    local white ratio past 0.5 even when the rest of the board doesn't,
    flipping polarity for that one cell and erasing its digit. Falling back
    to the per-cell heuristic below is only a last resort when no global
    decision is available (e.g. this function is called standalone).
    """

    mask = binary

    if invert is None:
        white_ratio = float(np.count_nonzero(mask) / mask.size)
        invert = white_ratio > 0.50

    if invert:
        mask = cv2.bitwise_not(mask)

    return mask


def center_and_scale_digit(
    digit_gray: np.ndarray,
    digit_mask: np.ndarray,
    output_size: int = 28,
    padding: int = 4,
) -> tuple[np.ndarray, np.ndarray]:

    canvas_gray = np.zeros((output_size, output_size), dtype=np.uint8)
    canvas_mask = np.zeros((output_size, output_size), dtype=np.uint8)

    if digit_mask is None or np.count_nonzero(digit_mask) == 0:
        return canvas_gray, canvas_mask

    pts = np.argwhere(digit_mask > 0)
    if pts.size == 0:
        return canvas_gray, canvas_mask

    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)

    digit_w = (x_max - x_min) + 1
    digit_h = (y_max - y_min) + 1

    crop_gray = digit_gray[y_min : y_max + 1, x_min : x_max + 1]
    crop_mask = digit_mask[y_min : y_max + 1, x_min : x_max + 1]

    max_content_size = output_size - (padding * 2)
    if max_content_size <= 0:
        max_content_size = output_size

    scale = min(max_content_size / digit_w, max_content_size / digit_h)

    new_w = max(1, int(round(digit_w * scale)))
    new_h = max(1, int(round(digit_h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_gray = cv2.resize(crop_gray, (new_w, new_h), interpolation=interpolation)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=interpolation)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2

    canvas_gray[
        y_offset : y_offset + new_h, x_offset : x_offset + new_w
    ] = resized_gray
    canvas_mask[
        y_offset : y_offset + new_h, x_offset : x_offset + new_w
    ] = resized_mask

    return canvas_gray, canvas_mask


def is_cell_empty(
    inverted_gray: np.ndarray,
    min_ink_ratio: float | None = None,
    margin_ratio: float | None = None,
) -> bool:
    if inverted_gray.ndim != 2:
        raise ValueError("inverted_gray must be a single-channel image")

    min_ink_ratio = (
        float(
            settings.get(
                "cell_extraction.empty_pixel_ratio_threshold",
                0.035,
            )
        )
        if min_ink_ratio is None
        else float(min_ink_ratio)
    )

    margin_ratio = (
        float(
            settings.get(
                "cell_extraction.empty_margin_ratio",
                0.08,
            )
        )
        if margin_ratio is None
        else float(margin_ratio)
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

    _, binary = cv2.threshold(
        inner,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    inner_height, inner_width = binary.shape

    min_component_area = max(
        4,
        int(round(binary.size * min_ink_ratio)),
    )

    (
        component_count,
        _,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])

        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[label, cv2.CC_STAT_AREA])

        right = x + component_width
        bottom = y + component_height

        touches_border = (
            x <= 0
            or y <= 0
            or right >= inner_width
            or bottom >= inner_height
        )

        if touches_border:
            component_extent = component_area / max(1, component_width * component_height)
            if component_extent > 0.85:
                continue

        if component_width < 2:
            continue

        if component_height < 4:
            continue

        if component_area >= min_component_area:
            return False

    return True


def detect_white_edge_margins(
    mask: np.ndarray,
    scan_depth: int = 4,
    min_white_ratio: float = 0.70,
) -> tuple[int, int, int, int]:
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")

    height, width = mask.shape

    scan_depth = max(
        0,
        min(
            int(scan_depth),
            height // 2,
            width // 2,
        ),
    )

    min_white_ratio = float(np.clip(min_white_ratio, 0.0, 1.0))

    def is_white_margin(line: np.ndarray) -> bool:
        if line.size == 0:
            return False

        white_pixels = np.count_nonzero(line > 127)

        white_ratio = white_pixels / line.size

        return white_ratio >= min_white_ratio

    top = 0
    for index in range(scan_depth):
        row = mask[index, :]

        if not is_white_margin(row):
            break

        top += 1

    bottom = 0
    for index in range(scan_depth):
        row = mask[height - 1 - index, :]

        if not is_white_margin(row):
            break

        bottom += 1

    left = 0
    for index in range(scan_depth):
        column = mask[:, index]

        if not is_white_margin(column):
            break

        left += 1

    right = 0
    for index in range(scan_depth):
        column = mask[:, width - 1 - index]

        if not is_white_margin(column):
            break

        right += 1

    return top, bottom, left, right


def zero_margin_and_shift_inward(
    image: np.ndarray,
    margins: tuple[int, int, int, int],
) -> np.ndarray:

    if image.ndim != 2:
        raise ValueError("image must be a single-channel image")

    top, bottom, left, right = margins
    height, width = image.shape

    top = max(0, min(int(top), height))
    bottom = max(0, min(int(bottom), height))
    left = max(0, min(int(left), width))
    right = max(0, min(int(right), width))

    result = image.copy()

    if top > 0:
        result[:top, :] = 0

    if bottom > 0:
        result[height - bottom :, :] = 0

    if left > 0:
        result[:, :left] = 0

    if right > 0:
        result[:, width - right :] = 0

    shift_y = bottom - top
    shift_x = right - left

    shifted = np.zeros_like(result)

    source_y1 = max(0, -shift_y)
    source_y2 = min(height, height - shift_y)

    source_x1 = max(0, -shift_x)
    source_x2 = min(width, width - shift_x)

    destination_y1 = max(0, shift_y)
    destination_y2 = destination_y1 + (source_y2 - source_y1)

    destination_x1 = max(0, shift_x)
    destination_x2 = destination_x1 + (source_x2 - source_x1)

    if source_y2 > source_y1 and source_x2 > source_x1:
        shifted[
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = result[
            source_y1:source_y2,
            source_x1:source_x2,
        ]

    return shifted


def clean_cell(
    cell: np.ndarray,
    binary_cell: np.ndarray,
    margin_ratio: float | None = None,
    output_size: int | None = None,
    min_area_ratio: float | None = None,
    invert: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    del margin_ratio
    del min_area_ratio

    gray = cell.copy()

    expected_size = int(settings.get("cell_extraction.digit_input_size", 28))

    output_size = expected_size if output_size is None else int(output_size)

    # `invert` should be decided ONCE for the whole board (see extract_cells)
    # and passed down here, rather than re-derived per cell — see
    # normalize_binary_mask's docstring for why the per-cell heuristic is
    # unreliable under uneven lighting.
    mask = normalize_binary_mask(binary_cell, invert=invert)

    mask = np.where(
        mask > 0,
        255,
        0,
    ).astype(np.uint8)

    inverted_gray = cv2.bitwise_not(gray)

    edge_scan_depth = int(settings.get("cell_extraction.edge_margin_scan_depth", 9))

    edge_min_white_ratio = float(
        settings.get(
            "cell_extraction.edge_margin_min_white_ratio",
            0.4,
        )
    )

    margins = detect_white_edge_margins(
        mask,
        scan_depth=edge_scan_depth,
        min_white_ratio=edge_min_white_ratio,
    )

    shifted_mask = zero_margin_and_shift_inward(
        mask,
        margins,
    )

    shifted_gray = zero_margin_and_shift_inward(
        inverted_gray,
        margins,
    )

    cleaned = np.where(
        shifted_mask > 0,
        shifted_gray,
        0,
    ).astype(np.uint8)

    empty = is_cell_empty(
        inverted_gray=cleaned,
    )

    if empty:
        empty_image = np.zeros(
            (output_size, output_size),
            dtype=np.uint8,
        )

        return empty_image, np.zeros((output_size, output_size), dtype=np.uint8), True

    nonzero = cleaned[cleaned > 0]

    if nonzero.size > 0:
        low = float(np.percentile(nonzero, 5))
        high = float(np.percentile(nonzero, 99))

        if high > low:
            cleaned_float = cleaned.astype(np.float32)

            cleaned_float = (cleaned_float - low) * (255.0 / (high - low))

            cleaned = np.clip(
                cleaned_float,
                0,
                255,
            ).astype(np.uint8)

    return cleaned, mask, False


def make_montage(
    cells: list[np.ndarray],
    cell_size: int | None = None,
) -> np.ndarray:
    del cell_size

    if len(cells) != 81:
        raise ValueError(f"Expected 81 cells, received {len(cells)}")

    first_cell = cells[0]

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
            raise ValueError(f"Cell {index} is not grayscale")

        if cell.shape != (
            cell_height,
            cell_width,
        ):
            raise ValueError(
                f"Cell {index} has shape {cell.shape}, expected {(cell_height, cell_width)}"
            )

        row, column = divmod(index, 9)

        y1 = row * cell_height
        y2 = y1 + cell_height

        x1 = column * cell_width
        x2 = x1 + cell_width

        montage[y1:y2, x1:x2] = cell

    return cv2.cvtColor(
        montage,
        cv2.COLOR_GRAY2BGR,
    )


def extract_cells(
    warped_bgr: np.ndarray,
    warped_binary: np.ndarray,
    output_dir: Path,
) -> dict[
    str,
    list[np.ndarray]
    | np.ndarray
    | str
    | list[bool]
    | list[str],
]:


    output_dir = Path(output_dir)

    cells_dir = output_dir / "cells"
    masks_dir = output_dir / "cell_masks"

    cells_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    masks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = warped_bgr.shape[:2]

    binary_height, binary_width = warped_binary.shape[:2]

    if binary_height != height or binary_width != width:
        raise ValueError(
            f"warped_bgr and warped_binary sizes differ: BGR={width}x{height}, binary={binary_width}x{binary_height}"
        )

    expected_cell_size = int(settings.get("cell_extraction.digit_input_size", 28))

    expected_board_size = expected_cell_size * 9

    if height != expected_board_size or width != expected_board_size:
        raise ValueError(
            f"Expected warped board size {expected_board_size}x{expected_board_size}, but received {width}x{height}."
        )

    cell_height = height // 9
    cell_width = width // 9

    board_white_ratio = float(np.count_nonzero(warped_binary) / warped_binary.size)
    board_invert = board_white_ratio > 0.50

    raw_cells: list[np.ndarray] = []
    binary_cells: list[np.ndarray] = []
    digit_masks: list[np.ndarray] = []
    cleaned_cells: list[np.ndarray] = []

    empty_flags: list[bool] = []
    cell_filenames: list[str] = []

    save_cells = bool(settings.get("cell_extraction.save_cells", True))

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

            binary_cell = warped_binary[
                y1:y2,
                x1:x2,
            ]

            if raw.ndim == 3:
                raw_gray = cv2.cvtColor(
                    raw,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                raw_gray = raw.copy()

            if binary_cell.ndim == 3:
                binary_gray = cv2.cvtColor(
                    binary_cell,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                binary_gray = binary_cell.copy()

            cleaned, digit_mask, empty = clean_cell(
                cell=raw_gray,
                binary_cell=binary_gray,
                invert=board_invert,
            )

            index = row * 9 + column

            cell_filename = f"cells/cell_{index:02d}.png"
            mask_filename = f"cell_masks/mask_{index:02d}.png"

            if save_cells:
                save_image(
                    output_dir / cell_filename,
                    cleaned,
                )

                save_image(
                    output_dir / mask_filename,
                    digit_mask,
                )

            raw_cells.append(raw_gray)
            binary_cells.append(normalize_binary_mask(binary_gray, invert=board_invert))
            digit_masks.append(digit_mask)
            cleaned_cells.append(cleaned)

            empty_flags.append(empty)
            cell_filenames.append(cell_filename)

    save_image(
        output_dir / "06_raw_cells_montage.png",
        make_montage(raw_cells),
    )

    save_image(
        output_dir / "07_binary_cells_montage.png",
        make_montage(binary_cells),
    )

    save_image(
        output_dir / "08_digit_masks_montage.png",
        make_montage(digit_masks),
    )

    montage_path = "09_masked_grayscale_montage.png"

    cleaned_montage = make_montage(cleaned_cells)

    save_image(
        output_dir / montage_path,
        cleaned_montage,
    )

    save_image(
        output_dir / "07_inverted_grayscale_montage.png",
        cleaned_montage,
    )

    return {
        "raw_cells": raw_cells,
        "binary_cells": binary_cells,
        "digit_masks": digit_masks,
        "clean_cells": cleaned_cells,
        "inverted_cells": cleaned_cells,
        "empty_flags": empty_flags,
        "cell_filenames": cell_filenames,
        "cells_montage": cleaned_montage,
        "cells_montage_path": montage_path,
    }