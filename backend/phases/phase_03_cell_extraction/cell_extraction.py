from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from common.images import save_image


def normalize_binary_mask(binary: np.ndarray) -> np.ndarray:
    """
    ماسک را به تصویر uint8 شامل 0 و 255 تبدیل می‌کند.

    فرض می‌کنیم foreground نسبت به background مساحت کمتری دارد.
    اگر بیشتر تصویر سفید باشد، polarity را معکوس می‌کنیم.
    """
    if binary.ndim == 3:
        binary = cv2.cvtColor(
            binary,
            cv2.COLOR_BGR2GRAY,
        )
    elif binary.ndim != 2:
        raise ValueError(
            f"Unsupported binary mask shape: {binary.shape}"
        )

    mask = np.where(
        binary > 127,
        255,
        0,
    ).astype(np.uint8)

    white_ratio = float(
        np.count_nonzero(mask) / mask.size
    )

    # اگر background سفید باشد، foreground را سفید می‌کنیم.
    if white_ratio > 0.50:
        mask = cv2.bitwise_not(mask)

    return mask


def extract_digit_bbox(
    binary_cell: np.ndarray,
    margin_ratio: float | None = None,
    min_area_ratio: float | None = None,
) -> tuple[tuple[int, int, int, int] | None, bool]:
    """
    از binary مربوط به یک سلول، فقط محدوده‌ی رقم (bbox) را استخراج می‌کند.

    خروجی:
        ((x1, y1, x2, y2), empty)
    """
    mask = normalize_binary_mask(binary_cell)

    height, width = mask.shape

    margin_ratio = (
        float(
            settings.get(
                "cell_extraction.mask_margin_ratio",
                0.10,
            )
        )
        if margin_ratio is None
        else float(margin_ratio)
    )

    min_area_ratio = (
        float(
            settings.get(
                "cell_extraction.mask_min_area_ratio",
                0.015,
            )
        )
        if min_area_ratio is None
        else float(min_area_ratio)
    )

    margin_y = max(1, int(round(height * margin_ratio)))
    margin_x = max(1, int(round(width * margin_ratio)))

    # حذف خطوط جدول در حاشیه
    mask[:margin_y, :] = 0
    mask[height - margin_y:, :] = 0
    mask[:, :margin_x] = 0
    mask[:, width - margin_x:] = 0

    (
        component_count,
        component_labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    min_main_area = max(
        3,
        int(round(mask.size * min_area_ratio)),
    )

    candidates: list[
        tuple[float, int, int, int, int, int, int]
    ] = []

    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_main_area:
            continue

        horizontal_line = (
            component_width >= int(width * 0.70)
            and component_height <= 2
        )

        vertical_line = (
            component_height >= int(height * 0.70)
            and component_width <= 2
        )

        if horizontal_line or vertical_line:
            continue

        center_x = x + component_width / 2.0
        center_y = y + component_height / 2.0

        dx = abs(center_x - width / 2.0) / max(width / 2.0, 1.0)
        dy = abs(center_y - height / 2.0) / max(height / 2.0, 1.0)

        center_distance = min(
            1.0,
            float(np.hypot(dx, dy) / np.sqrt(2.0)),
        )

        center_score = 1.0 - center_distance

        score = float(area) * (0.85 + 0.15 * center_score)

        candidates.append(
            (
                score,
                label,
                x,
                y,
                component_width,
                component_height,
                area,
            )
        )

    if not candidates:
        return None, True

    (
        _,
        main_label,
        main_x,
        main_y,
        main_width,
        main_height,
        _,
    ) = max(candidates, key=lambda item: item[0])

    main_right = main_x + main_width
    main_bottom = main_y + main_height

    fragment_padding = int(
        settings.get(
            "cell_extraction.mask_fragment_padding",
            3,
        )
    )

    expanded_x1 = max(0, main_x - fragment_padding)
    expanded_y1 = max(0, main_y - fragment_padding)
    expanded_x2 = min(width, main_right + fragment_padding)
    expanded_y2 = min(height, main_bottom + fragment_padding)

    keep_labels = {main_label}

    minimum_fragment_area = int(
        settings.get(
            "cell_extraction.mask_min_fragment_area",
            2,
        )
    )

    for label in range(1, component_count):
        if label == main_label:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < minimum_fragment_area:
            continue

        right = x + component_width
        bottom = y + component_height

        near_main_component = not (
            right < expanded_x1
            or x > expanded_x2
            or bottom < expanded_y1
            or y > expanded_y2
        )

        if near_main_component:
            keep_labels.add(label)

    kept_mask = np.where(
        np.isin(component_labels, list(keep_labels)),
        255,
        0,
    ).astype(np.uint8)

    ys, xs = np.where(kept_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None, True

    bbox_padding = int(
        settings.get(
            "cell_extraction.digit_bbox_padding",
            1,
        )
    )

    x1 = max(0, int(xs.min()) - bbox_padding)
    y1 = max(0, int(ys.min()) - bbox_padding)
    x2 = min(width, int(xs.max()) + 1 + bbox_padding)
    y2 = min(height, int(ys.max()) + 1 + bbox_padding)

    return (x1, y1, x2, y2), False

def center_digit_grayscale(
    digit_gray: np.ndarray,
    output_size: int,
    content_size: int | None = None,
) -> np.ndarray:
    if digit_gray.ndim != 2:
        raise ValueError("digit_gray must be 2D")

    if digit_gray.size == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    h, w = digit_gray.shape

    if h <= 0 or w <= 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    content_size = (
        int(settings.get("cell_extraction.digit_content_size", 20))
        if content_size is None
        else int(content_size)
    )

    content_size = max(1, min(content_size, output_size))

    scale = min(content_size / w, content_size / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    resized = cv2.resize(
        digit_gray,
        (new_w, new_h),
        interpolation=interpolation,
    )

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return canvas

def is_cell_empty(
    inverted_gray: np.ndarray,
    min_ink_ratio: float | None = None,
    margin_ratio: float | None = None,
) -> bool:
    """
    تشخیص خالی‌بودن سلول با یک ماسک موقت Otsu.

    تصویر خروجی تغییر نمی‌کند. ناحیه‌ی اطراف سلول نادیده گرفته
    می‌شود تا خطوط جدول باعث non-empty شدن خانه نشوند.
    """
    if inverted_gray.ndim != 2:
        raise ValueError(
            "inverted_gray must be a single-channel image"
        )

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
                0.16,
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

    # فقط برای empty detection؛ خروجی مدل binary نمی‌شود.
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

        component_width = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )
        component_height = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )
        component_area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        right = x + component_width
        bottom = y + component_height

        touches_border = (
            x <= 0
            or y <= 0
            or right >= inner_width
            or bottom >= inner_height
        )

        if touches_border:
            continue

        # نویز کوچک یا خط خیلی باریک را رقم در نظر نگیر.
        if component_width < 2:
            continue

        if component_height < 4:
            continue

        if component_area >= min_component_area:
            return False

    return True

def clean_cell(
    cell: np.ndarray,
    binary_cell: np.ndarray,
    margin_ratio: float | None = None,
    output_size: int | None = None,
    min_area_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    binary فقط برای پیدا کردن bbox رقم استفاده می‌شود.
    خروجی نهایی از grayscale اصلی گرفته می‌شود، نه از mask.

    خروجی:
        cleaned_grayscale
        debug_mask
        empty
    """
    if cell.ndim == 3:
        gray = cv2.cvtColor(
            cell,
            cv2.COLOR_BGR2GRAY,
        )
    elif cell.ndim == 2:
        gray = cell.copy()
    else:
        raise ValueError(f"Unsupported cell shape: {cell.shape}")

    if binary_cell.ndim == 3:
        binary_gray = cv2.cvtColor(
            binary_cell,
            cv2.COLOR_BGR2GRAY,
        )
    elif binary_cell.ndim == 2:
        binary_gray = binary_cell.copy()
    else:
        raise ValueError(
            f"Unsupported binary cell shape: {binary_cell.shape}"
        )

    expected_size = int(
        settings.get(
            "cell_extraction.digit_input_size",
            28,
        )
    )

    output_size = expected_size if output_size is None else int(output_size)

    expected_shape = (expected_size, expected_size)

    if gray.shape != expected_shape:
        raise ValueError(
            f"Expected grayscale cell shape {expected_shape}, received {gray.shape}"
        )

    if binary_gray.shape != expected_shape:
        raise ValueError(
            f"Expected binary cell shape {expected_shape}, received {binary_gray.shape}"
        )

    # رقم روشن، پس‌زمینه تیره
    inverted_gray = cv2.bitwise_not(gray)

    digit_bbox, empty = extract_digit_bbox(
        binary_cell=binary_gray,
        margin_ratio=margin_ratio,
        min_area_ratio=min_area_ratio,
    )

    # debug mask فقط برای ذخیره و مشاهده
    debug_mask = normalize_binary_mask(binary_gray)
    
    empty = is_cell_empty(
        inverted_gray=inverted_gray,
    )
    if empty:
        return (
            np.zeros(
                (output_size, output_size),
                dtype=np.uint8,
            ),
            debug_mask,
            True,
        )

    if empty or digit_bbox is None:
        empty_image = np.zeros_like(inverted_gray, dtype=np.uint8)
        return empty_image, debug_mask, True

    x1, y1, x2, y2 = digit_bbox

    # فقط bbox از grayscale واقعی crop می‌شود
    digit_gray = inverted_gray[y1:y2, x1:x2]

    if digit_gray.size == 0:
        empty_image = np.zeros_like(inverted_gray, dtype=np.uint8)
        return empty_image, debug_mask, True

    cleaned = center_digit_grayscale(
        digit_gray=digit_gray,
        output_size=output_size,
    )

    # نرمال‌سازی ملایم contrast
    nonzero = cleaned[cleaned > 0]
    if nonzero.size > 0:
        low = float(np.percentile(nonzero, 5))
        high = float(np.percentile(nonzero, 99))

        if high > low:
            cleaned = cleaned.astype(np.float32)
            cleaned = (cleaned - low) * (255.0 / (high - low))
            cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

    # لبه‌ها صفر
    cleaned[0, :] = 0
    cleaned[-1, :] = 0
    cleaned[:, 0] = 0
    cleaned[:, -1] = 0

    return cleaned, debug_mask, False


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
                f"expected "
                f"{(cell_height, cell_width)}"
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
    """
    warped_bgr:
        تصویر رنگی/خاکستری warp‌شده برای گرفتن شدت واقعی رقم.

    warped_binary:
        تصویر binary همان warp و دقیقاً با همان geometry،
        برای ساختن mask رقم.
    """
    if warped_bgr is None or warped_bgr.size == 0:
        raise ValueError(
            "warped_bgr is empty"
        )

    if warped_binary is None or warped_binary.size == 0:
        raise ValueError(
            "warped_binary is empty"
        )

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

    binary_height, binary_width = (
        warped_binary.shape[:2]
    )

    if (
        binary_height != height
        or binary_width != width
    ):
        raise ValueError(
            f"warped_bgr and warped_binary sizes differ: "
            f"BGR={width}x{height}, "
            f"binary={binary_width}x{binary_height}"
        )

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
            f"but received {width}x{height}."
        )

    cell_height = height // 9
    cell_width = width // 9

    raw_cells: list[np.ndarray] = []
    binary_cells: list[np.ndarray] = []
    digit_masks: list[np.ndarray] = []
    cleaned_cells: list[np.ndarray] = []

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
            )

            index = row * 9 + column

            cell_filename = (
                f"cells/cell_{index:02d}.png"
            )
            mask_filename = (
                f"cell_masks/mask_{index:02d}.png"
            )

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
            binary_cells.append(
                normalize_binary_mask(binary_gray)
            )
            digit_masks.append(digit_mask)
            cleaned_cells.append(cleaned)

            empty_flags.append(empty)
            cell_filenames.append(cell_filename)

    save_image(
        output_dir / "06_raw_cells_montage.png",
        make_montage(raw_cells),
    )

    save_image(
        output_dir
        / "07_binary_cells_montage.png",
        make_montage(binary_cells),
    )

    save_image(
        output_dir
        / "08_digit_masks_montage.png",
        make_montage(digit_masks),
    )

    montage_path = (
        "09_masked_grayscale_montage.png"
    )

    cleaned_montage = make_montage(
        cleaned_cells
    )

    save_image(
        output_dir / montage_path,
        cleaned_montage,
    )

    # نام قدیمی برای سازگاری با بخش‌های دیگر
    save_image(
        output_dir
        / "07_inverted_grayscale_montage.png",
        cleaned_montage,
    )

    return {
        "raw_cells": raw_cells,
        "binary_cells": binary_cells,
        "digit_masks": digit_masks,

        # برای سازگاری با کدهای قبلی
        "clean_cells": cleaned_cells,
        "inverted_cells": cleaned_cells,

        "empty_flags": empty_flags,
        "cell_filenames": cell_filenames,
        "cells_montage": cleaned_montage,
        "cells_montage_path": montage_path,
    }