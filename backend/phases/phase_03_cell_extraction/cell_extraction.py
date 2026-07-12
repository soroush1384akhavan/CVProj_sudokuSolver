from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from app.config import settings
from common.images import save_image


def is_cell_empty(binary: np.ndarray, min_ink_ratio: float | None = None) -> bool:
    min_ink_ratio = float(settings.get("cell_extraction.empty_pixel_ratio_threshold", 0.035)) if min_ink_ratio is None else min_ink_ratio
    ratio = float(np.count_nonzero(binary)) / float(binary.size)
    return ratio < min_ink_ratio


def sharpen(image: np.ndarray, amount: float = 1.5, sigma: float = 3.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def clean_cell(
    cell_bgr: np.ndarray,
    margin_ratio: float | None = None,
    output_size: int | None = None,
    min_area_ratio: float | None = None,
    sharpen_amount: float | None = None,
) -> tuple[np.ndarray, bool]:
    
    margin_ratio = float(settings.get("cell_extraction.margin_ratio", 0.14)) if margin_ratio is None else margin_ratio
    output_size = int(settings.get("cell_extraction.digit_input_size", 28)) if output_size is None else output_size
    min_area_ratio = float(settings.get("cell_extraction.min_digit_area_ratio", 0.03)) if min_area_ratio is None else min_area_ratio

    h, w = cell_bgr.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    cropped = cell_bgr[my : h - my, mx : w - mx]

    if cropped.ndim == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped

    final_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_binary_closed = cv2.morphologyEx(final_binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    ch, cw = final_binary_closed.shape[:2]
    min_area = min_area_ratio * (ch * cw)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_binary_closed, connectivity=8)

    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return np.zeros((output_size, output_size), dtype=np.uint8), True

    # 1. استخراج ماسک فقط برای خودِ عدد (بدون نویزهای اطراف)
    component_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    
    if is_cell_empty(component_mask):
        return np.zeros((output_size, output_size), dtype=np.uint8), True

    # 2. پیدا کردن باندینگ باکس دقیقِ عدد
    x, y, bw, bh = stats[best_label, cv2.CC_STAT_LEFT], stats[best_label, cv2.CC_STAT_TOP], \
                   stats[best_label, cv2.CC_STAT_WIDTH], stats[best_label, cv2.CC_STAT_HEIGHT]
    
    # برش زدن خود عدد از روی ماسک تمیز شده
    digit_crop = component_mask[y:y+bh, x:x+bw]

    # 3. قرار دادن عدد در مرکز یک بوم مربعی جدید (Centering)
    # ایجاد یک بوم مربع بر اساس بزرگترین ضلع عدد
    max_dim = max(bw, bh)
    square_digit = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    # چسباندن عدد در مرکز مربع
    dx = (max_dim - bw) // 2
    dy = (max_dim - bh) // 2
    square_digit[dy:dy+bh, dx:dx+bw] = digit_crop

    # 4. اضافه کردن یک مارجین/پدینگ امن دور عدد (مثلاً ۴ پیکسل) قبل از ری‌سایز نهایی
    # این کار باعث می‌شود عدد کاملاً شبیه MNIST یا داده‌های استاندارد سنتر شده شود.
    pad = 4
    padded_digit = cv2.copyMakeBorder(square_digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # 5. ری‌سایز نهایی به سایز ورودی مدل (مثلا 28x28)
    resized = cv2.resize(padded_digit, (output_size, output_size), interpolation=cv2.INTER_AREA)

    return resized, False


def make_montage(clean_cells: list[np.ndarray], cell_size: int | None = None) -> np.ndarray:
    cell_size = int(settings.get("cell_extraction.montage_cell_size", 40)) if cell_size is None else cell_size

    montage = np.ones((9 * cell_size, 9 * cell_size), dtype=np.uint8) * 255

    for idx, cell in enumerate(clean_cells):
        r, c = divmod(idx, 9)
        resized = cv2.resize(cell, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
        montage[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = resized

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
            clean, empty = clean_cell(raw)

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