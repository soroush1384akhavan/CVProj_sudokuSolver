from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
# اضافه کردن کتابخانه استاندارد برای آگمنتیشن محتاطانه
from torchvision import transforms

BACKEND_ROOT = Path(__file__).resolve().parents[5]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from common.images import imread_color
from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells


class GeneratedSudokuCellDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        cache_dir: str | Path | None = None,
        include_empty_cells: bool = False,
        transform: Any | None = None,
        refresh_cache: bool = False,
        label_field: str = "puzzle",
        apply_safe_augmentation: bool = False,  # فلگ جدید برای فعال‌سازی آگمنتیشن امن داخلی
    ) -> None:
        self.root_dir = Path(root_dir)
        self.labels_path = self.root_dir / "labels.csv"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.root_dir / "extracted_cells"
        self.include_empty_cells = include_empty_cells
        self.transform = transform
        self.refresh_cache = refresh_cache
        self.label_field = label_field
        self.apply_safe_augmentation = apply_safe_augmentation

        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Generated Sudoku labels.csv not found: {self.labels_path}")

        # تعریف آگمنتیشن بسیار محتاطانه مخصوص اعداد
        # این ترنسفورم فقط چرخش‌های جزیی دست‌اندازها و جابجایی‌های خیلی کم را شبیه‌سازی می‌کند
        self.safe_augment = transforms.Compose([
            transforms.RandomRotation(degrees=(-7, 7)),  # حداکثر ۷ درجه چرخش (برای جلوگیری از تبدیل ۶ به ۹)
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.05, 0.05),  # حداکثر ۵ درصد جابجایی افقی و عمودی عدد در خانه سودوکو
                scale=(0.95, 1.05)       # بزرگنمایی یا کوچکنمایی بسیار نامحسوس
            ),
        ])

        self.samples: list[tuple[Path, int]] = []
        self._build_index()

        if not self.samples:
            raise ValueError(f"No generated Sudoku cells collected from: {self.root_dir}")

    def _build_index(self) -> None:
        with self.labels_path.open("r", encoding="utf-8", newline="") as labels_file:
            reader = csv.DictReader(labels_file)

            for row in reader:
                image_path = self.root_dir / row["filename"].replace("\\", "/")
                labels = self._parse_flat_board(row[self.label_field])
                cell_paths = self._ensure_extracted_cells(image_path)

                for index, label in enumerate(labels):
                    if label == 0 and not self.include_empty_cells:
                        continue

                    self.samples.append((cell_paths[index], label))

    @staticmethod
    def _parse_flat_board(value: str) -> list[int]:
        digits = [int(char) for char in value.strip() if char.isdigit()]

        if len(digits) != 81:
            raise ValueError("Generated Sudoku board label must contain exactly 81 digits.")

        return digits

    def _ensure_extracted_cells(self, image_path: Path) -> list[Path]:
        if not image_path.is_file():
            raise FileNotFoundError(f"Generated Sudoku image not found: {image_path}")

        relative_stem = image_path.relative_to(self.root_dir).with_suffix("")
        output_dir = self.cache_dir / relative_stem
        cells_dir = output_dir / "cells"
        cell_paths = [cells_dir / f"cell_{index:02d}.png" for index in range(81)]

        if self.refresh_cache or not all(path.is_file() for path in cell_paths):
            image_bgr = imread_color(image_path)
            phase1 = preprocess_image(image_bgr, output_dir)
            phase2 = find_sudoku_grid(phase1["threshold"], output_dir)  # type: ignore[arg-type]
            extract_cells(phase2["warped_binary"], output_dir)  # type: ignore[arg-type]

        return cell_paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")

        # اگر قابلیت آگمنتیشن امن فعال بود و ترنسفورم خارجی نداشتیم (یا مکمل آن)
        if self.apply_safe_augmentation:
            image = self.safe_augment(image)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image_array = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image_array).unsqueeze(0)

        return image, label