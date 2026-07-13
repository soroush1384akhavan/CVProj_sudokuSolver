from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BACKEND_ROOT = Path(__file__).resolve().parents[5]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from common.images import imread_color
from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells


class DatSudokuCellDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        cache_dir: str | Path | None = None,
        include_empty_cells: bool = False,
        transform: Any | None = None,
        refresh_cache: bool = False,
        apply_safe_augmentation: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else self.root_dir / "extracted_cells"
        )
        self.include_empty_cells = include_empty_cells
        self.transform = transform
        self.refresh_cache = refresh_cache
        self.apply_safe_augmentation = apply_safe_augmentation

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"DAT Sudoku root folder not found: {self.root_dir}"
            )

        self.safe_augment = transforms.Compose([
            transforms.RandomRotation(degrees=(-7, 7)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
            ),
        ])

        self.samples: list[tuple[Path, int]] = []
        self._build_index()

        if not self.samples:
            raise ValueError(
                f"No DAT Sudoku cells collected from: {self.root_dir}"
            )

    def _build_index(self) -> None:
        for dat_path in sorted(self.root_dir.glob("*.dat")):
            image_path = dat_path.with_suffix(".jpg")

            if not image_path.is_file():
                continue

            labels = self._read_dat_board(dat_path)
            cell_paths = self._ensure_extracted_cells(image_path)

            for index, label in enumerate(labels):
                if label == 0 and not self.include_empty_cells:
                    continue

                cell_path = cell_paths[index]

                # اگر تمام پیکسل‌های سلول صفر باشند، وارد آموزش نشود
                cell_array = np.array(
                    Image.open(cell_path).convert("L")
                )

                if np.all(cell_array == 0):
                    print(
                        f"Skipped all-zero cell: "
                        f"path={cell_path}, label={label}"
                    )
                    continue

                self.samples.append((cell_path, label))

    @staticmethod
    def _read_dat_board(dat_path: Path) -> list[int]:
        lines = dat_path.read_text(
            encoding="utf-8"
        ).splitlines()

        board_lines = lines[2:11]

        if len(board_lines) != 9:
            raise ValueError(
                f"DAT file must contain 9 board rows: {dat_path}"
            )

        labels: list[int] = []

        for line in board_lines:
            row = [
                int(value)
                for value in line.split()
            ]

            if len(row) != 9:
                raise ValueError(
                    f"DAT board row must contain 9 values: {dat_path}"
                )

            labels.extend(row)

        return labels

    def _ensure_extracted_cells(
        self,
        image_path: Path,
    ) -> list[Path]:
        output_dir = self.cache_dir / image_path.stem
        cells_dir = output_dir / "cells"

        cell_paths = [
            cells_dir / f"cell_{index:02d}.png"
            for index in range(81)
        ]

        if (
            self.refresh_cache
            or not all(path.is_file() for path in cell_paths)
        ):
            image_bgr = imread_color(image_path)

            preprocessed = preprocess_image(
                image_bgr,
                output_dir,
            )

            grid_result = find_sudoku_grid(
                original_bgr=preprocessed["original"],
                preprocessed_binary=preprocessed["threshold"],
                output_dir=output_dir,
            )

            extract_cells(
                warped_bgr=grid_result["warped"],
                warped_binary=grid_result["warped_binary"],
                output_dir=output_dir,
            )

        return cell_paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")

        if self.apply_safe_augmentation:
            image = self.safe_augment(image)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image_array = (
                np.array(image, dtype=np.float32) / 255.0
            )

            image = torch.from_numpy(
                image_array
            ).unsqueeze(0)

        return image, label


CharlsSudokuCellDataset = DatSudokuCellDataset