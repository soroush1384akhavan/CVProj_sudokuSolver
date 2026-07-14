from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    BACKEND_ROOT = Path(__file__).resolve().parents[5]
except IndexError:
    BACKEND_ROOT = Path(__file__).resolve().parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from common.images import imread_color
from phases.phase_01_preprocessing.preprocess import preprocess_image
from phases.phase_02_grid_detection.grid_detection import find_sudoku_grid
from phases.phase_03_cell_extraction.cell_extraction import extract_cells


SUPPORTED_LANGUAGES = ("en", "fa")

DIGIT_TRANSLATION = str.maketrans(
    {
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


@dataclass(frozen=True)
class GeneratedSudokuCellSample:
    cell_path: Path
    label: int
    language: str
    source_image_path: Path
    cell_index: int
    kind: str


class GeneratedSudokuCellDataset(Dataset):
 
    def __init__(
        self,
        root_dir: str | Path,
        cache_dir: str | Path | None = None,
        languages: Sequence[str] = SUPPORTED_LANGUAGES,
        include_empty_cells: bool = False,
        transform: Any | None = None,
        refresh_cache: bool = False,
        label_field: str = "puzzle",
        apply_safe_augmentation: bool = False,
        return_language: bool = True,
        return_metadata: bool = False,
        strict_languages: bool = True,
        kinds: Iterable[str] | None = None,
        skip_visually_empty_nonzero: bool = True,
        content_crop_ratio: float = 0.12,
        min_content_area_ratio: float = 0.006,
        min_component_area_ratio: float = 0.0015,
        log_skipped_samples: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else self.root_dir / "extracted_cells"
        )
        self.languages = self._validate_languages(languages)
        self.include_empty_cells = include_empty_cells
        self.transform = transform
        self.refresh_cache = refresh_cache
        self.label_field = label_field
        self.apply_safe_augmentation = apply_safe_augmentation
        self.return_language = return_language
        self.return_metadata = return_metadata
        self.strict_languages = strict_languages
        self.kinds = set(kinds) if kinds is not None else None
        self.skip_visually_empty_nonzero = skip_visually_empty_nonzero
        self.content_crop_ratio = float(content_crop_ratio)
        self.min_content_area_ratio = float(min_content_area_ratio)
        self.min_component_area_ratio = float(min_component_area_ratio)
        self.log_skipped_samples = log_skipped_samples

        if not 0.0 <= self.content_crop_ratio < 0.45:
            raise ValueError("content_crop_ratio must be between 0.0 and 0.45.")
        if not 0.0 <= self.min_content_area_ratio <= 1.0:
            raise ValueError("min_content_area_ratio must be between 0.0 and 1.0.")
        if not 0.0 <= self.min_component_area_ratio <= 1.0:
            raise ValueError("min_component_area_ratio must be between 0.0 and 1.0.")

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Generated Sudoku root directory not found: {self.root_dir}")

        self.safe_augment = transforms.Compose(
            [
                transforms.RandomRotation(degrees=(-7, 7)),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
            ]
        )

        self.samples: list[GeneratedSudokuCellSample] = []
        self.skipped_visually_empty_nonzero = 0
        self.skipped_visually_empty_examples: list[tuple[Path, int, int, str]] = []
        self.language_roots = self._discover_language_roots()
        self._build_index()

        if self.skip_visually_empty_nonzero:
            print(
                "[DATASET] visually empty cells with non-zero labels skipped: "
                f"{self.skipped_visually_empty_nonzero}"
            )
            if self.log_skipped_samples and self.skipped_visually_empty_examples:
                print("[DATASET] first skipped examples:")
                for source_path, cell_index, label, language in self.skipped_visually_empty_examples:
                    print(
                        "  "
                        f"language={language} label={label} cell={cell_index} "
                        f"source={source_path}"
                    )

        if not self.samples:
            raise ValueError(
                "No generated Sudoku cells collected from "
                f"{self.root_dir} for languages={self.languages}."
            )

    @staticmethod
    def _validate_languages(languages: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(str(language).lower() for language in languages))

        if not normalized:
            raise ValueError("At least one language must be requested.")

        unsupported = [
            language for language in normalized if language not in SUPPORTED_LANGUAGES
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported languages: {unsupported}. "
                f"Supported languages: {SUPPORTED_LANGUAGES}"
            )

        return normalized

    def _discover_language_roots(self) -> dict[str, Path]:
        discovered: dict[str, Path] = {}
        missing: dict[str, Path] = {}

        # حالت استاندارد: root_dir همان ریشه run است و en/fa زیر آن قرار دارند.
        for language in self.languages:
            language_root = self.root_dir / language
            labels_path = language_root / "labels.csv"

            if labels_path.is_file():
                discovered[language] = language_root
            else:
                missing[language] = labels_path

        # سازگاری با حالت قدیمی: root_dir مستقیماً پوشه en یا fa باشد.
        if not discovered and (self.root_dir / "labels.csv").is_file():
            inferred_language = self.root_dir.name.lower()

            if inferred_language not in SUPPORTED_LANGUAGES:
                if len(self.languages) != 1:
                    raise ValueError(
                        "When root_dir directly contains labels.csv, either name the folder "
                        "'en'/'fa' or request exactly one language."
                    )
                inferred_language = self.languages[0]

            if inferred_language in self.languages:
                discovered[inferred_language] = self.root_dir
                missing.pop(inferred_language, None)

        if self.strict_languages:
            unresolved = {
                language: labels_path
                for language, labels_path in missing.items()
                if language not in discovered
            }
            if unresolved:
                details = ", ".join(
                    f"{language}: {labels_path}"
                    for language, labels_path in unresolved.items()
                )
                raise FileNotFoundError(
                    f"Missing requested language labels.csv files: {details}"
                )
        else:
            for language, labels_path in missing.items():
                if language not in discovered:
                    print(
                        f"[DATASET] language skipped; labels.csv not found: {labels_path}"
                    )

        if not discovered:
            raise FileNotFoundError(
                f"No language labels.csv files found under: {self.root_dir}"
            )

        return discovered

    def _build_index(self) -> None:
        for language, language_root in self.language_roots.items():
            labels_path = language_root / "labels.csv"

            with labels_path.open("r", encoding="utf-8", newline="") as labels_file:
                reader = csv.DictReader(labels_file)

                required_fields = {"filename", self.label_field}
                missing_fields = required_fields - set(reader.fieldnames or [])
                if missing_fields:
                    raise ValueError(
                        f"Missing columns in {labels_path}: {sorted(missing_fields)}"
                    )

                for row_number, row in enumerate(reader, start=2):
                    row_style = (row.get("style") or language).strip().lower()
                    if row_style and row_style != language:
                        raise ValueError(
                            f"Language mismatch in {labels_path}:{row_number}. "
                            f"Folder language={language}, row style={row_style}."
                        )

                    kind = (row.get("kind") or "unknown").strip().lower()
                    if self.kinds is not None and kind not in self.kinds:
                        continue

                    image_path = language_root / row["filename"].replace("\\", "/")
                    labels = self._parse_flat_board(row[self.label_field])
                    cell_paths = self._ensure_extracted_cells(
                        image_path=image_path,
                        language=language,
                        language_root=language_root,
                    )

                    if len(cell_paths) != 81:
                        raise ValueError(
                            f"Expected 81 extracted cells for {image_path}, "
                            f"got {len(cell_paths)}."
                        )

                    for cell_index, label in enumerate(labels):
                        if label == 0 and not self.include_empty_cells:
                            continue

                        cell_path = cell_paths[cell_index]

                        if (
                            label != 0
                            and self.skip_visually_empty_nonzero
                            and not self._cell_has_visible_content(cell_path)
                        ):
                            self.skipped_visually_empty_nonzero += 1

                            if len(self.skipped_visually_empty_examples) < 20:
                                self.skipped_visually_empty_examples.append(
                                    (image_path, cell_index, label, language)
                                )

                            continue

                        self.samples.append(
                            GeneratedSudokuCellSample(
                                cell_path=cell_path,
                                label=label,
                                language=language,
                                source_image_path=image_path,
                                cell_index=cell_index,
                                kind=kind,
                            )
                        )

    def _cell_has_visible_content(self, cell_path: Path) -> bool:
        """
        بررسی می‌کند که سلول واقعاً محتوای قابل‌مشاهده دارد یا نه.

        برای حذف اثر خط‌های جدول، حاشیهٔ سلول crop می‌شود. سپس با آستانه‌گذاری
        Otsu و connected components، نویزهای پراکنده حذف می‌شوند. نمونه فقط وقتی
        معتبر است که هم مجموع مساحت foreground و هم حداقل یک component معنادار
        از آستانه‌های تعیین‌شده بزرگ‌تر باشند.
        """
        gray = cv2.imread(str(cell_path), cv2.IMREAD_GRAYSCALE)

        if gray is None or gray.size == 0:
            return False

        height, width = gray.shape[:2]
        margin_y = int(round(height * self.content_crop_ratio))
        margin_x = int(round(width * self.content_crop_ratio))

        y1 = margin_y
        y2 = height - margin_y
        x1 = margin_x
        x2 = width - margin_x

        if y2 <= y1 or x2 <= x1:
            roi = gray
        else:
            roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        # رنگ غالب لبه‌های ROI به‌عنوان پس‌زمینه در نظر گرفته می‌شود.
        edge_pixels = np.concatenate(
            [roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]]
        )
        background_is_light = float(np.median(edge_pixels)) >= 127.0

        threshold_mode = (
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            if background_is_light
            else cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, foreground = cv2.threshold(roi, 0, 255, threshold_mode)

        # حذف لکه‌های تک‌پیکسلی و نویزهای بسیار ریز.
        kernel = np.ones((2, 2), dtype=np.uint8)
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )

        roi_area = int(roi.shape[0] * roi.shape[1])
        if roi_area <= 0 or component_count <= 1:
            return False

        min_component_area = max(3, int(round(roi_area * self.min_component_area_ratio)))
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        significant_areas = component_areas[component_areas >= min_component_area]

        if significant_areas.size == 0:
            return False

        significant_area_ratio = float(significant_areas.sum()) / float(roi_area)
        largest_component_ratio = float(significant_areas.max()) / float(roi_area)

        return (
            significant_area_ratio >= self.min_content_area_ratio
            and largest_component_ratio >= self.min_component_area_ratio
        )

    @staticmethod
    def _parse_flat_board(value: str) -> list[int]:
        normalized = value.strip().translate(DIGIT_TRANSLATION)
        digits = [int(char) for char in normalized if char.isdigit()]

        if len(digits) != 81:
            raise ValueError(
                "Generated Sudoku board label must contain exactly 81 digits. "
                f"Parsed {len(digits)} digits instead."
            )

        return digits

    def _ensure_extracted_cells(
        self,
        image_path: Path,
        language: str,
        language_root: Path,
    ) -> list[Path]:
        if not image_path.is_file():
            raise FileNotFoundError(f"Generated Sudoku image not found: {image_path}")

        relative_stem = image_path.relative_to(language_root).with_suffix("")
        output_dir = self.cache_dir / language / relative_stem
        cells_dir = output_dir / "cells"
        cell_paths = [cells_dir / f"cell_{index:02d}.png" for index in range(81)]

        if self.refresh_cache or not all(path.is_file() for path in cell_paths):
            image_bgr = imread_color(image_path)

            if image_bgr is None:
                raise ValueError(f"Failed to read generated Sudoku image: {image_path}")

            output_dir.mkdir(parents=True, exist_ok=True)

            preprocessed = preprocess_image(image_bgr, output_dir)

            grid_result = find_sudoku_grid(
                original_bgr=preprocessed["original"],
                preprocessed_binary=preprocessed["threshold"],
                output_dir=output_dir,
            )

            if not grid_result.get("found", False):
                print(f"[DATASET] Grid not found; fallback warp used: {image_path}")

            extract_cells(
                warped_bgr=grid_result["warped"],
                warped_binary=grid_result["warped_binary"],
                output_dir=output_dir,
            )

            missing_cells = [path for path in cell_paths if not path.is_file()]
            if missing_cells:
                raise RuntimeError(
                    f"Cell extraction did not create all 81 cells for {image_path}. "
                    f"Missing: {len(missing_cells)} files."
                )

        return cell_paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        with Image.open(sample.cell_path) as image_file:
            image = image_file.convert("L")

        if self.apply_safe_augmentation:
            image = self.safe_augment(image)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image_array).unsqueeze(0)

        if self.return_metadata:
            return {
                "image": image,
                "label": sample.label,
                "language": sample.language,
                "language_id": 0 if sample.language == "en" else 1,
                "cell_index": sample.cell_index,
                "cell_path": str(sample.cell_path),
                "source_image_path": str(sample.source_image_path),
                "kind": sample.kind,
            }

        if self.return_language:
            return image, sample.label, sample.language

        return image, sample.label