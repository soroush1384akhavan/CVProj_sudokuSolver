from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
from torchvision.datasets import ImageFolder

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset


BACKEND_ROOT = Path(__file__).resolve().parents[3]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings

try:
    from .datasets.hoda.hoda_dataset import HodaCDBDataset
    from .datasets.mnist.mnist_dataset import MNISTRawDataset
    from .datasets.chars74k.chars74k_dataset import Chars74KFntDataset
except ImportError:
    from phases.phase_04_digit_recognition.src.datasets.hoda.hoda_dataset import HodaCDBDataset
    from phases.phase_04_digit_recognition.src.datasets.mnist.mnist_dataset import MNISTRawDataset
    from phases.phase_04_digit_recognition.src.datasets.chars74k.chars74k_dataset import Chars74KFntDataset


def get_phase4_config() -> dict[str, Any]:
    config = settings.get("digit_recognition", {})

    if not isinstance(config, dict):
        raise KeyError("digit_recognition section must be a mapping in config.yml")

    return config

class RandomSudokuGridLines(A.ImageOnlyTransform):

    def __init__(
        self,
        max_horizontal_lines: int = 2,
        max_vertical_lines: int = 2,
        thickness_range: tuple[int, int] = (1, 3),
        edge_ratio: float = 0.22,
        line_value_range: tuple[int, int] = (0, 55),
        middle_line_probability: float = 0.20,
        p: float = 0.35,
    ):
        super().__init__(p=p)

        self.max_horizontal_lines = max_horizontal_lines
        self.max_vertical_lines = max_vertical_lines
        self.thickness_range = thickness_range
        self.edge_ratio = edge_ratio
        self.line_value_range = line_value_range
        self.middle_line_probability = middle_line_probability

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]

        horizontal_count = self.py_random.randint(
            0,
            self.max_horizontal_lines,
        )
        vertical_count = self.py_random.randint(
            0,
            self.max_vertical_lines,
        )

        if horizontal_count == 0 and vertical_count == 0:
            if self.py_random.random() < 0.5:
                horizontal_count = 1
            else:
                vertical_count = 1

        for _ in range(horizontal_count):
            y = self._random_line_position(
                length=height,
                allow_middle=True,
            )
            thickness = self.py_random.randint(
                self.thickness_range[0],
                self.thickness_range[1],
            )
            line_value = self.py_random.randint(
                self.line_value_range[0],
                self.line_value_range[1],
            )

            cv2.line(
                result,
                (0, y),
                (width - 1, y),
                color=self._color_for_image(result, line_value),
                thickness=thickness,
                lineType=cv2.LINE_8,
            )

        for _ in range(vertical_count):
            x = self._random_line_position(
                length=width,
                allow_middle=True,
            )
            thickness = self.py_random.randint(
                self.thickness_range[0],
                self.thickness_range[1],
            )
            line_value = self.py_random.randint(
                self.line_value_range[0],
                self.line_value_range[1],
            )

            cv2.line(
                result,
                (x, 0),
                (x, height - 1),
                color=self._color_for_image(result, line_value),
                thickness=thickness,
                lineType=cv2.LINE_8,
            )

        return result

    def _random_line_position(
        self,
        length: int,
        allow_middle: bool,
    ) -> int:
        if (
            allow_middle
            and self.py_random.random() < self.middle_line_probability
        ):
            return self.py_random.randint(
                max(0, int(length * 0.25)),
                max(0, int(length * 0.75)),
            )

        edge_size = max(1, int(length * self.edge_ratio))

        if self.py_random.random() < 0.5:
            return self.py_random.randint(0, edge_size)

        return self.py_random.randint(
            max(0, length - edge_size - 1),
            max(0, length - 1),
        )

    @staticmethod
    def _color_for_image(
        image: np.ndarray,
        value: int,
    ):
        if image.ndim == 2:
            return value

        return tuple([value] * image.shape[2])

    def get_transform_init_args_names(self):
        return (
            "max_horizontal_lines",
            "max_vertical_lines",
            "thickness_range",
            "edge_ratio",
            "line_value_range",
            "middle_line_probability",
            )

class DigitTransform:
    def __init__(
        self,
        size: int = 28,
        augment: bool = False,
        augment_config: dict[str, Any] | None = None,
    ):
        self.size = size
        self.augment = augment
        self.augment_config = augment_config or {}

        self.empty_threshold = int(
            self.augment_config.get("empty_threshold", 12)
        )

        pad_extra = int(self.augment_config.get("pad_extra", 8))
        affine_p = float(self.augment_config.get("affine_p", 0.35))
        translate_percent = float(
            self.augment_config.get("translate_percent", 0.03)
        )
        scale_min = float(self.augment_config.get("scale_min", 0.97))
        scale_max = float(self.augment_config.get("scale_max", 1.03))
        rotate_limit = float(self.augment_config.get("rotate_limit", 5))
        shear_limit = float(self.augment_config.get("shear_limit", 2))
        brightness_limit = float(self.augment_config.get("brightness_limit", 0.08))
        contrast_limit = float(self.augment_config.get("contrast_limit", 0.08))
        brightness_contrast_p = float(
            self.augment_config.get("brightness_contrast_p", 0.25)
        )
        blur_p = float(self.augment_config.get("blur_p", 0.04))

        base_transforms = [
            A.Resize(size, size),
        ]

        aug_transforms = [
            A.PadIfNeeded(
                min_height=size + pad_extra,
                min_width=size + pad_extra,
                border_mode=cv2.BORDER_REPLICATE,
                p=1.0,
            ),

            A.Affine(
                translate_percent={
                    "x": (-translate_percent, translate_percent),
                    "y": (-translate_percent, translate_percent),
                },
                scale=(scale_min, scale_max),
                rotate=(-rotate_limit, rotate_limit),
                shear={
                    "x": (-shear_limit, shear_limit),
                    "y": (-shear_limit, shear_limit),
                },
                border_mode=cv2.BORDER_REPLICATE,
                p=affine_p,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=brightness_contrast_p,
            ),

            # نویز گوسی ملایم
            A.GaussNoise(
                std_range=(0.02, 0.08),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.35,
            ),

            # Blur بعد از نویز، برای طبیعی‌تر شدن نویز
            A.GaussianBlur(
                blur_limit=(3, 3),
                p=blur_p,
            ),

            A.Resize(
                height=size,
                width=size,
            ),
            
            RandomSudokuGridLines(
                max_horizontal_lines=2,
                max_vertical_lines=2,
                thickness_range=(1, 1),
                edge_ratio=0.0,
                middle_line_probability=0.50,
                line_value_range=(140, 240),
                p=0.6,
            ),
        ]

        if self.augment:
            self.transform = A.Compose(aug_transforms)
        else:
            self.transform = A.Compose(base_transforms)

        self.safe_resize = A.Compose(base_transforms)

    def __call__(self, image):
        image = self._to_grayscale_uint8(image)

        safe = self.safe_resize(image=image)["image"]

        transformed = self.transform(image=image)
        image = transformed["image"]

        if self.augment and self._is_almost_empty(image):
            image = safe

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        return image

    @staticmethod
    def _to_grayscale_uint8(image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert("L"))
        else:
            image = np.array(image)

            if image.ndim == 3:
                image = image[:, :, 0]

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0

            image = image.astype(np.uint8)

        return image

    def _is_almost_empty(self, image: np.ndarray) -> bool:
        foreground_pixels = np.count_nonzero(image > 30)
        return foreground_pixels < self.empty_threshold


def _normalize_languages(languages: list[str] | str | None) -> set[str]:

    if languages is None:
        return {"fa", "en"}

    if isinstance(languages, str):
        languages = [languages]

    normalized = {lang.strip().lower() for lang in languages}

    valid = {"fa", "en"}
    invalid = normalized - valid
    if invalid:
        raise ValueError(f"Invalid language(s): {invalid}. Valid options: {valid}")

    if not normalized:
        return {"fa", "en"}

    return normalized


def build_digit_dataloaders(languages: list[str] | str | None = None):
    backend_root = BACKEND_ROOT

    phase_cfg = get_phase4_config()

    model_cfg = phase_cfg.get("model", {})
    data_cfg = phase_cfg.get("data", {})
    train_cfg = phase_cfg.get("training", {})
    aug_cfg = phase_cfg.get("augmentation", {})

    if languages is None:
        languages = data_cfg.get("languages")

    active_languages = _normalize_languages(languages)

    image_size = int(model_cfg.get("image_size", 28))
    validation_split = float(data_cfg.get("validation_split", 0.1))
    include_zero_digit = bool(data_cfg.get("include_zero_digit", False))
    seed = int(train_cfg.get("seed", 42))

    batch_size = int(train_cfg.get("batch_size", 64))

    num_workers = int(data_cfg.get("num_workers", 0))

    augmentation_enabled = bool(aug_cfg.get("enabled", True))

    train_transform = DigitTransform(
        size=image_size,
        augment=augmentation_enabled,
        augment_config=aug_cfg,
    )

    eval_transform = DigitTransform(
        size=image_size,
        augment=False,
        augment_config=aug_cfg,
    )

    train_aug_sources = []
    train_eval_sources = []
    test_sources = []

    dataset_sizes: dict[str, int] = {}

       # --- فارسی: Hoda + Generated Persian digits ---
    if "fa" in active_languages:
        hoda_cfg = data_cfg.get("hoda", {})
        hoda_root = backend_root / hoda_cfg.get(
            "raw_dir",
            "phases/phase_04_digit_recognition/src/datasets/hoda",
        )
        hoda_train_cdb = hoda_root / hoda_cfg.get(
            "train_cdb",
            "DigitDB/Train 60000.cdb",
        )
        hoda_test_cdb = hoda_root / hoda_cfg.get(
            "test_cdb",
            "DigitDB/Test 20000.cdb",
        )

        hoda_train_aug = HodaCDBDataset(
            cdb_path=hoda_train_cdb,
            image_size=image_size,
            include_zero_digit=include_zero_digit,
            transform=train_transform,
        )

        hoda_train_eval = HodaCDBDataset(
            cdb_path=hoda_train_cdb,
            image_size=image_size,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )

        hoda_test = HodaCDBDataset(
            cdb_path=hoda_test_cdb,
            image_size=image_size,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )

        train_aug_sources.append(hoda_train_aug)
        train_eval_sources.append(hoda_train_eval)
        test_sources.append(hoda_test)

        dataset_sizes["Hoda train"] = len(hoda_train_aug)
        dataset_sizes["Hoda test"] = len(hoda_test)

        generated_fa_cfg = data_cfg.get("generated_fa", {})

        if bool(generated_fa_cfg.get("enabled", False)):
            generated_fa_root = backend_root / generated_fa_cfg.get(
                "root_dir",
                "generated_digits",
            )

            if not generated_fa_root.is_dir():
                raise FileNotFoundError(
                    f"Generated Persian dataset not found: {generated_fa_root}"
                )

            generated_fa_aug = ImageFolder(
                root=str(generated_fa_root),
                transform=train_transform,
            )

            generated_fa_eval = ImageFolder(
                root=str(generated_fa_root),
                transform=eval_transform,
            )

            if not include_zero_digit:
                zero_label = generated_fa_aug.class_to_idx.get("0")

                generated_indices = [
                    index
                    for index, (_, label) in enumerate(generated_fa_aug.samples)
                    if label != zero_label
                ]

                generated_fa_aug = Subset(
                    generated_fa_aug,
                    generated_indices,
                )

                generated_fa_eval = Subset(
                    generated_fa_eval,
                    generated_indices,
                )

            train_aug_sources.append(generated_fa_aug)
            train_eval_sources.append(generated_fa_eval)

            dataset_sizes["Generated FA train"] = len(
                generated_fa_aug
            )

    # --- انگلیسی: MNIST + Chars74K ---
    if "en" in active_languages:
        mnist_cfg = data_cfg.get("mnist", {})
        chars74k_cfg = data_cfg.get("chars74k", {})

        mnist_root = backend_root / mnist_cfg.get(
            "root_dir",
            "phases/phase_04_digit_recognition/src/datasets/mnist",
        )
        chars74k_root = backend_root / chars74k_cfg.get(
            "root_dir",
            "phases/phase_04_digit_recognition/src/datasets/chars74k/EnglishFnt/Fnt",
        )

        mnist_train_aug = MNISTRawDataset(
            root_dir=mnist_root,
            train=True,
            include_zero_digit=include_zero_digit,
            transform=train_transform,
        )
        mnist_train_eval = MNISTRawDataset(
            root_dir=mnist_root,
            train=True,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )
        mnist_test = MNISTRawDataset(
            root_dir=mnist_root,
            train=False,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )

        chars74k_full_aug = Chars74KFntDataset(
            root_dir=chars74k_root,
            image_size=image_size,
            include_zero_digit=include_zero_digit,
            transform=train_transform,
        )
        chars74k_full_eval = Chars74KFntDataset(
            root_dir=chars74k_root,
            image_size=image_size,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )

        n_c74k = len(chars74k_full_aug)
        c74k_test_split = float(chars74k_cfg.get("test_split", 0.1))
        n_c74k_test = int(n_c74k * c74k_test_split)
        n_c74k_train = n_c74k - n_c74k_test

        c74k_indices = torch.randperm(
            n_c74k,
            generator=torch.Generator().manual_seed(seed),
        ).tolist()

        c74k_train_indices = c74k_indices[:n_c74k_train]
        c74k_test_indices = c74k_indices[n_c74k_train:]

        chars74k_train_aug = Subset(chars74k_full_aug, c74k_train_indices)
        chars74k_train_eval = Subset(chars74k_full_eval, c74k_train_indices)
        chars74k_test = Subset(chars74k_full_eval, c74k_test_indices)

        train_aug_sources.append(mnist_train_aug)
        train_eval_sources.append(mnist_train_eval)
        test_sources.append(mnist_test)
        dataset_sizes["MNIST train"] = len(mnist_train_aug)
        dataset_sizes["MNIST test"] = len(mnist_test)

        train_aug_sources.append(chars74k_train_aug)
        train_eval_sources.append(chars74k_train_eval)
        test_sources.append(chars74k_test)
        dataset_sizes["Chars74K train"] = len(chars74k_train_aug)
        dataset_sizes["Chars74K test"] = len(chars74k_test)

    if not train_aug_sources:
        raise ValueError(f"No datasets selected for languages={active_languages}")

    full_train_aug_dataset = ConcatDataset(train_aug_sources)
    full_train_eval_dataset = ConcatDataset(train_eval_sources)
    test_dataset = ConcatDataset(test_sources)

    n_total = len(full_train_aug_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val

    indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_train_aug_dataset, train_indices)
    val_dataset = Subset(full_train_eval_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Active languages: {sorted(active_languages)}")
    print("Datasets loaded:")
    for name, size in dataset_sizes.items():
        print(f"{name}: {size}")
    print(f"Train total:     {len(train_dataset)}")
    print(f"Val total:       {len(val_dataset)}")
    print(f"Test total:      {len(test_dataset)}")

    print("\nDataLoader config:")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print(f"Include zero digit: {include_zero_digit}")
    print(f"Seed: {seed}")
    print(f"Num workers: {num_workers}")
    print(f"Augmentation enabled: {augmentation_enabled}")

    return train_loader, val_loader, test_loader


def build_eval_only_datasets(languages: list[str] | str | None = None) -> dict:

    backend_root = BACKEND_ROOT

    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    data_cfg = phase_cfg.get("data", {})
    train_cfg = phase_cfg.get("training", {})

    if languages is None:
        languages = data_cfg.get("languages")

    active_languages = _normalize_languages(languages)

    image_size = int(model_cfg.get("image_size", 28))
    include_zero_digit = bool(data_cfg.get("include_zero_digit", False))
    seed = int(train_cfg.get("seed", 42))

    aug_cfg = phase_cfg.get("augmentation", {})
    eval_transform = DigitTransform(size=image_size, augment=False, augment_config=aug_cfg)

    result: dict = {}

    if "fa" in active_languages:
        hoda_cfg = data_cfg.get("hoda", {})
        hoda_root = backend_root / hoda_cfg.get("raw_dir", "phases/phase_04_digit_recognition/src/datasets/hoda")
        hoda_test_cdb = hoda_root / hoda_cfg.get("test_cdb", "DigitDB/Test 20000.cdb")

        result["hoda"] = HodaCDBDataset(
            cdb_path=hoda_test_cdb,
            image_size=image_size,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )

    if "en" in active_languages:
        mnist_cfg = data_cfg.get("mnist", {})
        chars74k_cfg = data_cfg.get("chars74k", {})

        mnist_root = backend_root / mnist_cfg.get("root_dir", "phases/phase_04_digit_recognition/src/datasets/mnist")
        chars74k_root = backend_root / chars74k_cfg.get(
            "root_dir", "phases/phase_04_digit_recognition/src/datasets/chars74k/EnglishFnt/Fnt"
        )

        result["mnist"] = MNISTRawDataset(
            root_dir=mnist_root,
            train=False,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )

        chars74k_full_eval = Chars74KFntDataset(
            root_dir=chars74k_root,
            image_size=image_size,
            include_zero_digit=include_zero_digit,
            transform=eval_transform,
        )

        n_c74k = len(chars74k_full_eval)
        c74k_test_split = float(chars74k_cfg.get("test_split", 0.1))
        n_c74k_test = int(n_c74k * c74k_test_split)

        c74k_indices = torch.randperm(n_c74k, generator=torch.Generator().manual_seed(seed)).tolist()
        c74k_test_indices = c74k_indices[n_c74k - n_c74k_test:]

        result["chars74k"] = Subset(chars74k_full_eval, c74k_test_indices)

    return result