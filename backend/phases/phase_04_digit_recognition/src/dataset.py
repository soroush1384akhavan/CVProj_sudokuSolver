from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

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
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
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
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=affine_p,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=brightness_contrast_p,
            ),
            A.GaussianBlur(
                blur_limit=(3, 3),
                p=blur_p,
            ),
            A.Resize(size, size),
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


def build_digit_dataloaders():
    backend_root = BACKEND_ROOT

    phase_cfg = get_phase4_config()

    model_cfg = phase_cfg.get("model", {})
    data_cfg = phase_cfg.get("data", {})
    train_cfg = phase_cfg.get("training", {})
    aug_cfg = phase_cfg.get("augmentation", {})

    image_size = int(model_cfg.get("image_size", 28))
    validation_split = float(data_cfg.get("validation_split", 0.1))
    include_zero_digit = bool(data_cfg.get("include_zero_digit", False))
    seed = int(train_cfg.get("seed", 42))

    batch_size = int(train_cfg.get("batch_size", 64))

    num_workers = int(data_cfg.get("num_workers", 0))

    augmentation_enabled = bool(aug_cfg.get("enabled", True))

    hoda_cfg = data_cfg.get("hoda", {})
    mnist_cfg = data_cfg.get("mnist", {})
    chars74k_cfg = data_cfg.get("chars74k", {})

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

    mnist_root = backend_root / mnist_cfg.get(
        "root_dir",
        "phases/phase_04_digit_recognition/src/datasets/mnist",
    )

    chars74k_root = backend_root / chars74k_cfg.get(
        "root_dir",
        "phases/phase_04_digit_recognition/src/datasets/chars74k/EnglishFnt/Fnt",
    )

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

    # Chars74K هیچ split رسمی train/test نداره (فقط یک پوشه‌ی تخت پر از عکس)،
    # پس خودمون یک split تصادفی و ثابت (با seed) روش می‌زنیم.
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

    full_train_aug_dataset = ConcatDataset([
        hoda_train_aug,
        mnist_train_aug,
        chars74k_train_aug,
    ])

    full_train_eval_dataset = ConcatDataset([
        hoda_train_eval,
        mnist_train_eval,
        chars74k_train_eval,
    ])

    test_dataset = ConcatDataset([
        hoda_test,
        mnist_test,
        chars74k_test,
    ])

    n_total = len(full_train_aug_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val

    indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(
        full_train_aug_dataset,
        train_indices,
    )

    val_dataset = Subset(
        full_train_eval_dataset,
        val_indices,
    )

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

    print("Datasets loaded:")
    print(f"Hoda train:      {len(hoda_train_aug)}")
    print(f"MNIST train:     {len(mnist_train_aug)}")
    print(f"Chars74K train:  {len(chars74k_train_aug)}")
    print(f"Train total:     {len(train_dataset)}")
    print(f"Val total:       {len(val_dataset)}")
    print(f"Hoda test:       {len(hoda_test)}")
    print(f"MNIST test:      {len(mnist_test)}")
    print(f"Chars74K test:   {len(chars74k_test)}")
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

def build_eval_only_datasets() -> dict:
    """
    برای هر ساب‌دیتاست (Hoda, MNIST, Chars74K) یک نسخه‌ی eval-only (بدون augmentation)
    از test set جداگانه برمی‌گردونه، تا بشه دقت مدل رو روی هرکدوم جداگانه سنجید.
    """
    backend_root = BACKEND_ROOT

    phase_cfg = get_phase4_config()
    model_cfg = phase_cfg.get("model", {})
    data_cfg = phase_cfg.get("data", {})
    train_cfg = phase_cfg.get("training", {})

    image_size = int(model_cfg.get("image_size", 28))
    include_zero_digit = bool(data_cfg.get("include_zero_digit", False))
    seed = int(train_cfg.get("seed", 42))

    aug_cfg = phase_cfg.get("augmentation", {})
    eval_transform = DigitTransform(size=image_size, augment=False, augment_config=aug_cfg)

    hoda_cfg = data_cfg.get("hoda", {})
    mnist_cfg = data_cfg.get("mnist", {})
    chars74k_cfg = data_cfg.get("chars74k", {})

    hoda_root = backend_root / hoda_cfg.get("raw_dir", "phases/phase_04_digit_recognition/src/datasets/hoda")
    hoda_test_cdb = hoda_root / hoda_cfg.get("test_cdb", "DigitDB/Test 20000.cdb")

    mnist_root = backend_root / mnist_cfg.get("root_dir", "phases/phase_04_digit_recognition/src/datasets/mnist")

    chars74k_root = backend_root / chars74k_cfg.get(
        "root_dir", "phases/phase_04_digit_recognition/src/datasets/chars74k/EnglishFnt/Fnt"
    )

    hoda_test = HodaCDBDataset(
        cdb_path=hoda_test_cdb,
        image_size=image_size,
        include_zero_digit=include_zero_digit,
        transform=eval_transform,
    )

    mnist_test = MNISTRawDataset(
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
    chars74k_test = Subset(chars74k_full_eval, c74k_test_indices)

    return {
        "hoda": hoda_test,
        "mnist": mnist_test,
        "chars74k": chars74k_test,
    }
