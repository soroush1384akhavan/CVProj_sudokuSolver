# phases/phase_04_digit_recognition/src/datasets/chars74k/chars74k_dataset.py

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

DIGIT_SAMPLE_FOLDERS = {i: f"Sample{i + 1:03d}" for i in range(10)}


class Chars74KFntDataset(Dataset):
    """
    فقط زیرمجموعه‌ی ارقام (Sample001-Sample010) از Chars74K English/Fnt رو می‌خونه.
    این نسخه شامل کاراکترهای چاپی رندرشده با فونت‌های کامپیوتری واقعیه (نه دست‌نویس).

    نکته‌ی مهم: تصاویر خام Chars74K معمولاً متن تیره روی زمینه‌ی روشن‌اند
    (برخلاف MNIST/Hoda که رقم روشن روی زمینه‌ی تاریکه). برای هماهنگی پولاریتی
    با بقیه‌ی دیتاست‌ها (و با خروجی واقعی clean_cell که THRESH_BINARY_INV می‌زنه)،
    این کلاس همیشه تصویر رو invert می‌کنه.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 28,
        include_zero_digit: bool = False,
        transform=None,
        invert: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.include_zero_digit = include_zero_digit
        self.transform = transform
        self.invert = invert

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Chars74K Fnt folder not found: {self.root_dir}")

        digits = list(range(0, 10)) if include_zero_digit else list(range(1, 10))

        self.samples: list[tuple[Path, int]] = []

        for digit in digits:
            folder_name = DIGIT_SAMPLE_FOLDERS[digit]
            folder_path = self.root_dir / folder_name

            if not folder_path.is_dir():
                raise FileNotFoundError(
                    f"Expected digit folder not found: {folder_path} "
                    f"(digit={digit})"
                )

            image_paths = sorted(
                p for p in folder_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not image_paths:
                raise ValueError(f"No images found for digit {digit} in: {folder_path}")

            label = digit

            for image_path in image_paths:
                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No samples collected from Chars74K root: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("L")
        image = np.array(image)

        if self.invert:
            image = 255 - image

        if self.transform is not None:
            image = self.transform(image)

        return image, label