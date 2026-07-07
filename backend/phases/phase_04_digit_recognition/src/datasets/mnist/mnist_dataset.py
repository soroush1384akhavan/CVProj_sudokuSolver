from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MNISTRawDataset(Dataset):

    def __init__(
        self,
        root_dir: str | Path,
        train: bool = True,
        include_zero_digit: bool = False,
        transform: Optional[object] = None,
    ):
        self.root_dir = Path(root_dir)
        self.train = train
        self.include_zero_digit = include_zero_digit
        self.transform = transform

        self.raw_dir = self.root_dir / "MNIST" / "raw"

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"MNIST raw directory not found: {self.raw_dir}")

        if train:
            images_file = self._find_file([
                "train-images-idx3-ubyte",
                "train-images.idx3-ubyte",
                "train-images-idx3-ubyte.gz",
                "train-images.idx3-ubyte.gz",
            ])
            labels_file = self._find_file([
                "train-labels-idx1-ubyte",
                "train-labels.idx1-ubyte",
                "train-labels-idx1-ubyte.gz",
                "train-labels.idx1-ubyte.gz",
            ])
        else:
            images_file = self._find_file([
                "t10k-images-idx3-ubyte",
                "t10k-images.idx3-ubyte",
                "t10k-images-idx3-ubyte.gz",
                "t10k-images.idx3-ubyte.gz",
            ])
            labels_file = self._find_file([
                "t10k-labels-idx1-ubyte",
                "t10k-labels.idx1-ubyte",
                "t10k-labels-idx1-ubyte.gz",
                "t10k-labels.idx1-ubyte.gz",
            ])

        self.images = self._read_images(images_file)
        self.labels = self._read_labels(labels_file)

        if len(self.images) != len(self.labels):
            raise ValueError("تعداد تصاویر و لیبل‌های MNIST برابر نیست.")

        self.indices = []

        for index, label in enumerate(self.labels):
            label = int(label)

            if not self.include_zero_digit and label == 0:
                continue

            self.indices.append(index)

    def _find_file(self, possible_names: list[str]) -> Path:
        for name in possible_names:
            path = self.raw_dir / name
            if path.exists():
                return path

        raise FileNotFoundError(
            f"هیچ‌کدام از این فایل‌ها پیدا نشدند: {possible_names}\n"
            f"مسیر بررسی‌شده: {self.raw_dir}"
        )

    def _open_file(self, path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rb")

        return open(path, "rb")

    def _read_images(self, path: Path) -> np.ndarray:
        with self._open_file(path) as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

            if magic != 2051:
                raise ValueError(f"فرمت فایل تصویر MNIST اشتباه است: {path}")

            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num_images, rows, cols)

        return images

    def _read_labels(self, path: Path) -> np.ndarray:
        with self._open_file(path) as f:
            magic, num_labels = struct.unpack(">II", f.read(8))

            if magic != 2049:
                raise ValueError(f"فرمت فایل لیبل MNIST اشتباه است: {path}")

            labels = np.frombuffer(f.read(), dtype=np.uint8)

        if len(labels) != num_labels:
            raise ValueError(f"تعداد لیبل‌های خوانده‌شده درست نیست: {path}")

        return labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_index = self.indices[index]

        image = self.images[real_index]
        label = int(self.labels[real_index])

        image = Image.fromarray(image).convert("L")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).float()
            image = image.unsqueeze(0) / 255.0

        return image, label