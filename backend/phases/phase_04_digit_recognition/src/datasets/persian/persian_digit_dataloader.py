from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PersianDigitDataset(Dataset):
    def __init__(self, root_dir: str | Path, image_size: int = 28):
        self.root_dir = Path(root_dir)

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.dataset = ImageFolder(
            root=str(self.root_dir),
            transform=self.transform,
        )

        expected_classes = [str(index) for index in range(10)]

        if self.dataset.classes != expected_classes:
            raise ValueError(
                f"Expected class folders {expected_classes}, "
                f"but found {self.dataset.classes}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        return image, label


def build_dataloaders(
    root_dir: str | Path,
    batch_size: int = 64,
    image_size: int = 28,
    validation_ratio: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    dataset = PersianDigitDataset(
        root_dir=root_dir,
        image_size=image_size,
    )

    validation_size = max(1, int(len(dataset) * validation_ratio))
    train_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Test the Persian digit DataLoader.")
    parser.add_argument("root_dir", type=str)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_loader, validation_loader = build_dataloaders(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        validation_ratio=args.validation_ratio,
        num_workers=args.num_workers,
    )

    images, labels = next(iter(train_loader))

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Pixel range: {images.min().item():.3f} to {images.max().item():.3f}")
    print(f"Labels: {labels[:16].tolist()}")