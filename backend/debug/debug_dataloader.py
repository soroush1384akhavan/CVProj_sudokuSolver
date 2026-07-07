## for run : python -m debug.debug_dataloader 

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import torch

from phases.phase_04_digit_recognition.src.dataset import build_digit_dataloaders


def debug_show_samples(
    loader,
    num_samples: int = 20,
    seed: int | None = None,
    title: str = "Random samples",
):
    dataset = loader.dataset

    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(dataset))

    random_indices = random.sample(
        range(len(dataset)),
        k=num_samples,
    )

    images = []
    labels = []

    for index in random_indices:
        image, label = dataset[index]

        if torch.is_tensor(label):
            label = label.item()

        images.append(image)
        labels.append(int(label))

    print(f"\n{title}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Selected indices: {random_indices}")
    print(f"Labels: {labels}")
    print(f"Min label: {min(labels)}")
    print(f"Max label: {max(labels)}")

    cols = 5
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(10, 2 * rows))

    for i, (image, label) in enumerate(zip(images, labels)):
        if torch.is_tensor(image):
            image = image.detach().cpu()

            if image.ndim == 3:
                image = image.squeeze(0)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    train_loader, val_loader, test_loader = build_digit_dataloaders()

    debug_show_samples(
        train_loader,
        num_samples=20,
        seed=None,
        title="Train loader random samples with augmentation",
    )

    debug_show_samples(
        val_loader,
        num_samples=20,
        seed=None,
        title="Validation loader random samples without augmentation",
    )


if __name__ == "__main__":
    main()