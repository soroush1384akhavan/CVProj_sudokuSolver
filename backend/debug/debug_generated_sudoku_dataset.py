## for run:
## python -m debug.debug_generated_sudoku_dataset
## python -m debug.debug_generated_sudoku_dataset --root storage/sudoku/synthetic/sudoku_medium_..._seed_.../images_variant

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from phases.phase_04_digit_recognition.src.datasets.fromsoduko.generated_dataloader import (
    GeneratedSudokuCellDataset,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to a generated-dataset run folder (containing labels.csv and images).",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include empty cells (label 0) in the dataset.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-extraction of cells, ignoring any existing cache.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of random samples to display.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sample selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading dataset from: {args.root}")
    dataset = GeneratedSudokuCellDataset(
        root_dir=args.root,
        include_empty_cells=args.include_empty,
        refresh_cache=args.refresh_cache,
        transform=None,
    )

    print(f"Total samples: {len(dataset)}")

    # توزیع لیبل‌ها رو چاپ کن - برای چک تعادل کلاس‌ها
    from collections import Counter
    labels = [label for _, label in dataset.samples]
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    for label in sorted(label_counts):
        print(f"  {label}: {label_counts[label]}")

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), k=num_samples)

    cols = 5
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(2.5 * cols, 2.5 * rows))

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        if torch.is_tensor(image):
            image = image.detach().cpu()
            if image.ndim == 3:
                image = image.squeeze(0)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"label={label}", fontsize=10)
        plt.axis("off")

    plt.suptitle(f"Random samples — {args.root}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()