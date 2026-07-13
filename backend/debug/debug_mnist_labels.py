## for run : python -m debug.debug_mnist_labels

import random
import numpy as np
import matplotlib.pyplot as plt

from phases.phase_04_digit_recognition.src.datasets.mnist.mnist_dataset import MNISTRawDataset
from phases.phase_04_digit_recognition.src.datasets.chars74k.chars74k_dataset import Chars74KFntDataset

mnist_dataset = MNISTRawDataset(
    root_dir="phases/phase_04_digit_recognition/src/datasets/mnist",
    train=False,
    include_zero_digit=False,
    transform=None,
)

chars74k_dataset = Chars74KFntDataset(
    root_dir="phases/phase_04_digit_recognition/src/datasets/chars74k/EnglishFnt/Fnt",
    include_zero_digit=False,
    transform=None,
)


def to_displayable(image):
    image = np.array(image)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    return image


mnist_indices = random.sample(range(len(mnist_dataset)), k=10)
chars74k_indices = random.sample(range(len(chars74k_dataset)), k=10)

plt.figure(figsize=(15, 8))

for i, idx in enumerate(mnist_indices):
    image, label = mnist_dataset[idx]
    image = to_displayable(image)

    plt.subplot(4, 5, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"MNIST label={label}")
    plt.axis("off")

for i, idx in enumerate(chars74k_indices):
    image, label = chars74k_dataset[idx]
    image = to_displayable(image)

    plt.subplot(4, 5, 10 + i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Chars74K label={label}")
    plt.axis("off")

plt.tight_layout()
plt.show()