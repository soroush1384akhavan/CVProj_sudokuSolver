## for run : python -m debug.debug_mnist_labels

import random
import numpy as np
import matplotlib.pyplot as plt

from phases.phase_04_digit_recognition.src.datasets.mnist.mnist_dataset import MNISTRawDataset

dataset = MNISTRawDataset(
    root_dir="phases/phase_04_digit_recognition/src/datasets/mnist",
    train=False,
    include_zero_digit=False,
    transform=None,
)

indices = random.sample(range(len(dataset)), k=20)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(indices):
    image, label = dataset[idx]

    # اگه numpy array یا tensor باشه و بعد کانال (1, H, W) داشته باشه، حذفش کن
    image = np.array(image)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    plt.subplot(4, 5, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"label={label}")
    plt.axis("off")
plt.tight_layout()
plt.show()