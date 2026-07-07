from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class HodaCDBDataset(Dataset):
    def __init__(
        self,
        cdb_path,
        image_size=28,
        include_zero_digit=False,
        transform=None,
    ):
        self.cdb_path = Path(cdb_path)
        self.image_size = image_size
        self.include_zero_digit = include_zero_digit
        self.transform = transform

        if not self.cdb_path.exists():
            raise FileNotFoundError(f"CDB file not found: {self.cdb_path}")

        hoda_dir = self.cdb_path.parents[1]
        sys.path.insert(0, str(hoda_dir))

        import HodaDatasetReader

        self.images, self.labels = HodaDatasetReader.read_hoda_dataset(
            dataset_path=str(self.cdb_path),
            images_height=self.image_size,  
            images_width=self.image_size,
            one_hot=False,
            reshape=False,
        )

        self.samples = []

        for image, label in zip(self.images, self.labels):
            label = int(label)

            if not self.include_zero_digit and label == 0:
                continue

            self.samples.append((image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.samples[index]

        image = np.array(image).astype(np.uint8)

        if image.ndim == 3:
            image = image[:, :, 0]

        image = Image.fromarray(image).convert("L")
        image = image.resize((self.image_size, self.image_size))

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).float()
            image = image.unsqueeze(0) / 255.0

        return image, label