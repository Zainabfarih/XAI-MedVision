"""PyTorch dataset for the LIDC lung-CT classification task."""
import os
import random

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

# Project root used to resolve the relative paths stored in the CSV splits.
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())


def resolve_path(path):
    """Resolve a CSV path: absolute paths are kept, relative ones are
    anchored to PROJECT_ROOT so the splits work from any machine."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


class LIDCDataset(Dataset):
    def __init__(self, df, transform=None, use_albumentations=False, pretrain=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.pretrain = pretrain
        if not pretrain:
            self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(resolve_path(row["path"])).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            # Skip a corrupted/missing file by drawing another sample.
            print(f"Skipping unreadable image: {row['path']}")
            return self.__getitem__(random.randint(0, len(self.df) - 1))

        if self.transform is not None:
            if self.use_albumentations:
                image = self.transform(image=np.array(image))["image"]
            else:
                image = self.transform(image)

        if self.pretrain:
            return image
        return image, self.labels[idx]

    def get_class_weights(self):
        cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                                  y=self.labels.numpy())
        return torch.tensor(cw, dtype=torch.float32)
