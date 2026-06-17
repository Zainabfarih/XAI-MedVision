
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class LIDCDataset(Dataset):
    def __init__(self, df, transform=None, use_albumentations=False, pretrain=False):
        self.df                 = df.reset_index(drop=True)
        self.transform          = transform
        self.use_albumentations = use_albumentations
        self.pretrain           = pretrain
        if not pretrain:
            self.labels = torch.tensor(df['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import random
        from PIL import UnidentifiedImageError
        import numpy as np

        row = self.df.iloc[idx]
        try:
            image = Image.open(row['path']).convert('RGB')
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            print(f"Skipping corrupted image: {row['path']}")
            # If the image is broken, randomly pick another index to keep the batch size constant
            random_idx = random.randint(0, len(self.df) - 1)
            return self.__getitem__(random_idx)

        if self.transform is not None:
            if self.use_albumentations:
                aug   = self.transform(image=np.array(image))
                image = aug['image']
            else:
                image = self.transform(image)

        if self.pretrain:
            return image
        return image, self.labels[idx]

    def get_class_weights(self):
        cw = compute_class_weight('balanced', classes=np.array([0,1]),
                               y=self.labels.numpy())
        return torch.tensor(cw, dtype=torch.float32)
