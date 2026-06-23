"""Build train/val/test DataLoaders from the processed split CSVs."""
import os
import json

import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import LIDCDataset


def get_loaders(data_proc_dir, batch_size=32, num_workers=2):
    with open(os.path.join(data_proc_dir, "config.json")) as f:
        cfg = json.load(f)

    mean, std, img_size = cfg["mean"], cfg["std"], cfg["img_size"]
    class_weights = torch.tensor(cfg["class_weights"], dtype=torch.float32)

    train_df = pd.read_csv(os.path.join(data_proc_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_proc_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_proc_dir, "test.csv"))

    # Vertical flip is intentionally omitted to keep anatomical orientation.
    train_tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    eval_tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    train_loader = DataLoader(LIDCDataset(train_df, train_tfm), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(LIDCDataset(val_df, eval_tfm), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(LIDCDataset(test_df, eval_tfm), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights, cfg
