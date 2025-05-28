import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def rle2mask(rle, shape=(1600, 256)):
    if pd.isnull(rle):
        return np.zeros(shape, dtype=np.uint8).T
    rle = list(map(int, rle.strip().split()))
    starts, lengths = rle[0::2], rle[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        mask[s:e] = 1
    return mask.reshape(shape).T

class SteelDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        mask = np.zeros((4, 256, 1600), dtype=np.float32)

        for cls in range(4):
            rle = self.df.loc[
                (self.df['ImageId'] == name) & (self.df['ClassId'] == cls + 1),
                'EncodedPixels'
            ].values
            mask[cls] = rle2mask(rle[0] if len(rle) else None)
        return image, mask
