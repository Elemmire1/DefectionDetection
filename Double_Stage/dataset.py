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
    def __init__(self, image_dir, mask_file, cls, transform=None, test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = []
        self.df = pd.read_csv(mask_file)
        self.cls = cls
        # self.image_names = [name for name in os.listdir(image_dir) if name.endswith('.jpg')]
        for name in os.listdir(image_dir):
            if name.endswith('.jpg'):
                rle = self.df.loc[
                    (self.df['ImageId'] == name) & (self.df['ClassId'] == cls),
                    'EncodedPixels'
                ].values
                if len(rle):
                    self.image_names.append(name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        image = np.array(image)
        mask = np.zeros((256, 1600), dtype=np.float32)
        rle = self.df.loc[
            (self.df['ImageId'] == name) & (self.df['ClassId'] == self.cls),
            'EncodedPixels'
        ].values

        mask = rle2mask(rle[0] if len(rle) else None)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()

        return image, mask
