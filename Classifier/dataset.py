import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def mask2rle(img):
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, shape=(256, 1600)):
    if pd.isnull(rle):
        return np.zeros(shape, dtype=np.uint8)
    rle = list(map(int, rle.strip().split()))
    starts, lengths = rle[0::2], rle[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        mask[s:e] = 1
    return mask.reshape(shape)

class SteelDataset(Dataset):
    def __init__(self, df, image_dir, training_catagory, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.training_catagory = training_catagory
        self.image_names = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image, catagory = self.__getcatagorybyname__(name)
        return image, catagory[self.training_catagory - 1]

    def __getmaskbyname__(self, name):
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        mask = np.zeros((4, 256, 1600), dtype=np.float32)

        for cls in range(4):
            rle = self.df.loc[
                (self.df['ImageId'] == name) & (self.df['ClassId'] == cls + 1),  # 注意 ClassId 是字符串  真的吗？
                'EncodedPixels'
            ].values
            mask[cls] = rle2mask(rle[0] if len(rle) else None)
        return image, mask

    def __getcatagorybyname__(self, name):
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        catagory = np.zeros(4, dtype=np.float32)

        for i in range(4):
            matching_defect = self.df.loc[
                (self.df['ImageId'] == name) &
                (self.df['ClassId'] == i + 1),
                'EncodedPixels'
            ]
            if not matching_defect.empty and not pd.isnull(matching_defect.iloc[0]):
                catagory[i] = 1.0
        return image, catagory
