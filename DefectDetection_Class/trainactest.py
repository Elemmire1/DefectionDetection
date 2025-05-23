import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from cnn import CNN
from dataset import mask2rle
from loadmodel import load_model_cnn, load_model_mix, load_model_resnet
import pandas as pd
import torchvision.transforms as T
from dataset import SteelDataset

model = load_model_mix()

test_dir = "./data/train_images"
test_imgs = sorted(os.listdir(test_dir))

transform = transforms.Compose([
    transforms.ToTensor()
])

df = pd.read_csv("./data/train.csv")
transform2 = T.Compose([T.ToTensor()])
dataset = SteelDataset(
    df=df,
    image_dir='data/train_images',
    training_catagory=1,
    transform=transform2
)

sum = 0
acc = [0] * 4

with torch.no_grad():
    for name in tqdm(test_imgs, desc="Testing"):
        img_path = os.path.join(test_dir, name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).cuda()
            
        sum += 1

        ans = dataset.__getcatagorybyname__(name)[1]
        for i in range(4):
            pred = (model[i](img).cuda().item() > 0.5)
            if pred == (ans[i].item() > 0.5):
                acc[i] += 1

print("acc: ", acc[0]/sum, ",", acc[1]/sum, ",", acc[2]/sum, ",", acc[3]/sum)
        


