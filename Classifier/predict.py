import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from resnet import load_model
from dataset import mask2rle

model = load_model()

test_dir = "../data/test_images"
test_imgs = sorted(os.listdir(test_dir))

transform = transforms.Compose([
    transforms.ToTensor()
])

# 打开 CSV 文件，准备写入
with open("prediction_task.csv", mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ImageId", "1", "2", "3", "4"])

    # 预测并写入每张图像的结果
    with torch.no_grad():
        for name in tqdm(test_imgs, desc="Predicting"):
            img_path = os.path.join(test_dir, name)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).cuda()

            pred = [None] * 4
            for i in range(4):
                pred[i] = model[i](img).cuda().item()

            writer.writerow([name, pred[0], pred[1], pred[2], pred[3]])
