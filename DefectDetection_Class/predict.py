import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from unet_model import UNet

def mask2rle(img):
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 加载模型
model = UNet(4).cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 测试图像路径
test_dir = "./data/test_images"
test_imgs = sorted(os.listdir(test_dir))  # 排序确保一致性

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 打开 CSV 文件，准备写入
with open("submission.csv", mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ImageId", "EncodedPixels", "ClassId"]) 

    # 预测并写入每张图像的结果
    with torch.no_grad():
        for name in tqdm(test_imgs, desc="Predicting"):
            img_path = os.path.join(test_dir, name)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).cuda()
            
            # 得到预测 [4, H, W]
            pred = model(img)[0].cpu().numpy()

            # 遍历每个类别
            for cls in range(4):
                mask = (pred[cls] > 0.5).astype(np.uint8)
                rle = mask2rle(mask)
                writer.writerow([name, rle, cls+1])
