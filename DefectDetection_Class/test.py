import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from dataset import SteelDataset
from loadmodel import load_model_resnet

# 加载模型：第4类缺陷，索引为3
model = load_model_resnet()
model4 = model[3].eval().cuda()

# 设置图像路径和转换
test_dir = "./data/train_images"
test_imgs = sorted(os.listdir(test_dir))
print(len(test_imgs))
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据集获取标签
df = pd.read_csv("./data/train.csv")
dataset = SteelDataset(
    df=df,
    image_dir='data/train_images',
    training_catagory=4,  
    transform=transform
)

# 初始化计数器
TP = TN = FP = FN = 0

with torch.no_grad():
    for name in tqdm(test_imgs, desc="Testing Class 4"):
        # 读取并预处理图像
        img_path = os.path.join(test_dir, name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).cuda()

        # 模型预测
        pred = model4(img_tensor).item()
        pred_label = 1 if pred > 0.5 else 0

        # 真实标签
        gt_vec = dataset.__getcatagorybyname__(name)[1]
        gt_label = int(gt_vec[3].item() > 0.5)

        # 分类评估
        if gt_label == 1 and pred_label == 1:
            TP += 1
        elif gt_label == 0 and pred_label == 0:
            TN += 1
        elif gt_label == 0 and pred_label == 1:
            FP += 1
        elif gt_label == 1 and pred_label == 0:
            FN += 1

# 计算指标
total = TP + TN + FP + FN
acc = (TP + TN) / total if total > 0 else 0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

# 输出结果
print(f"Class 4 Accuracy       : {acc:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f} (原本没缺陷却预测成有)")
print(f"False Negative Rate (FNR): {fnr:.4f} (原本有缺陷却预测成无)")