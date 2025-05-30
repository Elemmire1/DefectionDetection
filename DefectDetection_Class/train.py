import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from unet_model import UNet
from dataset import SteelDataset
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import os

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)  # 如果输出没有激活函数
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 数据加载
df = pd.read_csv("./data/train.csv")
transform = T.Compose([T.ToTensor()])
dataset = SteelDataset(df, "./data/train_images", transform)

# 划分训练/验证集
train_len = int(0.7 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# 模型、优化器、损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = DiceLoss()

# 训练 + 验证
for epoch in range(10):
    model.train()
    total_train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/10 - Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # 验证阶段
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

# 保存模型
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "checkpoints/checkpoint.pth")
print("✅ 模型已保存为 model.pth 和 checkpoints/checkpoint.pth")
