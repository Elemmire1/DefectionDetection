import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from dataset import SteelDataset
from resnet import ResNetBinary
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
import os

# === 设置参数 ===
training_catagory = 3  
batch_size = 8
num_epochs = 10
lr = 1e-4

# === 数据增强和原始 transform ===
base_transform = T.Compose([
    T.Resize((224, 1400)),
    T.ToTensor()
])

aug_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomResizedCrop(size=(224, ), scale=(0.8, 1.0)),
    T.ToTensor()
])

# === 读取数据 ===
df = pd.read_csv("./data/train.csv")
dataset_plain = SteelDataset(df, 'data/train_images', training_catagory, transform=base_transform)
dataset_aug = SteelDataset(df, 'data/train_images', training_catagory, transform=aug_transform)

# 合并：原图 + 增强图
full_dataset = ConcatDataset([dataset_plain, dataset_aug])

# 划分训练集和验证集
train_len = int(0.7 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_set, val_set = random_split(full_dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# === 模型与优化器 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetBinary().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# === 训练循环 ===
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        imgs, labels = imgs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(imgs).view(-1)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # === 验证阶段 ===
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device).float()
            outputs = model(imgs).view(-1)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

    # 更新 scheduler
    scheduler.step(avg_val_loss)

# === 保存模型 ===
os.makedirs("checkpointsresnet", exist_ok=True)
torch.save(model.state_dict(), f"modelresnet{training_catagory}.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f"checkpointsresnet/checkpointresnet{training_catagory}.pth")
print("模型已保存")