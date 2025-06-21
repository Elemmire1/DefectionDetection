import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import SteelDataset
from resnet import ResNetBinary
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)
train_losses = [[] for _ in range(5)]  # 用于存储每个类别的训练损失
val_losses = [[] for _ in range(5)]  # 用于存储每个类别的验证损失

for i in range(4):
    # === 参数设置 ===
    training_catagory = i+1  # 第几类缺陷
    batch_size = 8
    num_epochs = 10
    lr = 1e-4

    # === 数据增强 transform，仅使用 torchvision ===
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),

    ])

    # === 读取数据并构建数据集 ===
    df = pd.read_csv("../data/train.csv")

    dataset = SteelDataset(df, '../data/train_images', training_catagory, transform=train_transform)

    # 训练验证集划分
    train_len = int(0.7 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    # 替换验证集的 transform
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # === 模型设置 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetBinary().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # === 训练过程 ===
    best_val_loss = float('inf')  # 初始化最小验证损失

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
        train_losses[training_catagory].append(avg_train_loss)
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
        val_losses[training_catagory].append(avg_val_loss)
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

        # 调整学习率
        scheduler.step(avg_val_loss)

        # === 保存最佳模型 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"models/class_{training_catagory}.pth")
            print(f"✅ Saved best model at epoch {epoch+1} with val_loss {avg_val_loss:.4f}")

    # 最终提示
    print("训练完成，已保存 val_loss 最佳的模型。")

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(12, 5))
for cls in range(1, 5):
    plt.plot(train_losses[cls], label=f'Class {cls} Train Loss')
    plt.plot(val_losses[cls], label=f'Class {cls} Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig(f"images/loss.png")
