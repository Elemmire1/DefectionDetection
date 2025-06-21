import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import get_model
from dataset import SteelDataset
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
import os
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

alpha = 1.0
Parameters = f"loss_{alpha}_dataenhancement_no"
Print = True
Visualize = False

print(f"Parameters: {Parameters}")

transform = A.Compose([
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

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

def compute_dice(pred, target, eps=1e-6):
    pred = (pred > 0.5).bool()
    target = target.bool()
    intersection = (pred & target).float().sum()
    return (2. * intersection + eps) / (pred.float().sum() + target.float().sum() + eps)

dice_scores = [[] for _ in range(5)]  # 用于存储每个类别的 Dice 分数
train_losses = [[] for _ in range(5)]  # 用于存储每个类别的训练损失
val_losses = [[] for _ in range(5)]  # 用于存储每个类别的验证损失

for cls in range(1, 5):
    dataset = SteelDataset("../data/train_images",
                           "../data/train.csv", cls, transform=transform)

    # 划分训练/验证集
    train_len = int(0.7 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 选择模型类型和编码器
    model_type = "unet"
    encoder_name = "efficientnet-b3"
    classes = 1  # 一种钢铁缺陷类别（是/不是）

    # 创建模型
    model = get_model(model_type, encoder_name, classes).to(device)
    print(f"创建模型: {model_type} 使用编码器 {encoder_name}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 使用 SMP 的损失函数
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    # focal_loss = smp.losses.FocalLoss(mode='binary', gamma=2.0, alpha=0.75)
    loss_fn = lambda pred, target: alpha * bce_loss(pred, target) + (1 - alpha) * dice_loss(pred, target)
    # loss_fn = lambda pred, target: dice_loss(pred, target) + focal_loss(pred, target)

    # 训练参数
    num_epochs = 50
    early_stop_patience = 10
    early_stop_count = 0
    best_val_loss = float('inf')

    # 创建保存检查点的文件夹
    os.makedirs("models", exist_ok=True)

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()

        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

        for imgs, masks in train_bar:

            imgs, masks = imgs.to(device), masks.to(device)

            # 前向传播
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", learning_rate=f"{optimizer.param_groups[0]['lr']:.6f}")

        avg_train_loss = train_loss / len(train_loader)
        train_losses[cls].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        dice_score = 0.0
        First = True
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                dice_score += compute_dice(outputs, masks)
                if Visualize and First:
                    # 假设 mask 是一个 numpy 数组，shape: (H, W)，值为 0 或 1
                    plt.imshow(outputs[0][0].cpu().sigmoid().numpy() > 0.5, cmap='gray')
                    plt.title("Binary Mask")
                    plt.axis('off')
                    plt.show()
                    plt.imshow(masks[0][0].cpu().numpy(), cmap='gray')
                    plt.title("STD Mask")
                    plt.axis('off')
                    plt.show()
                    First = False

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = (dice_score / len(val_loader)).cpu().numpy()
        val_losses[cls].append(avg_val_loss)
        dice_scores[cls].append(avg_dice_score)

        # 打印训练和验证损失
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Avg Dice Score: {avg_dice_score:.4f}")

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"models/{model_type}_{encoder_name}_{Parameters}_class_{cls}.pth")
            print(f"✅ 保存最佳模型，验证损失: {best_val_loss:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            print(f"早停触发，验证损失在 {early_stop_patience} 个周期内没有改善。")
            break

    print(f"✅ 类 {cls} 训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: models/{model_type}_{encoder_name}_{Parameters}_class_{cls}.pth")

if Print:
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
    plt.savefig(f"images/{model_type}_{encoder_name}_{Parameters}_loss.png")

    plt.figure(figsize=(12, 5))
    for cls in range(1, 5):
        plt.plot(dice_scores[cls], label=f'Class {cls} Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Dice Score per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/{model_type}_{encoder_name}_{Parameters}_dice_score.png")
