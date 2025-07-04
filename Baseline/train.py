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

# 数据加载
df = pd.read_csv("../data/train.csv")
transform = T.Compose([
    T.ToTensor(),
])

dataset = None
dataset = SteelDataset(df, "../data/train_images", transform)

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
classes = 4

# 创建模型
model = None
model = get_model(model_type, encoder_name, classes).to(device)
print(f"创建模型: {model_type} 使用编码器 {encoder_name}")

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 损失函数
dice_loss = smp.losses.DiceLoss(mode='multilabel')
bce_loss = smp.losses.SoftBCEWithLogitsLoss()
focal_loss = smp.losses.FocalLoss(mode='multilabel', gamma=2.0)
# loss_fn = lambda pred, target: dice_loss(pred, target) + focal_loss(pred, target)
loss_fn = lambda pred, target: dice_loss(pred, target) + bce_loss(pred, target)

# 训练参数
num_epochs = 1
early_stop_patience = 5
early_stop_count = 0
best_val_loss = float('inf')

# 创建保存检查点的文件夹
os.makedirs("models", exist_ok=True)

train_losses = []
val_losses = []

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
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    # First = True
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            # if First == True:
            #     # 假设 mask 是一个 numpy 数组，shape: (H, W)，值为 0 或 1
            #     plt.imshow(outputs[0][0].cpu().sigmoid().numpy() > 0.5, cmap='gray')
            #     plt.title("Binary Mask")
            #     plt.axis('off')
            #     plt.show()
            #     plt.imshow(masks[0][0].cpu().numpy(), cmap='gray')
            #     plt.title("STD Mask")
            #     plt.axis('off')
            #     plt.show()
            #     First = False

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # 打印训练和验证损失
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"models/{model_type}_{encoder_name}.pth")
        print(f"✅ 保存最佳模型，验证损失: {best_val_loss:.4f}")
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= early_stop_patience:
        print(f"早停触发，验证损失在 {early_stop_patience} 个周期内没有改善。")
        break

plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

print(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.4f}")
print(f"模型保存在: models/{model_type}_{encoder_name}.pth")
