import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import SteelDataset
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import os
from cnn import CNN
from resnet import ResNetBinary

# 代表是哪个种类的0/1分类器
training_catagory = 4

df = pd.read_csv("./data/train.csv")
transform = T.Compose([T.ToTensor()])
dataset = SteelDataset(
    df=df,
    image_dir='data/train_images',
    training_catagory=training_catagory,
    transform=transform
)

train_len = int(0.7 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetBinary().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

epoches = 1
# 训练 + 验证
for epoch in range(epoches):
    model.train()
    total_train_loss = 0
    for imgs, catagories in tqdm(train_loader, desc=f"Epoch {epoch+1}/"+str(epoches)+" - Training"):
        imgs, catagories = imgs.to(device), catagories.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, catagories)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # 验证阶段
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, catagories in tqdm(val_loader, desc="Validation"):
            imgs, catagories = imgs.to(device), catagories.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, catagories)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

# 保存模型
os.makedirs("checkpointsresnet", exist_ok=True)
torch.save(model.state_dict(), "modelresnet" + str(training_catagory) + ".pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "checkpointsresnet/checkpointresnet" + str(training_catagory) + ".pth")
print("模型已保存")