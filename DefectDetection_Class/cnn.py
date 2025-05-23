import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 计算全连接层的输入维度
        # 经过3次池化后: 1600/8 x 256/8 = 200 x 32
        self.fc1 = nn.Linear(64 * 200 * 32, 512)
        self.fc2 = nn.Linear(512, 1)  # 二分类输出
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积块 1
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积块 2
        x = self.pool(F.relu(self.conv2(x)))
        # 卷积块 3
        x = self.pool(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(-1, 64 * 200 * 32)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()