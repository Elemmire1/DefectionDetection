import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBinary(nn.Module):
    def __init__(self):
        super(ResNetBinary, self).__init__()
        # 加载预训练的 ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 修改最后的全连接层为二分类
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x).squeeze()

def load_model():
    model = [None] * 4
    for i in range(4):
        model[i] = ResNetBinary().cuda()
        model[i].load_state_dict(torch.load("models/class_" + str(i+1) + ".pth"))
        model[i].eval()
    return model
