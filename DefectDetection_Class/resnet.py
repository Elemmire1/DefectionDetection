import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBinary(nn.Module):
    def __init__(self):
        super(ResNetBinary, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
