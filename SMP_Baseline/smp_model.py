import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SMPModel(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=4):
        """
        使用 SMP 库创建分割模型

        参数:
            encoder_name: 编码器backbone的名称（如resnet34, efficientnet-b0等）
            encoder_weights: 预训练权重（如'imagenet'或None）
            in_channels: 输入通道数
            classes: 输出类别数（钢铁缺陷分类的类别数）
        """
        super().__init__()

        # 创建 U-Net 模型
        self.model = smp.Unet(
            encoder_name=encoder_name,       # 选择编码器，例如 resnet34
            encoder_weights=encoder_weights, # 使用在 ImageNet 上预训练的权重
            in_channels=in_channels,         # 模型输入通道数 (RGB)
            classes=classes,                 # 模型输出通道数 (钢铁缺陷的类别)
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入图像张量，形状为 [B, C, H, W]

        返回:
            output: 预测的分割掩码，形状为 [B, classes, H, W]
        """
        return self.model(x)


# 也可以尝试其他分割模型和编码器
def get_smp_model(model_type='unet', encoder_name='resnet34', classes=4):
    """
    创建一个 SMP 模型的工厂函数

    参数:
        model_type: 分割模型的类型（'unet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus'）
        encoder_name: 编码器的名称
        classes: 输出类别数

    返回:
        model: SMP 模型实例
    """
    if model_type.lower() == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=classes,
        )
    elif model_type.lower() == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=classes,
        )
    elif model_type.lower() == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=classes,
        )
    elif model_type.lower() == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=classes,
        )
    elif model_type.lower() == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=classes,
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model
