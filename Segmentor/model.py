import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

Kaggle = False

def get_model(model_type='unet', encoder_name='resnet34', classes=1):
    """
    参数:
        model_type: 分割模型的类型（'unet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'unet++'）
        encoder_name: 编码器的名称
        classes: 输出类别数

    返回:
        model: SMP 模型实例
    """
    if model_type.lower() == 'unet':
        if Kaggle == True:
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=classes,
            )
        else:
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=3,
                classes=classes,
            )
    elif model_type.lower() == 'deeplabv3':
        if Kaggle == True:
            model = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=classes,
            )
        else:
            model = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=3,
                classes=classes,
            )

    if Kaggle == True:
        if encoder_name == "resnet34":
            state_dict = torch.load("/kaggle/input/resnet34/resnet34-333f7ec4.pth", weights_only=False)
        elif encoder_name == "resnet50":
            state_dict = torch.load("/kaggle/input/resnet50/resnet50-19c8e357.pth", weights_only=False)
        elif encoder_name == "efficientnet-b3":
            state_dict = torch.load("/kaggle/input/efficientnetb3/efficientnet-b3-5fb5a3c3.pth", weights_only=False)
        else:
            raise ValueError(f"不支持的 Encoder：{encoder_name}")

        model.encoder.load_state_dict(state_dict)

    return model
