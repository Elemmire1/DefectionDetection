import os
import csv
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from model import get_smp_model
import segmentation_models_pytorch as smp
from scipy.ndimage import label
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

def mask2rle(mask):
    """
    将掩码转换为 RLE (Run Length Encoding) 格式
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    starts = runs[0::2]
    ends = runs[1::2]
    lengths = ends - starts
    rle = ' '.join(str(s) + ' ' + str(l) for s, l in zip(starts, lengths))
    return rle

threshold_total = [0, 450, 550, 700, 2000]
threshold = [0, 250, 300, 50, 50]
def remove_threshold(mask, cls):
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_mask, num_features = label(mask, structure=structure)
    for i in range(1, num_features + 1):
        if np.sum(labeled_mask == i) < threshold[cls]:
            labeled_mask[labeled_mask == i] = 0
    if np.sum(labeled_mask) < threshold_total[cls]:
        labeled_mask = np.zeros_like(mask)
    else:
        labeled_mask[labeled_mask > 0] = 1
    return labeled_mask

def main():
    results = []

    for cls in range(1, 5):
        print(f"正在处理类别 {cls} 的预测...")
        # 模型配置
        model_type1 = "unet"  # 与训练时使用的模型类型保持一致
        encoder_name1 = "efficientnet-b3"  # 与训练时使用的编码器保持一致
        model_type2 = "deeplabv3"  # 与训练时使用的模型类型保持一致
        encoder_name2 = "efficientnet-b3"  # 与训练时使用的编码器保持一致
        classes = 1  # 钢铁缺陷的类别数

        # 设备配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 加载模型
        model1 = get_smp_model(model_type1, encoder_name1, classes).to(device)
        if model_type1 == "unet++":
            model_path1 = f"pretrained/{cls}_unet++_{encoder_name1}_best.pth"
        else:
            model_path1 = f"pretrained/{cls}_{model_type1}_{encoder_name1}_best.pth"
        model2 = get_smp_model(model_type2, encoder_name2, classes).to(device)
        if model_type2 == "unet++":
            model_path2 = f"pretrained/{cls}_unet++_{encoder_name2}_best.pth"
        else:
            model_path2 = f"pretrained/{cls}_{model_type2}_{encoder_name2}_best.pth"

        # 检查模型是否存在
        if not os.path.exists(model_path1):
            print(f"⚠️ 模型文件不存在: {model_path1}")
            print("请先运行 train_smp.py 进行模型训练，或检查模型路径是否正确。")
            return
        if not os.path.exists(model_path2):
            print(f"⚠️ 模型文件不存在: {model_path2}")
            print("请先运行 train_smp.py 进行模型训练，或检查模型路径是否正确。")
            return

        # 加载模型权重
        model1.load_state_dict(torch.load(model_path1))
        model1.eval()
        print(f"✅ 已加载模型: {model_path1}")
        model2.load_state_dict(torch.load(model_path2))
        model2.eval()
        print(f"✅ 已加载模型: {model_path2}")

        # 测试图像路径
        test_dir = "../data/test_images"
        if not os.path.exists(test_dir):
            print(f"⚠️ 测试图像目录不存在: {test_dir}")
            return

        df = pd.read_csv(f"prediction_task.csv")
        test_imgs = []
        for name in os.listdir(test_dir):
            if name.endswith('.jpg'):
                prob = df.loc[
                    (df['ImageId'] == name),
                    str(cls)
                ].values
                if len(prob) and prob[0] > 0.5:
                    test_imgs.append(name)
                else:
                    results.append([f"{name}_{cls}", ""])

        # test_imgs = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

        if not test_imgs:
            print(f"⚠️ 在 {test_dir} 中没有找到 jpg 图像")
            return

        print(f"找到 {len(test_imgs)} 张测试图像")

        # 准备输出 CSV 文件
        submission_path = "./submission.csv"

        with torch.no_grad():
            for name in tqdm(test_imgs, desc="预测中"):
                img_path = os.path.join(test_dir, name)
                image = Image.open(img_path).convert("RGB")
                image = np.array(image)
                aug = transform(image=image)
                image_tensor = aug['image'].unsqueeze(0).to(device)

                # 预测
                pred1 = model1(image_tensor).cpu().sigmoid()
                pred2 = model2(image_tensor).cpu().sigmoid()
                pred = (pred1 + pred2) / 2

                # 阈值处理
                pred_binary = (pred > 0.5).float().numpy().squeeze()
                class_pred = pred_binary[0] if len(pred_binary.shape) > 2 else pred_binary

                class_pred = remove_threshold(class_pred, cls)
                if class_pred.sum() > 0:  # 如果有检测到缺陷
                    rle = mask2rle(class_pred)
                else:
                    rle = ""  # 没有检测到缺陷

                # 添加到结果列表
                results.append([f"{name}_{cls}", rle])

    df = pd.DataFrame(results, columns=["ImageId_ClassId", "EncodedPixels"])
    df.to_csv(submission_path, index=False)
    print(f"✅ 预测完成，结果已保存到: {submission_path}")

main()
