import os
import csv
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from model import get_smp_model
import segmentation_models_pytorch as smp
from scipy.ndimage import label

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

threshold_total = [0, 600, 600, 900, 2000]
threshold = 150
def remove_threshold(mask, cls):
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_mask, num_features = label(mask, structure=structure)
    for i in range(1, num_features + 1):
        if np.sum(labeled_mask == i) < threshold:
            labeled_mask[labeled_mask == i] = 0
    if np.sum(labeled_mask) < threshold_total[cls]:
        labeled_mask = np.zeros_like(mask)
    else:
        labeled_mask[labeled_mask > 0] = 1
    return labeled_mask


def main():
    # 模型配置
    model_type = "unet"  # 与训练时使用的模型类型保持一致
    encoder_name = "resnet34"  # 与训练时使用的编码器保持一致
    classes = 4  # 钢铁缺陷的类别数

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = get_smp_model(model_type, encoder_name, classes).to(device)
    if model_type == "unet++":
        model_path = f"pretrained/smp_unet_{encoder_name}_best.pth"
    else:
        model_path = f"pretrained/smp_{model_type}_{encoder_name}_best.pth"

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"⚠️ 模型文件不存在: {model_path}")
        print("请先运行 train_smp.py 进行模型训练，或检查模型路径是否正确。")
        return

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✅ 已加载模型: {model_path}")

    # 测试图像路径
    test_dir = "../data/test_images"
    if not os.path.exists(test_dir):
        print(f"⚠️ 测试图像目录不存在: {test_dir}")
        return

    test_imgs = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

    if not test_imgs:
        print(f"⚠️ 在 {test_dir} 中没有找到 jpg 图像")
        return

    print(f"找到 {len(test_imgs)} 张测试图像")

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 准备输出 CSV 文件
    submission_path = "./submission.csv"
    with open(submission_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId_ClassId", "EncodedPixels"])  # 写入表头

        # 预测并保存结果
        with torch.no_grad():
            for name in tqdm(test_imgs, desc="预测中"):
                img_path = os.path.join(test_dir, name)
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)

                # 预测
                pred = model(image_tensor).cpu().sigmoid()

                # 阈值处理
                pred_binary = (pred > 0.5).float().numpy().squeeze()

                # 对每个类别处理
                for cls in range(4):
                    class_pred = pred_binary[cls]
                    class_pred = remove_threshold(class_pred, cls)
                    if class_pred.sum() > 0:  # 如果有检测到缺陷
                        rle = mask2rle(class_pred)
                    else:
                        rle = ""  # 没有检测到缺陷

                    # 写入结果
                    writer.writerow([f"{name}_{cls+1}", rle])

    print(f"✅ 预测完成，结果已保存到: {submission_path}")

main()
