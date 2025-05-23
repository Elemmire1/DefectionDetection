import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from cnn import CNN
from dataset import mask2rle
from resnet import ResNetBinary

def load_model_cnn():
    model = [None] * 4
    for i in range(4):
        model[i] = CNN().cuda()
        model[i].load_state_dict(torch.load("modelcnn" + str(i+1) + ".pth"))
        model[i].eval()
    return model

def load_model_mix():
    model = [None] * 4
    for i in [0, 2, 3]:
        model[i] = ResNetBinary().cuda()
        model[i].load_state_dict(torch.load("modelresnet" + str(i+1) + ".pth"))
        model[i].eval()
    model[1] = CNN().cuda()
    model[1].load_state_dict(torch.load("modelcnn" + str(2) + ".pth"))
    model[1].eval()
    return model

def load_model_resnet():
    model = [None] * 4
    for i in range(4):
        model[i] = ResNetBinary().cuda()
        model[i].load_state_dict(torch.load("modelresnet" + str(i+1) + ".pth"))
        model[i].eval()
    return model
