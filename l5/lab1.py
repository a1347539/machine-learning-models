import os
import numpy as np
import torch
import torchvision
import tarfile
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tt
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from resNet9 import *

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

model = rn_9()

model.load_state_dict(torch.load('cifar10-resnet9.pth', map_location=torch.device('cpu')))

title = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

img = Image.open('1.jfif')
newsize = (32, 32)
img = img.resize(newsize)

img = tfms(img)

valid_ds = ImageFolder('./test', tfms)

def prediction_image(image):
    xb = image.unsqueeze(0)
    preds = model(xb)
    _, pred = torch.max(preds, dim=1)
    return valid_ds.classes[pred[0].item()], pred[0]

def get_info(data, flag=True):
    if flag:
        img, label = data
        print('label:', valid_ds.classes[label], 'prediction:', prediction_image(img))
    if not flag:
        img = data
        print('prediction:', prediction_image(img)[0])
    plt.imshow(img.permute(1,2,0))

    plt.show()

get_info(img, False)
