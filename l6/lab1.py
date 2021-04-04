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

latent_size = 64
hidden_size = 256
image_size = 784

G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

G.load_state_dict(torch.load("G.ckpt", map_location=torch.device('cpu')))

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

out = G(torch.randn(1, latent_size))

img = denorm(out.reshape(-1, 28, 28)).detach()
img.shape

plt.imshow(img.permute(1,2,0), cmap='gray')

plt.show()
