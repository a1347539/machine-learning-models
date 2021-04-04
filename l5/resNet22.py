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
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

def conv_2d(in_channels, out_channels, stride=1, kernel_size=3):
  return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                   kernel_size=kernel_size,stride=stride,
                   padding=kernel_size//2, bias=False)
  
def bn_relu_conv(in_channels, out_channels):
  return nn.Sequential(nn.BatchNorm2d(in_channels),
                       nn.ReLU(inplace=True),
                       conv_2d(in_channels=in_channels, out_channels=out_channels))

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.bn = nn.BatchNorm2d(in_channels)
    self.conv1 = conv_2d(in_channels=in_channels, out_channels=out_channels, stride=stride)
    self.conv2 = bn_relu_conv(in_channels=out_channels, out_channels=out_channels)
    self.shortcut = lambda x:x
    if in_channels != out_channels:
      self.shortcut = conv_2d(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=stride, kernel_size=1)
      
  def forward(self, x):
    x = F.relu(self.bn(x), inplace=True)
    r = self.shortcut(x)
    x = self.conv1(x)
    x = self.conv2(x)
    return x.add_(r)

def make_group(N, in_channels, out_channels, stride):
  start = ResidualBlock(in_channels, out_channels, stride)
  rest = [ResidualBlock(out_channels, out_channels) for j in range(1, N)]
  return [start] + rest

class Flatten(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x.view(x.size(0), -1)

class WideResNet(nn.Module):
  def __init__(self, n_groups, N, n_classes, k=1, n_start=16):
    super().__init__()
    layers = [conv_2d(3, n_start)]
    n_channels = [n_start]

    for i in range(n_groups):
      n_channels.append(n_start*(2**i)*k)
      stride = 2 if i>0 else 1
      layers += make_group(N, n_channels[i],
                           n_channels[i+1], stride)
    
    layers += [nn.BatchNorm2d(n_channels[3]),
               nn.ReLU(inplace=True),
               nn.AdaptiveAvgPool2d(1),
               Flatten(),
               nn.Linear(n_channels[3], n_classes)]

    self.features = nn.Sequential(*layers)

  def forward(self, x):
    return self.features(x)

def wrn_22():
  return WideResNet(n_groups=3, N=3, n_classes=10, k=6)
