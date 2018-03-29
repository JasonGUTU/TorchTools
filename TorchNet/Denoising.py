import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)


def GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=-1.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(np.random.normal(loc=mean, scale=sigma, size=size))
    return torch.clamp(noise + tensor, min=min, max=max)


def PoissonNoising(tensor, lamb, noise_size=None, min=-1.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(np.random.poisson(lam=lamb, size=size))
    return torch.clamp(noise + tensor, min=min, max=max)


class DeNet1641(nn.Module):
    """
    1641 parameters
    No residual
    Leaky ReLU
    """
    def __init__(self):
        super(DeNet1641, self).__init__()
        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(0.15)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(0.15)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.LeakyReLU(0.15)
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.LeakyReLU(0.15)
        self.pad5 = nn.ReflectionPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.tanh = F.tanh

    def forward(self, input):
        return self.tanh(self.conv5(self.pad5(
            self.relu4(self.conv4(self.pad4(
                self.relu3(self.conv3(self.pad3(
                    self.relu2(self.conv2(self.pad2(
                        self.relu1(self.conv1(self.pad1(input)))
                    )))
                )))
            )))
        )))








