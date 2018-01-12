import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import swish

class FeatureExtractor(nn.Module):

    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        # TODO convert x: RGB to BGR
        return self.features(x)


class residualBlock(nn.Module):

    def __init__(self, in_channels=64, kernel=3, mid_channels=64, out_channels=64, stride=1, activation=swish):
        super(residualBlock, self).__init__()
        self.act = activation
        self.pad1 = nn.ReflectionPad2d((kernel // 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.pad2 = nn.ReflectionPad2d((kernel // 2))
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(self.pad1(x))))
        return self.bn2(self.conv2(self.pad2(y))) + x


class upsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation=swish):
        super(upsampleBlock, self).__init__()
        self.act = activation
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=0)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return self.act(self.shuffler(self.conv(self.pad(x))))


class deconvUpsampleBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_1=5, kernel_2=3, activation=swish):
        self.act = activation
        super(deconvUpsampleBlock, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_1, stride=2, padding=kernel_1 // 2)
        # self.deconv_2 = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=kernel_2, padding=kernel_2 // 2)

    def forward(self, x):
        return self.act(self.deconv_1(x))

