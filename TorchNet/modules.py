import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

from .activation import swish

class FeatureExtractor(nn.Module):

    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        # TODO convert x: RGB to BGR
        return self.features(x)


class residualBlock(nn.Module):

    def __init__(self, in_channels=64, kernel=3, mid_channels=64, out_channels=64, stride=1, activation=relu):
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


class residualBlockNoBN(nn.Module):

    def __init__(self, in_channels=64, kernel=3, mid_channels=64, out_channels=64, stride=1, activation=relu):
        super(residualBlockNoBN, self).__init__()
        self.act = activation
        self.pad1 = nn.ReflectionPad2d((kernel // 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel, stride=stride, padding=0)
        self.pad2 = nn.ReflectionPad2d((kernel // 2))
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=0)

    def forward(self, x):
        y = self.act(self.conv1(self.pad1(x)))
        return self.conv2(self.pad2(y)) + x



class upsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation=relu):
        super(upsampleBlock, self).__init__()
        self.act = activation
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=0)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return self.act(self.shuffler(self.conv(self.pad(x))))


class deconvUpsampleBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_1=5, kernel_2=3, activation=relu):
        self.act = activation
        super(deconvUpsampleBlock, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_1, stride=2, padding=kernel_1 // 2)
        # self.deconv_2 = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=kernel_2, padding=kernel_2 // 2)

    def forward(self, x):
        return self.act(self.deconv_1(x))


class Features4Layer(nn.Module):
    """
    Basic feature extractor, 4 layer version
    """
    def __init__(self, features=64, activation=relu):
        """
        :param frame: The input frame image
        :param features: feature maps per layer
        """
        super(Features4Layer, self).__init__()
        self.act = activation

        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(1, features, 5, stride=1, padding=0)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(features)

        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(features)

        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(features, features, 3, stride=1, padding=0)

    def forward(self, frame):
        return self.act(self.conv4(self.pad4(
            self.act(self.bn3(self.conv3(self.pad3(
                self.act(self.bn2(self.conv2(self.pad2(
                    self.act(self.conv1(self.pad1(frame)))
                ))))
            ))))
        )))


class Features3Layer(nn.Module):
    """
    Basic feature extractor, 4 layer version
    """
    def __init__(self, features=64, activation=relu):
        """
        :param frame: The input frame image
        :param features: feature maps per layer
        """
        super(Features3Layer, self).__init__()
        self.act = activation

        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(1, features, 5, stride=1, padding=0)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(features)

        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(features, features, 3, stride=1, padding=0)

    def forward(self, frame):
        return self.act(self.conv3(self.pad3(
            self.act(self.bn2(self.conv2(self.pad2(
                self.act(self.conv1(self.pad1(frame)))
            ))))
        )))


class LateUpsamplingBlock(nn.Module):
    """
    this is another up-sample block for step upsample
    |------------------------------|
    |           features           |
    |------------------------------|
    |   n   |   residual blocks    |
    |------------------------------|
    | Pixel shuffle up-sampling x2 |
    |------------------------------|
    """
    def __init__(self, features=64, n_res_block=3):
        """
        :param features: number of feature maps input
        :param n_res_block: number of residual blocks
        """
        super(LateUpsamplingBlock, self).__init__()
        self.n_residual_blocks = n_res_block

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock(features))

        self.upsample = upsampleBlock(features, features * 4)

    def forward(self, features):
        for i in range(self.n_residual_blocks):
            features = self.__getattr__('residual_block' + str(i + 1))(features)
        return self.upsample(features)


class LateUpsamplingBlockNoBN(nn.Module):
    """
    this is another up-sample block for step upsample
    |------------------------------|
    |           features           |
    |------------------------------|
    |   n   |   residual blocks    |
    |------------------------------|
    | Pixel shuffle up-sampling x2 |
    |------------------------------|
    """
    def __init__(self, features=64, n_res_block=3):
        """
        :param features: number of feature maps input
        :param n_res_block: number of residual blocks
        """
        super(LateUpsamplingBlockNoBN, self).__init__()
        self.n_residual_blocks = n_res_block

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlockNoBN(features))

        self.upsample = upsampleBlock(features, features * 4)

    def forward(self, features):
        for i in range(self.n_residual_blocks):
            features = self.__getattr__('residual_block' + str(i + 1))(features)
        return self.upsample(features)

