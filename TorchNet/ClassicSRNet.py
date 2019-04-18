import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

import math

from .modules import residualBlock, upsampleBlock, DownsamplingShuffle


class SRResNet_Residual_Block(nn.Module):
    def __init__(self):
        super(SRResNet_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class SRResNetRGBX4(nn.Module):
    def __init__(self, min=0.0, max=1.0, tanh=False):
        super(SRResNetRGBX4, self).__init__()
        self.min = min
        self.max = max
        self.tanh = tanh

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, clip=True):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        if self.tanh:
            return F.tanh(out)
        else:
            return torch.clamp(out, min=self.min, max=self.max) if clip else out


class SRResNetYX4(nn.Module):
    def __init__(self, min=0.0, max=1.0, tanh=True):
        super(SRResNetYX4, self).__init__()
        self.min = min
        self.max = max
        self.tanh = tanh

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, clip=True):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        if self.tanh:
            return F.tanh(out)
        else:
            return torch.clamp(out, min=self.min, max=self.max) if clip else out


class SRResNetYX2(nn.Module):
    def __init__(self, min=0.0, max=1.0, tanh=True):
        super(SRResNetYX2, self).__init__()
        self.min = min
        self.max = max
        self.tanh = tanh

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        if self.tanh:
            return F.tanh(out)
        else:
            return torch.clamp(out, min=self.min, max=self.max)


class DownSampleResNetYX4(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(DownSampleResNetYX4, self).__init__()
        self.min = min
        self.max = max
        self.down_shuffle = DownsamplingShuffle(4)

        self.conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 6)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(self.down_shuffle(x)))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.conv_output(out)
        return torch.clamp(out, min=self.min, max=self.max)


class FSRCNNY(nn.Module):
    """
    Sequential(
      (0): Conv2d (1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (1): LeakyReLU(0.2, inplace)
      (2): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (3): LeakyReLU(0.2, inplace)
      (4): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (5): LeakyReLU(0.2, inplace)
      (6): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (7): LeakyReLU(0.2, inplace)
      (8): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (9): LeakyReLU(0.2, inplace)
      (10): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (11): LeakyReLU(0.2, inplace)
      (12): upsampleBlock(
        (act): LeakyReLU(0.2, inplace)
        (pad): ReflectionPad2d((1, 1, 1, 1))
        (conv): Conv2d (64, 256, kernel_size=(3, 3), stride=(1, 1))
        (shuffler): PixelShuffle(upscale_factor=2)
      )
      (13): upsampleBlock(
        (act): LeakyReLU(0.2, inplace)
        (pad): ReflectionPad2d((1, 1, 1, 1))
        (conv): Conv2d (64, 256, kernel_size=(3, 3), stride=(1, 1))
        (shuffler): PixelShuffle(upscale_factor=2)
      )
      (14): Conv2d (64, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    )
    """
    def __init__(self, scala=4):
        super(FSRCNNY, self).__init__()
        self.scala = scala
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1), upsampleBlock(64, 64 * 4, activation=nn.LeakyReLU(0.2, inplace=True)))

        self.convf = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, input):
        out = self.relu1(self.conv1(input))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        x = self.relu6(self.conv6(out))

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return F.tanh(self.convf(x))

