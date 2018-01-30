import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import upsampleBlock
from .activation import swish


def _detector_parser(image_size, ):
    pass


def _2detector_label():
    pass


class LightFaceDetector(nn.Module):
    """
    This is a lightweight face landmark detector using fully convolution layer. detect m points
    |-------------------------------|
    |     Input Y Channel Image     |
    |-------------------------------|
    |  Convolution_1 n32k5s1, ReLu  |
    |-------------------------------|
    |          MaxPooling 2x        |
    |-------------------------------|
    |  Convolution_2 n32k3s1, ReLu  |
    |-------------------------------|
    |  Convolution_3 n32k3s1, ReLu  |
    |-------------------------------|
    |        Pixel Shuffle 2x       |
    |-------------------------------|
    | Convolution_4 nmk3s1, Sigmoid |
    |-------------------------------|
    """
    def __init__(self, landmarks=5, activation=swish, in_channel=1):
        super(LightFaceDetector, self).__init__()
        self.act = activation
        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(in_channel, 32, 5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.upsample = upsampleBlock(32, 128)
        self.conv4 = nn.Conv2d(32, landmarks, 3, stride=1, padding=1)

    def forward(self, input):
        return F.sigmoid(self.conv4(self.upsample(self.act(self.conv3(self.act(self.conv2(self.pool(self.act(self.conv1(input))))))))))


# class FaceDetectorFullyConnect(nn.Module):
#     """
#     This is a lightweight face landmark detector using Fully Connected Layer. detect m points
#     |-------------------------------|
#     |     Input Y Channel Image     |
#     |-------------------------------|
#     |  Convolution_1 n32k5s1, ReLu  |
#     |-------------------------------|
#     |          MaxPooling 2x        |
#     |-------------------------------|
#     |  Convolution_2 n32k3s1, ReLu  |
#     |-------------------------------|
#     |  Convolution_3 n32k3s1, ReLu  |
#     |-------------------------------|
#     |     Fully Connected Layer     |
#     |-------------------------------|
#     """
#     def __init__(self, image_size, landmarks=5, activation=swish, in_channel=1, image_size):
#         super(FaceDetectorFullyConnect, self).__init__()
#         self.act = activation
#         self.conv1 = nn.Conv2d(in_channel, 16, 5, stride=1, padding=2)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool2d(2)
#         self.conv4 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
#         self.dense = nn.Linear()




