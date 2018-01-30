import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

from .modules import Features4Layer, Features3Layer, residualBlock, upsampleBlock, LateUpsamplingBlock
from .activation import swish




class MaxActivationFusion(nn.Module):
    """
    model implementation of the Maximum-activation Detail Fusion
    This is not a complete SR model, just **Fusion Part**
    """
    def __init__(self, features=64, feature_extractor=Features4Layer, activation=swish):
        """
        :param features: the number of final feature maps
        """
        super(MaxActivationFusion, self).__init__()
        self.features = feature_extractor(features, activation=activation)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        """
        :param frame_1: frame t-2
        :param frame_2: frame t-1
        :param frame_3: frame t
        :param frame_4: frame t+1
        :param frame_5: frame t+2
        :return: features
        """
        frame_1_feature = self.features(frame_1)
        frame_2_feature = self.features(frame_2)
        frame_3_feature = self.features(frame_3)
        frame_4_feature = self.features(frame_4)
        frame_5_feature = self.features(frame_5)

        frame_1_feature = frame_1_feature.view((1, ) + frame_1_feature.size())
        frame_2_feature = frame_2_feature.view((1, ) + frame_2_feature.size())
        frame_3_feature = frame_3_feature.view((1, ) + frame_3_feature.size())
        frame_4_feature = frame_4_feature.view((1, ) + frame_4_feature.size())
        frame_5_feature = frame_5_feature.view((1, ) + frame_5_feature.size())

        cat = torch.cat((frame_1_feature, frame_2_feature, frame_3_feature, frame_4_feature, frame_5_feature), dim=0)
        return torch.max(cat, 0)[0]


class EaryFusion(nn.Module):
    """
    model implementation of the Maximum-activation Detail Fusion
    This is not a complete SR model, just **Fusion Part**
    """
    def __init__(self, features=64, feature_extractor=Features3Layer, activation=swish):
        """
        :param features: the number of final feature maps
        """
        super(EaryFusion, self).__init__()
        self.act = activation
        self.features = feature_extractor(features, activation=activation)
        self.conv = nn.Conv2d(features * 5, features, 1, stride=1, padding=0)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        """
        :param frame_1: frame t-2
        :param frame_2: frame t-1
        :param frame_3: frame t
        :param frame_4: frame t+1
        :param frame_5: frame t+2
        :return: features
        """
        return self.act(self.conv(
            torch.cat(
                (self.features(frame_1),
                 self.features(frame_2),
                 self.features(frame_3),
                 self.features(frame_4),
                 self.features(frame_5)),
                dim=1)
        ))


class HallucinationOrigin(nn.Module):
    """
    Original Video Face Hallucination Net
    |---------------------------------|
    |         Input features          |
    |---------------------------------|
    |   n   |    Residual blocks      |
    |---------------------------------|
    | Big short connect from features |
    |---------------------------------|
    |       Convolution and BN        |
    |---------------------------------|
    |    Pixel Shuffle Up-sampling    |
    |---------------------------------|
    |         Final Convolution       |
    |---------------------------------|
    |              Tanh               |
    |---------------------------------|
    """
    def __init__(self, scala=8, features=64, n_residual_blocks=9, big_short_connect=False, output_channel=1):
        """
        :param scala: scala factor
        :param n_residual_blocks: The number of residual blocks
        :param Big_short_connect: Weather the short connect between the input features and the Conv&BN
        """
        super(HallucinationOrigin, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.scala = scala
        self.connect = big_short_connect

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock(features))

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(features)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1), upsampleBlock(features, features * 4))

        self.pad2 = nn.ReflectionPad2d(3)
        self.conv2 = nn.Conv2d(features, output_channel, 7, stride=1, padding=0)

    def forward(self, features):
        y = features.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        if self.connect:
            x = self.bn(self.conv(self.pad(y))) + features
        else:
            x = self.bn(self.conv(self.pad(y)))

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return  F.tanh(self.conv2(self.pad2(x)))


class StepHallucinationNet(nn.Module):
    """
    |-----------------------------------|
    |             features              |
    |-----------------------------------|
    | log2(scala) | LateUpsamplingBlock |
    |-----------------------------------|
    |       Convolution and Tanh        |
    |-----------------------------------|
    """
    def __init__(self, scala=8, features=64, little_res_blocks=3, output_channel=1):
        """

        :param scala: scala factor
        :param features:
        :param little_res_blocks: The number of residual blocks in every late upsample blocks
        :param output_channel: default to be 1 for Y channel
        """
        super(StepHallucinationNet, self).__init__()
        self.scala = scala
        self.features = features
        self.n_res_blocks = little_res_blocks

        for i in range(int(log2(self.scala))):
            self.add_module('lateUpsampling' + str(i + 1), LateUpsamplingBlock(features, n_res_block=little_res_blocks))

        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(features, output_channel, 7, stride=1, padding=0)

    def forward(self, features):
        for i in range(int(log2(self.scala))):
            features = self.__getattr__('lateUpsampling' + str(i + 1))(features)
        return F.tanh(self.conv(self.pad(features)))













