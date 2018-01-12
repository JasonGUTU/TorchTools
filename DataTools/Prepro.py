import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os
import random
import os.path
import numpy as np
from ..functional import *


def _id(x):
    """
    return x
    :param x:
    :return:
    """
    return x


def _sigmoid_to_tanh(x):
    """
    range [0, 1] to range [-1, 1]
    :param x: tensor type
    :return: tensor
    """
    return (x - 0.5) * 2.


def _tanh_to_sigmoid(x):
    """
    range [-1, 1] to range [0, 1]
    :param x:
    :return:
    """
    return x * 0.5 + 0.5


def _255_to_tanh(x):
    """
    range [0, 255] to range [-1, 1]
    :param x:
    :return:
    """
    return (x - 127.5) / 127.5


def _tanh_to_255(x):
    """
    range [-1. 1] to range [0, 255]
    :param x:
    :return:
    """
    return x * 127.5 + 127.5


# TODO: _sigmoid_to_255(x), _255_to_sigmoid(x)
# def _sigmoid_to_255(x):
# def _255_to_sigmoid(x):


def random_pre_process(img):
    """
    Random pre-processing the input Image
    :param img: PIL.Image
    :return: PIL.Image
    """
    if bool(random.getrandbits(1)):
        img = hflip(img)
    if bool(random.getrandbits(1)):
        img = vflip(img)
    angle = random.randrange(-15, 15)
    return rotate(img, angle)

