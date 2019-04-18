try:
    from math import log10
except ImportError:
    from math import log
    def log10(x):
        return log(x) / log(10.)

import torch

from .functional import to_tensor


def mse(x, y):
    """
    MSE Error
    :param x: tensor
    :param y: tensor
    :return: float
    """
    diff = x - y
    diff = diff * diff
    return torch.mean(diff)


def psnr(x, y, peak=1.):
    """
    psnr from tensor
    :param x: tensor
    :param y: tensor
    :return: float (mse, psnr)
    """
    _mse = mse(x, y)
    return _mse, 10 * log10((peak ** 2) / _mse)


def PSNR(x, y):
    """
    PSNR from PIL.Image
    :param x: PIL.Image
    :param y: PIL.Image
    :return: float (mse, psnr)
    """
    return psnr(to_tensor(x), to_tensor(y), peak=1.)



