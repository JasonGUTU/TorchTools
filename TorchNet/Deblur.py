import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

try:
    from pytorch_fft.fft.autograd import Ifft2d, Fft2d
    import pytorch_fft.fft as fft
except:
    pass

import numpy as np
import torch

fft_cuda = Fft2d()
ifft_cuda = Ifft2d()


def complex_multi(real_1, real_2, imag_1, imag_2):
    real = real_1 * real_2 - imag_1 * imag_2
    imag = imag_2 * real_1 + imag_1 * real_2
    return real, imag


def complex_div(real_1, real_2, imag_1, imag_2):
    down = real_2 ** 2 + imag_2 ** 2
    up_real, up_imag = complex_multi(real_1, real_2, imag_1, -imag_2)
    return up_real / down, up_imag / down


def complex_abs(real, imag):
    return torch.sqrt(real ** 2 + imag ** 2)


def fftshift_cpu(fft_map, axes=None):
    ndim = len(fft_map.size())
    if axes is None:
        axes = list(range(ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    y = fft_map
    for k in axes:
        n = fft_map.size()[k]
        p2 = (n + 1) // 2
        mylist = np.concatenate((np.arange(p2, n), np.arange(p2)))
        y = torch.index_select(y, k, Variable(torch.LongTensor(mylist)))
    return y


def fftshift_cuda(fft_map, axes=None):
    ndim = len(fft_map.size())
    if axes is None:
        axes = list(range(ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    y = fft_map
    for k in axes:
        n = fft_map.size()[k]
        p2 = (n + 1) // 2
        mylist = np.concatenate((np.arange(p2, n), np.arange(p2)))
        y = torch.index_select(y, k, Variable(torch.LongTensor(mylist)).cuda())
    return y


def center_padding(canvas, H, W):
    """
    (pad_l, pad_r, pad_t, pad_b )
    """
    canvas_p = canvas // 2
    H_p = H // 2
    W_p = W // 2
    pad_t = H_p - canvas_p
    pad_b = H - pad_t - canvas
    pad_l = W_p - canvas_p
    pad_r = W - pad_l - canvas
    return pad_l, pad_r, pad_t, pad_b


def kernel_filp(kernel):
    canvas_1 = kernel.size()[1]
    canvas_2 = kernel.size()[2]
    kernel_h = torch.index_select(kernel, 1, Variable(torch.LongTensor(np.arange(canvas_1-1, -1, -1))))
    kernel_w = torch.index_select(kernel_h, 2, Variable(torch.LongTensor(np.arange(canvas_2-1, -1, -1))))
    return kernel_w


def convolution_fft_cuda(image, psf, canvas=64, eps=1e-8):
    img_real = image
    _, H, W = image.size()
    img_imag = torch.zeros_like(img_real).cuda()
    psf_pad_real = F.pad(psf, center_padding(canvas, H, W))
    psf_pad_imag = torch.zeros_like(psf_pad_real).cuda()
    input_fft_r, input_fft_i = fft_cuda(img_real, img_imag)
    psf_fft_r, psf_fft_i = fft_cuda(psf_pad_real, psf_pad_imag)
    conv_real, conv_imag = complex_multi(input_fft_r, psf_fft_r + eps, input_fft_i, psf_fft_i)
    conv = ifft_cuda(conv_real, conv_imag)
    conv_mage = torch.sqrt(conv[0] ** 2 + conv[1] ** 2)
    conv_img = fftshift_cuda(conv_mage)
    return conv_img


def inverse_convolution_cuda(image, psf, canvas=64, eps=1e-8):
    img_real = image
    _, H, W = image.size()
    img_imag = torch.zeros_like(img_real).cuda()
    psf_pad_real = F.pad(psf, center_padding(canvas, H, W))
    psf_pad_imag = torch.zeros_like(psf_pad_real).cuda()
    input_fft_r, input_fft_i = fft_cuda(img_real, img_imag)
    psf_fft_r, psf_fft_i = fft_cuda(psf_pad_real, psf_pad_imag)
    deconv_real, deconv_imag = complex_div(input_fft_r, psf_fft_r + eps, input_fft_i, psf_fft_i)
    deconv = ifft_cuda(deconv_real, deconv_imag)
    deconv_mage = torch.sqrt(deconv[0] ** 2 + deconv[1] ** 2)
    deconv_img = fftshift_cuda(deconv_mage)
    return deconv_img


def wiener_filter_cuda(image, psf, canvas=64, eps=1e-8, K=1e-8):
    img_real = image
    _, H, W = image.size()
    img_imag = torch.zeros_like(img_real).cuda()
    psf_pad_real = F.pad(psf, center_padding(canvas, H, W))
    psf_pad_imag = torch.zeros_like(psf_pad_real).cuda()
    input_fft_r, input_fft_i = fft_cuda(img_real, img_imag)
    psf_fft_r, psf_fft_i = fft_cuda(psf_pad_real, psf_pad_imag)
    psf_fft_r += eps
    psf_fft_prime_r, psf_fft_prime_i = complex_div(psf_fft_r, psf_fft_r**2 + psf_fft_i**2 + K, -psf_fft_i, 0.0)
    deconv_r, deconv_i = complex_multi(input_fft_r, psf_fft_prime_r, input_fft_i, psf_fft_prime_i)
    deconv = ifft_cuda(deconv_r, deconv_i)
    deconv_mage = torch.sqrt(deconv[0] ** 2 + deconv[1] ** 2)
    deconv_img = fftshift_cuda(deconv_mage)
    return deconv_img

