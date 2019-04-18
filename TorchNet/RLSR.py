import numpy as np
import random
import functools
from math import ceil

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

import math

from ..DataTools.Loaders import PIL2VAR, VAR2PIL
from ..Functions import functional as Func
from .modules import residualBlock, upsampleBlock, DownsamplingShuffle, Attention, Flatten, BatchBlur, b_GaussianNoising, b_GPUVar_Bicubic, b_CPUVar_Bicubic
from .activation import swish
from .ClassicSRNet import SRResNet_Residual_Block


triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))


def kernel_visualize(kernel, l=15, scale=8):
    tensor_kernel = torch.FloatTensor(kernel)
    pil_kernel = VAR2PIL(Variable(tensor_kernel).view((1,1,l,l)) / tensor_kernel.max())
    pil_re_kernel = Func.resize(pil_kernel, l*scale, interpolation=Image.NEAREST)
    return pil_re_kernel


def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.25, sig_max=4.0, scaling=3, l=15, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.25, sig_max=4.0, l=15, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=15, sig_min=0.25, sig_max=4.0, rate_iso=0.3, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def random_batch_kernel(batch, l=15, sig_min=0.25, sig_max=4.0, rate_iso=0.3, scaling=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


class BatchSRKernel(object):
    def __init__(self, l=15, sig_min=0.25, sig_max=4.0, rate_iso=0.3, scaling=3):
        self.l = l
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, batch, tensor=False):
        return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate, scaling=self.scaling, tensor=tensor)


def random_noise_level(high, rate_cln=0.3):
    if np.random.uniform() < rate_cln:
        return 0.0
    else:
        return np.random.uniform() * high


def random_batch_noise(batch, high, rate_cln=0.2):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def _trajectory(canvas=64, iters=2000, max_len=60, impulsive_shake=0.005, inertia=0.7, big_shake=0.2, gaussian_shake=10.0):
    tot_len = 0
    big_expl_count = 0
    x = np.array([complex(0.0, 0.0)] * (iters))

    expl = impulsive_shake
    centripetal = inertia * np.random.uniform(0, 1)
    # probability of big shake
    prob_big_shake = big_shake * np.random.uniform(0, 1)
    # term determining, at each sample, the random component of the new direction
    gaussian_shake = gaussian_shake * np.random.uniform(0, 1)
    init_angle = 360 * np.random.uniform(0, 1)

    img_v0 = np.sin(np.deg2rad(init_angle))
    real_v0 = np.cos(np.deg2rad(init_angle))

    v0 = complex(real_v0, img_v0)
    v = v0 * max_len / (iters - 1)

    if expl > 0:
        v = v0 * expl

    for t in range(iters - 1):
        if np.random.uniform() < prob_big_shake * expl:
            next_direction = 2 * v * (np.exp(complex(0, np.pi + (np.random.uniform() - 0.5))))
            big_expl_count += 1
        else:
            next_direction = 0

        dv = next_direction + expl * (gaussian_shake * complex(np.random.randn(), np.random.randn()) - centripetal * x[t]) * (max_len / (iters - 1))

        v += dv
        v = (v / float(np.abs(v))) * (max_len / float((iters - 1)))
        x[t + 1] = x[t] + v
        tot_len += abs(x[t + 1] - x[t])

    # center the motion
    x += complex(float(-np.min(x.real)), float(-np.min(x.imag)))
    x = x - complex(x[0].real % 1., x[0].imag % 1.) + complex(1, 1)
    x += complex(ceil((canvas - max(x.real)) / 2), ceil((canvas - max(x.imag)) / 2))

    return x, tot_len, big_expl_count


def _PSF_resample(canvas, trajectory):
    PSF = np.zeros((canvas, canvas))
    iters = len(trajectory)
    prevT = 0.0
    vT = 1.0

    for t in range(iters):
        # print(j, t)
        if (vT * iters >= t) and (prevT * iters < t - 1):
            t_proportion = 1
        elif (vT * iters >= t - 1) and (prevT * iters < t - 1):
            t_proportion = vT * iters - (t - 1)
        elif (vT * iters >= t) and (prevT * iters < t):
            t_proportion = t - (prevT * iters)
        elif (vT * iters >= t - 1) and (prevT * iters < t):
            t_proportion = (vT - prevT) * iters
        else:
            t_proportion = 0

        m2 = int(np.minimum(canvas - 1, np.maximum(1, np.math.floor(trajectory[t].real))))
        M2 = int(m2 + 1)
        m1 = int(np.minimum(canvas - 1, np.maximum(1, np.math.floor(trajectory[t].imag))))
        M1 = int(m1 + 1)

        PSF[m1, m2] += t_proportion * triangle_fun_prod(trajectory[t].real - m2, trajectory[t].imag - m1)
        PSF[m1, M2] += t_proportion * triangle_fun_prod(trajectory[t].real - M2, trajectory[t].imag - m1)
        PSF[M1, m2] += t_proportion * triangle_fun_prod(trajectory[t].real - m2, trajectory[t].imag - M1)
        PSF[M1, M2] += t_proportion * triangle_fun_prod(trajectory[t].real - M2, trajectory[t].imag - M1)

    return PSF / (iters)


def random_motion_blur(canvas):
    x = _trajectory(canvas)[0]
    return _PSF_resample(canvas, x)


def batch_motion_blur(batch, canvas, tensor=True):
    batch_kernel = np.zeros((batch, canvas, canvas))
    for i in range(batch):
        batch_kernel[i] = random_motion_blur(canvas)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k]


class PCAEncoder(object):
    def __init__(self, weight, cuda=False):
        self.weight = torch.load(weight)
        self.size = self.weight.size()
        if cuda:
            self.weight = Variable(self.weight).cuda()
        else:
            self.weight = Variable(self.weight)

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size()
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))

    def decode(self, code):
        code = code.view([15,1])



class SRMDPreprocessing(object):
    def __init__(self, scala, pca, para_input=15, kernel=15, noise=True, cuda=False, sig_min=0.25, sig_max=4.0, rate_iso=0.3, scaling=3, rate_cln=0.2, noise_high=0.08):
        self.encoder = PCAEncoder(pca, cuda=cuda)
        self.kernel_gen = BatchSRKernel(l=kernel, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
        self.blur = BatchBlur(l=kernel)
        self.para_in = para_input
        self.l = kernel
        self.noise = noise
        self.scala = scala
        self.para_input = 15
        self.cuda = cuda
        self.rate_cln = rate_cln
        self.noise_high = noise_high

    def __call__(self, hr_tensor, kernel=False):
        ### hr_tensor is tensor, not cuda tensor
        B, C, H, W = hr_tensor.size()
        b_kernels = Variable(self.kernel_gen(B, tensor=True)).cuda() if self.cuda else Variable(self.kernel_gen(B, tensor=True))
        # blur
        if self.cuda:
            hr_blured_var = self.blur(Variable(hr_tensor).cuda(), b_kernels)
        else:
            hr_blured_var = self.blur(Variable(hr_tensor), b_kernels)
        # kernel encode
        kernel_code = self.encoder(b_kernels) # B x self.para_input
        # Down sample
        if self.cuda:
            lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scala)
        else:
            lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scala)

        # Noisy
        if self.noise:
            Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            lr_noised_t = b_GaussianNoising(lr_blured_t, Noise_level)
        else:
            Noise_level = torch.zeros((B, 1))
            lr_noised_t = lr_blured_t

        if self.cuda:
            Noise_level = Variable(Noise_level).cuda()
            re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code
            lr_re = Variable(lr_noised_t).cuda()
        else:
            Noise_level = Variable(Noise_level)
            re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code
            lr_re = Variable(lr_noised_t)
        return (lr_re, re_code, b_kernels) if kernel else (lr_re, re_code)

    def processing(self, hr_tensor, kernels, noise_level=0., noise=True, cuda=False):
        # all input: tensor
        B, C, H, W = hr_tensor.size()
        B_k, H_k, W_k = kernels.size()
        B_n, N_n = noise_level.size()
        assert (B == B_k) and (B == B_n), 'check Batch size'
        kernels = Variable(kernels)
        if cuda:
            hr_blured_var = self.blur(Variable(hr_tensor).cuda(), kernels.cuda())
        else:
            hr_blured_var = self.blur(Variable(hr_tensor), kernels)

        # kernel encode
        kernel_code = self.encoder(kernels)  # B x self.para_input
        if cuda:
            lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scala)
        else:
            lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scala)

        if noise:
            lr_noised_t = b_GaussianNoising(lr_blured_t, noise_level)
        else:
            lr_noised_t = lr_blured_t

        if self.cuda:
            Noise_level = Variable(noise_level).cuda()
            re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if noise else kernel_code
            lr_re = Variable(lr_noised_t).cuda()
        else:
            Noise_level = Variable(noise_level)
            re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if noise else kernel_code
            lr_re = Variable(lr_noised_t)
        return lr_re, re_code


class IsoGaussian(object):
    def __init__(self, scala, para_input=1, kernel=15, noise=False, cuda=False, sig_min=0.2, sig_max=4.0, noise_high=0.0):
        self.blur = BatchBlur(l=kernel)
        self.min = sig_min
        self.max = sig_max
        self.para_in = para_input
        self.l = kernel
        self.noise = noise
        self.scala = scala
        self.cuda = cuda
        self.noise_high = noise_high

    def __call__(self, hr_tensor):
        B, C, H, W = hr_tensor.size()
        kernel_width = np.random.uniform(low=self.min, high=self.max, size=(B, 1))
        batch_kernel = np.zeros((B, self.l, self.l))
        for i in range(B):
            batch_kernel[i] = isotropic_gaussian_kernel(self.l, kernel_width[i], tensor=False)
        kernels = Variable(torch.FloatTensor(batch_kernel))

        if self.cuda:
            hr_blured_var = self.blur(Variable(hr_tensor).cuda(), kernels.cuda())
        else:
            hr_blured_var = self.blur(Variable(hr_tensor), kernels)

        # kernel encode
        kernel_code = Variable(torch.FloatTensor(kernel_width))  # B x self.para_input
        if self.cuda:
            lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scala)
        else:
            lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scala)

        if self.noise:
            lr_noised_t = b_GaussianNoising(lr_blured_t, self.noise_high)
        else:
            lr_noised_t = lr_blured_t

        if self.cuda:
            re_code = kernel_code.cuda()
            lr_re = Variable(lr_noised_t).cuda()
        else:
            re_code = kernel_code
            lr_re = Variable(lr_noised_t)
        return lr_re, re_code


class SRCNN_CAB(nn.Module):
    def __init__(self, input_c=1, FC=2048, l=15, scala=2):
        super(SRCNN_CAB, self).__init__()
        self.input_c = input_c
        self.scala = scala
        self.kernel_linear = nn.Linear(in_features=l*l, out_features=FC, bias=True)
        self.W1_linear = nn.Linear(in_features=FC, out_features=input_c*64*9*9, bias=True)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=True, padding=1)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=input_c, kernel_size=5, bias=True, padding=2)

    def forward(self, img, kernel):
        kernel_flat = kernel.view((kernel.size(0), -1))
        kernel_FC = F.relu(self.kernel_linear(kernel_flat))
        W1 = self.W1_linear(kernel_FC)
        W1_ = W1.view((kernel.size(0), 64, self.input_c, 9, 9))
        img_up = F.upsample(img, scale_factor=self.scala, mode='bilinear')
        conv1 = []
        for i in range(kernel.size(0)):
            conv1.append(F.conv2d(img_up[i:i+1, :, :, :], W1_[i], padding=4))
        Feature1 = F.relu(torch.cat(conv1, dim=0))
        ## out_channels x in_channels/groups x kH x kW
        Feature2 = F.relu(self.Conv2(Feature1))
        out = self.Conv3(Feature2)
        return out


class FSRCNNY_MD(nn.Module):
    def __init__(self, scala=4, input_para=15, min=0.0, max=1.0):
        super(FSRCNNY_MD, self).__init__()
        self.scala = scala
        self.min = min
        self.max = max

        self.conv1 = nn.Conv2d(in_channels=1 + input_para, out_channels=64, kernel_size=7, stride=1, padding=3,
                               bias=False)
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
            self.add_module('upsample' + str(i + 1),
                            upsampleBlock(64, 64 * 4, activation=nn.LeakyReLU(0.2, inplace=True)))

        self.convf = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))
        cat_input = torch.cat([input, code_exp], dim=1)
        out = self.relu1(self.conv1(cat_input))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        x = self.relu6(self.conv6(out))

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return torch.clamp(self.convf(x), min=self.min, max=self.max) if clip else self.convf(x)


class SRMD_Block(nn.Module):
    def __init__(self, ch_in, bn=True):
        super(SRMD_Block, self).__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(ch_in, 128, 3, 1, 1),  # ch_in, ch_out, kernel_size, stride, pad
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(ch_in, 128, 3, 1, 1),  # ch_in, ch_out, kernel_size, stride, pad
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        return self.block(x)


class SRMD_Net(nn.Module):
    def __init__(self, scala=4, input_para=15, min=0.0, max=1.0, bn=True):
        super(SRMD_Net, self).__init__()
        self.scala = scala
        self.min = min
        self.max = max
        self.input_para = input_para
        self.bn = bn
        self.net = self.make_net()

    def make_net(self):
        layers = [
            SRMD_Block(self.input_para + 3, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),

            nn.Conv2d(in_channels=128, out_channels=self.scala**2 * 3, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=self.scala)
        ]
        return nn.Sequential(*layers)

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        # B, C_l, H_l = code.size()
        B, C_l= code.size()
        # code_exp = code.view((B, C_l * H_l, 1, 1)).expand((B, H_l, H, W))
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))
        cat_input = torch.cat([input, code_exp], dim=1)
        result = self.net(cat_input)
        return result
        # return torch.clamp(self.convf(x), min=self.min, max=self.max) if clip else self.convf(x)


class CodeInit(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, pooling='avg'):
        super(CodeInit, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ])

        if pooling == 'avg':
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.globalPooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.Dense = nn.Sequential(*[
            nn.Linear(ndf, ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf * 2, code_len),
        ])

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        flat_dense = flat.view(flat.size()[:2])
        dense = self.Dense(flat_dense)
        return dense


class CodeAgent(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, pooling='avg'):
        super(CodeAgent, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ])

        if pooling == 'avg':
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.globalPooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_len, ndf),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Linear(ndf * 2, ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf * 2, code_len)
        ])

    def forward(self, input, code):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        flat_dense = flat.view(flat.size()[:2])
        code_dense = self.code_dense(code)
        global_dense = torch.cat([flat_dense, code_dense], dim=1)
        code_residual = self.global_dense(global_dense)
        return torch.add(code, code_residual)


class SFT_Layer(nn.Module):
    def __init__(self, ndf=64, para=15):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + ndf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, ndf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + ndf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, ndf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat([feature_maps, para_maps], dim=1)
        mul = F.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class SFT_Residual_Block(nn.Module):
    def __init__(self, ndf=64, para=15):
        super(SFT_Residual_Block, self).__init__()
        self.sft1 = SFT_Layer(ndf=ndf, para=para)
        self.sft2 = SFT_Layer(ndf=ndf, para=para)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, feature_maps, para_maps):
        x = feature_maps
        fea1 = F.relu(self.sft1(x, para_maps))
        fea2 = F.relu(self.sft2(self.conv1(fea1), para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)


class SFTMD(nn.Module):
    def __init__(self, input_channel=1, input_para=15, scala=4, min=0.0, max=1.0, residuals=16):
        super(SFTMD, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.reses = residuals

        self.conv1 = nn.Conv2d(input_channel + input_para, 64, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        for i in range(residuals):
            self.add_module('SFT-residual' + str(i + 1), SFT_Residual_Block(ndf=64, para=input_para))

        self.sft_mid = SFT_Layer(ndf=64, para=input_para)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.scala = scala
        if scala == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scala == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64*9, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scala == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=input_channel, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))

        input_cat = torch.cat([input, code_exp], dim=1)
        before_res = self.conv3(
                self.relu_conv2(self.conv2(
                    self.relu_conv1(self.conv1(input_cat))
                ))
            )

        res = before_res
        for i in range(self.reses):
            res = self.__getattr__('SFT-residual' + str(i + 1))(res, code_exp)

        mid = self.sft_mid(res, code_exp)
        mid = F.relu(mid)
        mid = self.conv_mid(mid)

        befor_up = torch.add(before_res, mid)

        uped = self.upscale(befor_up)

        out = self.conv_output(uped)
        return torch.clamp(out, min=self.min, max=self.max) if clip else out


class CodeAgentV2(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=32, norm_layer=nn.InstanceNorm2d, pooling='avg'):
        super(CodeAgentV2, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf * 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf * 2),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf * 2),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf * 2),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf * 2),
            nn.ReLU(),
        ])

        self.ConvNetv1 = nn.Sequential(*[
            nn.Conv2d(ndf * 2, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
        ])

        self.ConvNetv2 = nn.Sequential(*[
            nn.Conv2d(ndf * 2, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
        ])

        self.ConvNetv3 = nn.Sequential(*[
            nn.Conv2d(ndf * 2, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(),
        ])

        if pooling == 'avg':
            self.globalPoolingV1 = nn.AdaptiveAvgPool2d((1, 1))
            self.globalPoolingV2 = nn.AdaptiveAvgPool2d((1, 1))
            self.globalPoolingV3 = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.globalPoolingV1 = nn.AdaptiveMaxPool2d((1, 1))
            self.globalPoolingV2 = nn.AdaptiveMaxPool2d((1, 1))
            self.globalPoolingV3 = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.globalPoolingV1 = nn.AdaptiveAvgPool2d((1, 1))
            self.globalPoolingV2 = nn.AdaptiveAvgPool2d((1, 1))
            self.globalPoolingV3 = nn.AdaptiveAvgPool2d((1, 1))

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_len, ndf),
            nn.ReLU(),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Linear(ndf * 4, ndf * 2),
            nn.ReLU(),
            nn.Linear(ndf * 2, code_len)
        ])

    def forward(self, input, code):
        conv = self.ConvNet(input)
        conv1 = self.globalPoolingV1(self.ConvNetv1(conv))
        conv2 = self.globalPoolingV2(self.ConvNetv2(conv))
        conv3 = self.globalPoolingV3(self.ConvNetv3(conv))
        flat = torch.cat([conv1, conv2, conv3], dim=1)
        flat_dense = flat.view(flat.size()[:2])
        code_dense = self.code_dense(code)
        global_dense = torch.cat([flat_dense, code_dense], dim=1)
        code_residual = self.global_dense(global_dense)
        return torch.add(code, code_residual)


class CodeAgentV3(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, pooling='avg'):
        super(CodeAgentV3, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ])

        if pooling == 'avg':
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.code_dense = nn.Sequential(*[
            nn.Conv2d(code_len, ndf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, code_len, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ])

    def forward(self, input, code):
        B, C, H, W = input.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))
        conv_input = self.ConvNet(input)
        conv_code = self.code_dense(code_exp)
        conv = torch.cat([conv_input, conv_code], dim=1)
        code_res = self.global_dense(conv)
        flat = self.globalPooling(code_res)
        flat_dense = flat.view(flat.size()[:2])
        return torch.add(code, flat_dense)


class CorrectorV2(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, use_bias=True):
        super(CorrectorV2, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.AttnNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, 1, kernel_size=5, stride=2, padding=2, bias=use_bias),
        ])

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_len, ndf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf, ndf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, code_len, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ])

        self.ndf = ndf
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, input, code, res=False, map=False):
        B, C_l = code.size()
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size()
        attn_map = F.sigmoid(F.upsample(self.AttnNet(input), size=(H_f, W_f), mode='bilinear')).expand(B, C_l, H_f, W_f).contiguous()

        conv_code = self.code_dense(code).view((B, self.ndf, 1, 1)).expand((B, self.ndf, H_f, W_f))
        conv = torch.cat([conv_input, conv_code], dim=1)
        code_res = self.global_dense(conv)

        # Delta_h_p
        flat = self.globalPooling(code_res)
        Delta_h_p = flat.view(flat.size()[:2])

        # Delta_h
        weighted_res = torch.mul(code_res.detach(), attn_map)
        Delta_h = torch.div(torch.sum(weighted_res.view((B, C_l, H_f * W_f)), dim=2), torch.sum(attn_map.view((B, C_l, H_f * W_f)), dim=2))
        if res:
            return (Delta_h, Delta_h_p, attn_map) if map else (Delta_h, Delta_h_p)
        else:
            return (Delta_h + code, Delta_h_p + code, attn_map) if map else (Delta_h + code, Delta_h_p + code)


class Corrector(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, use_bias=True):
        super(Corrector, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_len, ndf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf, ndf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, code_len, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ])

        self.ndf = ndf
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, input, code, res=False):
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size()

        conv_code = self.code_dense(code).view((B, self.ndf, 1, 1)).expand((B, self.ndf, H_f, W_f))
        conv = torch.cat([conv_input, conv_code], dim=1)
        code_res = self.global_dense(conv)

        # Delta_h_p
        flat = self.globalPooling(code_res)
        Delta_h_p = flat.view(flat.size()[:2])

        if res:
            return Delta_h_p
        else:
            return Delta_h_p + code


class CorrectorP(nn.Module):
    def __init__(self, code_len=1, input_nc=1, ndf=64, use_bias=True):
        super(CorrectorP, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Conv2d(ndf + 1, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, code_len, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ])

        self.ndf = ndf
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, input, code, res=False):
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size()

        conv_code = code.view((B, 1, 1, 1)).expand((B, 1, H_f, W_f))
        conv = torch.cat([conv_input, conv_code], dim=1)
        code_res = self.global_dense(conv)

        # Delta_h_p
        flat = self.globalPooling(code_res)
        Delta_h_p = flat.view(flat.size()[:2])

        if res:
            return Delta_h_p
        else:
            return Delta_h_p + code


class CorrectorV4(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, use_bias=True):
        super(CorrectorV4, self).__init__()

        self.ConvNetSR = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.ConvNetLR = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_len, ndf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf, ndf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Conv2d(ndf * 3, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, code_len, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ])

        self.ndf = ndf
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, inputSR, inputLR, code, scala=4, res=False):
        B, C_l = code.size()
        B, C, H, W = inputSR.size()
        inputLR = F.upsample(inputLR, size=(H, W), mode='bilinear')
        conv_input = self.ConvNetSR(inputSR)
        conv_lr = self.ConvNetLR(inputLR)
        B, C_f, H_f, W_f = conv_input.size()

        conv_code = self.code_dense(code).view((B, self.ndf, 1, 1)).expand((B, self.ndf, H_f, W_f))
        conv = torch.cat([conv_input, conv_code, conv_lr], dim=1)
        code_res = self.global_dense(conv)

        # Delta_h_p
        flat = self.globalPooling(code_res)
        Delta_h_p = flat.view(flat.size()[:2])

        if res:
            return Delta_h_p
        else:
            return Delta_h_p + code


class Predictor(nn.Module):
    def __init__(self, code_len=15, input_nc=3, ndf=64, use_bias=True):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, code_len, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view((flat.size()[:2]))


class PredictorP(nn.Module):
    def __init__(self, code_len=1, input_nc=1, ndf=64, use_bias=True):
        super(PredictorP, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, code_len, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])


        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view((flat.size()[:2]))


class SFTMD_CF(nn.Module):
    def __init__(self, input_channel=3, scala=4, input_para=15, min=0.0, max=1.0):
        super(SFTMD_CF, self).__init__()
        self.scala = scala
        self.min = min
        self.max = max

        self.conv_input = nn.Conv2d(in_channels=input_channel + input_para, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        if self.scala == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif self.scala == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64*9, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif self.scala == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=input_channel, kernel_size=9, stride=1, padding=4, bias=False)

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

    def forward(self, x, code, clip=True):
        B, C, H, W = x.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))
        cat_input = torch.cat([x, code_exp], dim=1)
        out = self.relu(self.conv_input(cat_input))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale(out)
        out = self.conv_output(out)
        return torch.clamp(out, min=self.min, max=self.max) if clip else out


class SFT_C_Residual_Block(nn.Module):
    def __init__(self, ndf=64, para=15):
        super(SFT_C_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ndf + para, out_channels=ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, feature_maps, para_maps):
        x = feature_maps
        conc = torch.cat([x, para_maps], dim=1)
        fea1 = F.relu(self.conv1(conc))
        fea2 = F.relu(self.conv2(fea1))
        return torch.add(feature_maps, fea2)


class SFTMD_CI(nn.Module):
    def __init__(self, input_channel=3, input_para=15, scala=4, min=0.0, max=1.0, residuals=16):
        super(SFTMD_CI, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.reses = residuals

        self.conv1 = nn.Conv2d(input_channel + input_para, 64, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        for i in range(residuals):
            self.add_module('SFT-C-residual' + str(i + 1), SFT_C_Residual_Block(ndf=64, para=input_para))

        self.sft_mid = SFT_Layer(ndf=64, para=input_para)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.scala = scala
        if scala == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scala == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64 * 9, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scala == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=input_channel, kernel_size=9, stride=1, padding=4,
                                     bias=False)

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))

        input_cat = torch.cat([input, code_exp], dim=1)
        before_res = self.conv3(
            self.relu_conv2(self.conv2(
                self.relu_conv1(self.conv1(input_cat))
            ))
        )

        res = before_res
        for i in range(self.reses):
            res = self.__getattr__('SFT-C-residual' + str(i + 1))(res, code_exp)

        mid = self.sft_mid(res, code_exp)
        mid = F.relu(mid)
        mid = self.conv_mid(mid)

        befor_up = torch.add(before_res, mid)

        uped = self.upscale(befor_up)

        out = self.conv_output(uped)
        return torch.clamp(out, min=self.min, max=self.max) if clip else out

