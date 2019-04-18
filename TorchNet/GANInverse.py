# pytorch>=1.0
import numpy as np
import random
import functools
from math import ceil
import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from PIL import Image

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

import math

from ..DataTools.Loaders import PIL2VAR, VAR2PIL
from ..DataTools.Prepro import _tanh_to_sigmoid, _sigmoid_to_tanh
from ..DataTools.Loaders import _add_batch_one, _remove_batch
from ..DataTools.FileTools import _image_file
from ..Functions import functional as Func
from .modules import residualBlock, upsampleBlock, DownsamplingShuffle, Attention, Flatten, BatchBlur, b_GaussianNoising, b_GPUVar_Bicubic, b_CPUVar_Bicubic
from .activation import swish
from .ClassicSRNet import SRResNet_Residual_Block


PGGAN_LATENT = [(512, 1, 1),
              (512, 4, 4), (512, 4, 4),
              (512, 8, 8), (512, 8, 8),
              (512, 16, 16), (512, 16, 16),
              (512, 32, 32), (512, 32, 32),
              (256, 64, 64), (256, 64, 64),
              (128, 128, 128), (128, 128, 128),
              (64, 256, 256), (64, 256, 256),
              (32, 512, 512), (32, 512, 512),
              (16, 1024, 1024), (16, 1024, 1024),
              (3, 1024, 1024)]


def PGGAN_parser(layer_number, gan_model):
    # layer_number count from 0
    rest_model = nn.Sequential(*list(gan_model.children())[layer_number:]).cuda()
    before_model = nn.Sequential(*list(gan_model.children())[:layer_number]).cuda()
    input_size = PGGAN_LATENT[layer_number]
    return rest_model, before_model, input_size


class Naive_Inverser(object):
    def __init__(self, model, input_size, output_size):
        self.model = model
        self.model.eval()
        self.in_size = input_size
        self.out_size = output_size

    def __call__(self, gt, iterations=2000, learning_rate=0.01, criterion=nn.MSELoss(reduction='sum'), init=None, intermediate_step=-1, intermediate_stop=1000):
        assert list(gt.size()[1:]) == self.out_size, "check output size"
        batch_size = gt.size()[0]
        if init == None:
            z_estimate = torch.randn((batch_size,) + self.in_size).cuda()  # our estimate, initialized randomly
        else:
            z_estimate = init
        z_estimate.requires_grad = True

        optimizer = optim.Adam([z_estimate], lr=learning_rate)

        # Opt
        z_middle = list()
        for i in range(iterations):
            y_estimate = self.model(z_estimate)
            optimizer.zero_grad()
            loss = criterion(y_estimate, gt.detach())
            if intermediate_step >= 1 and i <= intermediate_stop:
                if i % intermediate_step == 0:
                    z_middle.append(z_estimate.cpu())
                    print("iter {:04d}: y_error = {:03g}".format(i, loss.item()))
            loss.backward()
            optimizer.step()
        return z_estimate, z_middle if intermediate_step >= 1 else z_estimate


class Convex_Sphere_Inverser(Naive_Inverser):
    def __init__(self, model, input_size, output_size):
        super(Convex_Sphere_Inverser, self).__init__(model, input_size, output_size)

    def __ceil__(self, gt, iterations=2000, learning_rate=0.01, criterion=nn.MSELoss(reduction='sum'), intermediate_step=-1, init=None, norm=1):
        assert list(gt.size()[1:]) == self.out_size, "check output size"
        if init == None:
            z_estimate = torch.randn(self.in_size).cuda()  # our estimate, initialized randomly
        else:
            z_estimate = init
        z_estimate.requires_grad = True

        optimizer = optim.Adam([z_estimate], lr=learning_rate)

        # Opt
        for i in range(iterations):
            y_estimate = self.model(z_estimate)
            optimizer.zero_grad()
            loss = criterion(y_estimate, gt.detach())
            if intermediate_step >= 1:
                if i % intermediate_step == 0:
                    print("iter {:04d}: y_error = {:03g}".format(i, loss.item()))
            loss.backward()
            optimizer.step()
            z_estimate = z_estimate / torch.sqrt(torch.sum(torch.pow(z_estimate, 2)))
        return z_estimate


class LBFGS_Inverser(Naive_Inverser):
    def __init__(self, model, input_size, output_size):
        super(LBFGS_Inverser, self).__init__(model, input_size, output_size)

    def __ceil__(self, gt, iterations=1000, learning_rate=0.1, history_size=100, criterion=nn.MSELoss(reduction='sum'), init=None, intermediate_step=-1, intermediate_stop=1000):
        assert list(gt.size()[1:]) == self.out_size, "check output size"
        if init == None:
            z_estimate = torch.randn(self.in_size).cuda()  # our estimate, initialized randomly
        else:
            z_estimate = init
        z_estimate.requires_grad = True

        optimizer = optim.LBFGS([z_estimate], lr=learning_rate, history_size=history_size)

        def closure():
            y_estimate = self.model(z_estimate)
            loss = criterion(y_estimate, gt.detach())
            optimizer.zero_grad()
            loss.backward()
            return loss

        z_middle = list()
        for i in range(iterations):
            y_estimate = self.model(z_estimate)
            optimizer.zero_grad()
            loss = criterion(y_estimate, gt.detach())
            if intermediate_step >= 1 and i <= intermediate_stop:
                if i % intermediate_step == 0:
                    z_middle.append(z_estimate.cpu())
                    print("iter {:04d}: y_error = {:03g}".format(i, loss.item()))
            loss.backward()
            optimizer.step(closure)
        return z_estimate, z_middle if intermediate_step >= 1 else z_estimate


class Encoder_Inverser(object):
    def __init__(self, model, input_size, output_size, encoder):
        self.model = model
        self.model.eval()
        self.encoder = encoder
        self.encoder.eval()
        self.in_size = input_size
        self.out_size = output_size


    def __call__(self, gt, iterations=500, learning_rate=0.01, criterion=nn.MSELoss(reduction='sum'), intermediate_step=-1, intermediate_stop=1000):
        assert list(gt.size()[1:]) == self.out_size, "check output size"
        batch_size = gt.size()[0]

        z_estimate = self.encoder(gt).detach()
        z_estimate.requires_grad = True

        optimizer = optim.Adam([z_estimate], lr=learning_rate)

        # Opt
        z_middle = list()
        for i in range(iterations):
            y_estimate = self.model(z_estimate)
            optimizer.zero_grad()
            loss = criterion(y_estimate, gt.detach())
            if intermediate_step >= 1 and i <= intermediate_stop:
                if i % intermediate_step == 0:
                    z_middle.append(z_estimate.cpu())
                    print("iter {:04d}: y_error = {:03g}".format(i, loss.item()))
            loss.backward()
            optimizer.step()
        return z_estimate, z_middle if intermediate_step >= 1 else z_estimate


class StyleGAN_w_prime_Inverser(object):
    def __init__(self, model, input_size, output_size=1024, layer_number=18):
        self.model = model
        self.model.eval()
        self.in_size = input_size
        self.layer_number = layer_number
        self.out_size = output_size

    def __call__(self, gt, iterations=1000, learning_rate=0.01, criterion=nn.MSELoss(reduction='sum'), init=None, intermediate_step=-1, intermediate_stop=1000):
        assert list(gt.size()[1:]) == self.out_size, "check output size"
        batch_size = gt.size()[0]
        if init == None:
            z_estimate = []
            for i in range(self.layer_number):
                z_estimate.append(torch.randn((batch_size,) + self.in_size).cuda())  # our estimate, initialized randomly
        else:
            z_estimate = init
        for z in z_estimate:
            z.requires_grad = True

        optimizer = optim.Adam(z_estimate, lr=learning_rate)

        # Opt
        z_middle = list()
        for i in range(iterations):
            y_estimate = self.model(z_estimate)
            optimizer.zero_grad()
            loss = criterion(y_estimate, gt.detach())
            if intermediate_step >= 1 and i <= intermediate_stop:
                if i % intermediate_step == 0:
                    z_middle.append(z_estimate.cpu())
                    print("iter {:04d}: y_error = {:03g}".format(i, loss.item()))
            loss.backward()
            optimizer.step()
        return z_estimate, z_middle if intermediate_step >= 1 else z_estimate
