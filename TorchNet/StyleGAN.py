# python 3.6
"""Contains the implementation of generator described in style GAN.

Different from the official tensorflow model in `stylegan_tf_offical`, this is a
simple pytorch version which only contains the generator part. This class is
specially used for inference. However, it can be easily converted from the
official tensorflow model with the provided tool `stylegan_converter.py`.

For more details, please check the original paper:
https://arxiv.org/pdf/1812.04948.pdf
"""

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StyleGANGenerator']

# Defines a dictionary, which maps the target resolution of the final generated
# image to numbers of filters used in each convolutional layer in sequence.
_RESOLUTIONS_TO_CHANNELS = {
    8: [512, 512, 512],
    16: [512, 512, 512, 512],
    32: [512, 512, 512, 512, 512],
    64: [512, 512, 512, 512, 512, 256],
    128: [512, 512, 512, 512, 512, 256, 128],
    256: [512, 512, 512, 512, 512, 256, 128, 64],
    512: [512, 512, 512, 512, 512, 256, 128, 64, 32],
    1024: [512, 512, 512, 512, 512, 256, 128, 64, 32, 16],
}


class StyleGANGenerator(nn.Module):
    """Defines the generator module in style GAN.

    Note that the generated images are with RGB color channels.
    """

    def __init__(self,
               resolution=1024,
               w_dim=512,
               w_truncation_psi=0.7,
               w_truncation_layers=8,
               randomize_noise=False,
               start='z'):
        """Initializes the generator with basic settings.

        Inputs:
          resolution: The resolution of the final output image.
          w_dim: The dimension of the disentangled latent vectors, w.
          w_truncation_psi: Style strength multiplier for the truncation trick.
            `None` indicates no truncation.
          w_truncation_layers: Number of layers for which to apply the truncation
            trick. `None` indicates no truncation.

        Raises:
          ValueError: If the input `resolution` is not supported.
        """
        super().__init__()

        try:
            channels = _RESOLUTIONS_TO_CHANNELS[resolution]
        except KeyError:
            raise ValueError(f'Invalid resolution: {resolution}!\n'
                           f'Resolutions allowed: '
                           f'{list(_RESOLUTIONS_TO_CHANNELS)}.')

        self.start = start
        self.mapping = MappingBlock(final_space_dim=w_dim)
        self.num_layers = int(np.log2(resolution)) * 2 - 2
        self.w_dim = w_dim
        if w_truncation_psi and w_truncation_layers:
            self.use_truncation = True
        else:
            self.use_truncation = False
            w_truncation_psi = 1
            w_truncation_layers = 0
        self.register_buffer('w_avg', torch.zeros(w_dim))
        layer_idx = np.arange(self.num_layers).reshape(1, self.num_layers, 1)
        coefs = np.ones_like(layer_idx, dtype=np.float32)
        coefs[layer_idx < w_truncation_layers] *= w_truncation_psi
        self.register_buffer('truncation', torch.from_numpy(coefs))

        for i in range(1, len(channels)):
            if i == 1:
                self.add_module('conv0', FirstConvBlock(channels[0], randomize_noise))
            else:
                self.add_module(
                f'conv{i * 2 - 2}',
                UpConvBlock(layer_idx=i * 2 - 2,
                            in_channels=channels[i - 1],
                            out_channels=channels[i],
                            randomize_noise=randomize_noise))
            self.add_module(
                f'conv{i * 2 - 1}',
                ConvBlock(layer_idx=i * 2 - 1,
                        in_channels=channels[i],
                        out_channels=channels[i],
                        randomize_noise=randomize_noise))
        self.add_module('output', LastConvBlock(channels[-1]))

    def forward(self, input):
        if self.start == 'z':
            w_ = self.mapping(input)
            w = w_.view(-1, 1, self.w_dim).repeat(1, self.num_layers, 1)
            # w = w_.view(-1, 1, self.w_dim).expand([-1, self.num_layers, self.w_dim])
            if self.use_truncation:
                w_avg = self.w_avg.view(1, 1, self.w_dim)
                w = w_avg + (w - w_avg) * self.truncation
        elif self.start == 'w':
            w_ = input
            w = w_.view(-1, 1, self.w_dim).repeat(1, self.num_layers, 1)
            # w = w_.view(-1, 1, self.w_dim).expand([-1, self.num_layers, self.w_dim])
            if self.use_truncation:
                w_avg = self.w_avg.view(1, 1, self.w_dim)
                w = w_avg + (w - w_avg) * self.truncation
        elif self.start == 'w+':
            w = input
        else:
            w_ = self.mapping(input)
            w = w_.view(-1, 1, self.w_dim).repeat(1, self.num_layers, 1)
            # w = w_.view(-1, 1, self.w_dim).expand([-1, self.num_layers, self.w_dim])
            if self.use_truncation:
                w_avg = self.w_avg.view(1, 1, self.w_dim)
                w = w_avg + (w - w_avg) * self.truncation
        # w = w.view(-1, 1, self.w_dim).repeat(1, self.num_layers, 1)
        # if self.use_truncation:
        #     w_avg = self.w_avg.view(1, 1, self.w_dim)
        #     w = w_avg + (w - w_avg) * self.truncation
        w_list = torch.split(w, 1, dim=1)

        x = self.conv0(w_list[0].view((-1, self.w_dim)))
        for i in range(1, self.num_layers):
            x = self.__getattr__(f'conv{i}')(x, w_list[i].view((-1, self.w_dim)))
        x = self.output(x)

        return x


class MappingBlock(nn.Sequential):
    """Implements the latent space mapping block used in style GAN.

    Basically, this block executes several dense layers in sequence.
    """

    def __init__(self,
                   resolution=1024,
                   normalize_input=True,
                   input_space_dim=512,
                   hidden_space_dim=512,
                   final_space_dim=512):
        sequence = OrderedDict()

        def _add_layer(layer, name=None):
            name = name or f'dense{len(sequence) + (not normalize_input) - 1}'
            sequence[name] = layer

        if normalize_input:
            _add_layer(PixelNormLayer(), name='normalize')
        res = resolution
        while res / 2 >= 4:
            in_dim = input_space_dim if res == resolution else hidden_space_dim
            out_dim = hidden_space_dim if resolution > 8 else final_space_dim
            _add_layer(DenseBlock(in_dim, out_dim))
            res //= 2
        super().__init__(sequence)


    def forward(self, x):
        if len(x.shape) != 2:
            raise ValueError(f'The input tensor should be with shape [batch_size, '
                           f'noise_dim], but {x.shape} received!')
        return super().forward(x)


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(f'The input tensor should be with shape [batch_size, '
                           f'num_channels, height, width], but {x.shape} received!')
        x = x - torch.mean(x, dim=[2, 3], keepdim=True)
        x = x / torch.sqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) + self.epsilon)
        return x


class ResolutionScalingLayer(nn.Module):
    """Implements the resolution scaling layer.

    Basically, this layer can be used to upsample or downsample feature maps from
    spatial domain with nearest neighbor interpolation.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class BlurLayer(nn.Module):
    """Implements the blur layer used in style GAN."""

    def __init__(self,
                   channels,
                   kernel=[1, 2, 1],
                   normalize=True,
                   flip=False,
                   stride=1):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, 3)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        if flip:
            kernel = kernel[::-1, ::-1]
        kernel = kernel.reshape(3, 3, 1, 1)
        kernel = np.tile(kernel, [1, 1, channels, 1])
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)


class NoiseApplyingLayer(nn.Module):
    """Implements the noise applying layer used in style GAN."""

    def __init__(self, layer_idx, channels, randomize_noise=False):
        super().__init__()
        self.randomize_noise = randomize_noise
        self.res = 2**(layer_idx // 2 + 2)
        self.register_buffer('noise', torch.randn(1, 1, self.res, self.res))
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(f'The input tensor should be with shape [batch_size, '
                           f'num_channels, height, width], but {x.shape} received!')
        if self.randomize_noise:
            noise = torch.randn(x.shape[0], 1, self.res, self.res)
            if x.is_cuda:
                noise = noise.cuda()
        else:
            noise = self.noise
        return x + noise * self.weight.view(1, -1, 1, 1)


class StyleModulationLayer(nn.Module):
    """Implements the style modulation layer used in style GAN."""

    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.channels = channels
        self.dense = DenseBlock(in_features=w_dim,
                                out_features=channels*2,
                                wscale_gain=1.0,
                                wscale_lr_multiplier=1.0,
                                activation_type='linear')

    def forward(self, x, w):
        if len(w.shape) != 2:
            raise ValueError(f'The input tensor should be with shape [batch_size, '
                           f'num_channels], but {x.shape} received!')
        style = self.dense(w)
        style = style.view(-1, 2, self.channels, 1, 1)
        # style_list = torch.split(style, 1, dim=1)
        # return x * (style_list[0] + 1) + style_list[1]
        return x * (style[:, 0] + 1) + style[:, 1]


class WScaleLayer(nn.Module):
    """Implements the layer to scale weight variable and add bias.

    Note that, the weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
    layer), and only scaled with a constant number , which is not trainable, in
    this layer. However, the bias variable is trainable in this layer.
    """
    def __init__(self,
                   in_channels,
                   out_channels,
                   kernel_size,
                   gain=np.sqrt(2.0),
                   lr_multiplier=1.0):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gain / np.sqrt(fan_in) * lr_multiplier
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.lr_multiplier = lr_multiplier

    def forward(self, x):
        if len(x.shape) == 4:
            return x * self.scale + self.bias.view(1, -1, 1, 1) * self.lr_multiplier
        elif len(x.shape) == 2:
            return x * self.scale + self.bias.view(1, -1) * self.lr_multiplier
        else:
            raise ValueError(f'The input tensor should be with shape [batch_size, '
                           f'num_channels, height, width], or [batch_size, '
                           f'num_channels], but {x.shape} received!')


class EpilogueBlock(nn.Module):
    """Implements the epilogue block of each conv block."""

    def __init__(self,
                   layer_idx,
                   channels,
                   randomize_noise=False,
                   normalization_fn='instance'):
        super().__init__()
        self.apply_noise = NoiseApplyingLayer(layer_idx, channels, randomize_noise)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if normalization_fn == 'pixel':
            self.norm = PixelNormLayer()
        elif normalization_fn == 'instance':
            self.norm = InstanceNormLayer()
        else:
            raise NotImplementedError(f'Not implemented normalization function: '
                                    f'{normalization_fn}!')
        self.style_mod = StyleModulationLayer(channels)

    def forward(self, x ,w):
        x1 = self.apply_noise(x)
        x2 = x1 + self.bias.view(1, -1, 1, 1)
        x3 = self.activate(x2)
        x4 = self.norm(x3)
        x5 = self.style_mod(x4, w)
        return x5


class FirstConvBlock(nn.Module):
    """Implements the first convolutional block used in style GAN.

    Basically, this block starts from a const input, which is `ones(512, 4, 4)`.
    """

    def __init__(self, channels, randomize_noise=False):
        super().__init__()
        self.first_layer = nn.Parameter(torch.ones(1, channels, 4, 4))
        self.epilogue = EpilogueBlock(layer_idx=0,
                                      channels=channels,
                                      randomize_noise=randomize_noise)

    def forward(self, w):
        x = self.first_layer.repeat(w.shape[0], 1, 1, 1)
        x = self.epilogue(x, w)
        return x


class UpConvBlock(nn.Module):
    """Implements the convolutional block used in progressive GAN.

    Basically, this block is used as the first convolutional block for each
    resolution, which will execute upsampling.
    """

    def __init__(self,
                   layer_idx,
                   in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   add_bias=False,
                   wscale_gain=np.sqrt(2.0),
                   wscale_lr_multiplier=1.0,
                   randomize_noise=False):
        """Initializes the class with block settings.

        Inputs:
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          stride: Stride parameter for convolution operation.
          padding: Padding parameter for convolution operation.
          dilation: Dilation rate for convolution operation.
          add_bias: Whether to add bias onto the convolutional result.
          wscale_gain: The gain factor for `wscale` layer.
          wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
            layer.

        Raises:
          ValueError: If the block is not applied to the first block for a
            particular resolution.
        """
        super().__init__()
        if layer_idx % 2 == 1:
            raise ValueError(f'This block is implemented as the first block of each '
                           f'resolution, but is applied to layer {layer_idx}!')

        self.layer_idx = layer_idx

        if self.layer_idx > 9:
            self.weight = nn.Parameter(
              torch.randn(kernel_size, kernel_size, in_channels, out_channels))

        else:
            self.upsample = ResolutionScalingLayer()
            self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=1,
                               bias=add_bias)

        fan_in = in_channels * kernel_size * kernel_size
        self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
        self.blur = BlurLayer(channels=out_channels)
        self.epilogue = EpilogueBlock(layer_idx=layer_idx,
                                      channels=out_channels,
                                      randomize_noise=randomize_noise)

    def forward(self, x, w):
        if self.layer_idx > 9:
            kernel = self.weight * self.scale
            kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
            kernel = (kernel[1:, 1:] + kernel[:-1, 1:] +
                    kernel[1:, :-1] + kernel[:-1, :-1])
            kernel = kernel.permute(2, 3, 0, 1)
            x = F.conv_transpose2d(x, kernel, stride=2, padding=1)
        else:
            x = self.upsample(x)
            x = self.conv(x) * self.scale
        x = self.blur(x)
        x = self.epilogue(x, w)
        return x


class ConvBlock(nn.Module):
    """Implements the convolutional block used in progressive GAN.

    Basically, this block is used as the second convolutional block for each
    resolution.
    """
    def __init__(self,
                   layer_idx,
                   in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   add_bias=False,
                   wscale_gain=np.sqrt(2.0),
                   wscale_lr_multiplier=1.0,
                   randomize_noise=False):
        """Initializes the class with block settings.

        Inputs:
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          stride: Stride parameter for convolution operation.
          padding: Padding parameter for convolution operation.
          dilation: Dilation rate for convolution operation.
          add_bias: Whether to add bias onto the convolutional result.
          wscale_gain: The gain factor for `wscale` layer.
          wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
            layer.

        Raises:
          ValueError: If the block is not applied to the second block for a
            particular resolution.
        """
        super().__init__()
        if layer_idx % 2 == 0:
            raise ValueError(f'This block is implemented as the second block of each '
                           f'resolution, but is applied to layer {layer_idx}!')

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=1,
                              bias=add_bias)
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
        self.epilogue = EpilogueBlock(layer_idx=layer_idx,
                                      channels=out_channels,
                                      randomize_noise=randomize_noise)

    def forward(self, x, w):
        x = self.conv(x) * self.scale
        x = self.epilogue(x, w)
        return x


class LastConvBlock(nn.Module):
    """Implements the last convolutional block used in style GAN.

    Basically, this block converts the final feature map to RGB image.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=3,
                              kernel_size=1,
                              bias=False)
        self.scale = 1 / np.sqrt(channels)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        x = self.conv(x) * self.scale
        x = x + self.bias.view(1, -1, 1, 1)
        return x


class DenseBlock(nn.Module):
    """Implements the dense block used in style GAN.

    Basically, this block executes fully-connected layer, weight-scale layer,
    and activation layer in sequence.
    """

    def __init__(self,
                   in_features,
                   out_features,
                   add_bias=False,
                   wscale_gain=np.sqrt(2.0),
                   wscale_lr_multiplier=0.01,
                   activation_type='lrelu'):
        """Initializes the class with block settings.

        Inputs:
          in_features: Number of channels of the input tensor fed into this block.
          out_features: Number of channels of the output tensor.
          add_bias: Whether to add bias onto the fully-connected result.
          wscale_gain: The gain factor for `wscale` layer.
          wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
            layer.
          activation_type: Type of activation function. Support `linear` and
            `lrelu`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()
        self.linear = nn.Linear(in_features=in_features,
                              out_features=out_features,
                              bias=add_bias)
        self.wscale = WScaleLayer(in_channels=in_features,
                                  out_channels=out_features,
                                  kernel_size=1,
                                  gain=wscale_gain,
                                  lr_multiplier=wscale_lr_multiplier)
        if activation_type == 'linear':
            self.activate = (lambda x: x)
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                    f'{activation_type}!')

    def forward(self, x):
        x = self.linear(x)
        x = self.wscale(x)
        x = self.activate(x)
        return x

