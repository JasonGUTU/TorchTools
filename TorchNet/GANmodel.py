# pytorch>=1.0
# PGGAN
import numpy as np
import random
import functools
from math import ceil
import itertools

import torch
import torch.nn as nn
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


"""Contains the implementation of generator described in progressive GAN.

Different from the official tensorflow model in `pggan_tf_offical`, this is a
simple pytorch version which only contains the generator part. This class is
specially used for inference. However, it can be easily converted from the
official tensorflow model with the provided tool `model_converter.py`.

For more details, please check the original paper:
https://arxiv.org/pdf/1710.10196.pdf
"""

__all__ = ['ProgressiveGANGenerator']

# Defines a dictionary, which maps the target resolution of the final generated
# image to numbers of filters used in each convolutional layer in sequence.
_RESOLUTION_TO_CONV_CHANNELS = {
    8: [512, 512, 512],
    16: [512, 512, 512, 512],
    32: [512, 512, 512, 512, 512],
    64: [512, 512, 512, 512, 512, 256],
    128: [512, 512, 512, 512, 512, 256, 128],
    256: [512, 512, 512, 512, 512, 256, 128, 64],
    512: [512, 512, 512, 512, 512, 256, 128, 64, 32],
    1024: [512, 512, 512, 512, 512, 256, 128, 64, 32, 16],
}


_ACC_LIST_PROGANEncoder = {
            1: -1,
            2: -2,
            3: -3,
            4: -4,
            5: -5,
            6: -6,
            7: -7,
            8: -8,
            9: -9,
            10: -10,
            11: -11,
            12: -12,
            13: -13,
            14: -14,
            15: -15,
            16: -16,
            17: -17,
            18: -18,
            19: -19
        }


class ProgressiveGANGenerator(nn.Sequential):
    """Defines the generator module in progressive GAN.

    Note that the generated images are with RGB color channels.
    """

    def __init__(self, resolution=1024, final_tanh=False):
        """Initializes the generator with basic settings.

        Inputs:
          resolution: The resolution of the final output image.
          final_tanh: Whether to use a `tanh` funtion to clamp the pixel values of
            the output image to range [-1, 1].

        Raises:
          ValueError: If the input `resolution` is not supported.
        """
        try:
          channels = _RESOLUTION_TO_CONV_CHANNELS[resolution]
        except KeyError:
          raise ValueError('Invalid resolution: {}!\nResolutions allowed: {}.'
              .format(resolution, list(_RESOLUTION_TO_CONV_CHANNELS)))

        sequence = OrderedDict()

        def _add_layer(layer, name=None):
          name = name or 'layer{}'.format(len(sequence) + 1)
          sequence[name] = layer

        _add_layer(ConvBlock(channels[0], channels[1], kernel_size=4, padding=3))
        _add_layer(ConvBlock(channels[1], channels[1]))
        for i in range(2, len(channels)):
          _add_layer(ConvBlock(channels[i-1], channels[i], upsample=True))
          _add_layer(ConvBlock(channels[i], channels[i]))
        _add_layer(ConvBlock(in_channels=channels[-1],
                             out_channels=3,
                             kernel_size=1,
                             padding=0,
                             wscale_gain=1.0,
                             activation_type='tanh' if final_tanh else 'linear'),
                   name='output_{}x{}'.format(resolution, resolution))
        super().__init__(sequence)

    def forward(self, x):
        if len(x.shape) != 2:
          raise ValueError('The input tensor should be with shape [batch_size, '
                           'noise_dim], but {} received!'.format(x.shape))
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return super().forward(x)


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ResolutionScalingLayer(nn.Module):
    """Implements the resolution scaling layer.

    Basically, this layer can be used to upsample or downsample feature maps from
    spatial domain with nearest neighbor interpolation.
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x,
                                     scale_factor=self.scale_factor,
                                     mode='nearest')


# class WScaleLayer(nn.Module):
#     """Implements the layer to scale weight variable and add bias.
#
#     Note that, the weight variable is trained in `nn.Conv2d` layer, and only
#     scaled with a constant number, which is not trainable, in this layer. However,
#     the bias variable is trainable in this layer.
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2.0)):
#         super().__init__()
#         fan_in = in_channels * kernel_size * kernel_size
#         self.scale = gain / np.sqrt(fan_in)
#         self.bias = nn.Parameter(torch.randn(out_channels))
#
#     def forward(self, x):
#         return x * self.scale + self.bias.view(1, -1, 1, 1)


class ConvBlock(nn.Module):
    """Implements the convolutional block used in progressive GAN.

    Basically, this block executes pixel-wise normalization layer, upsampling
    layer (if needed), convolutional layer, weight-scale layer and leaky-relu
    layer in sequence.
    """

    def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               upsample=False,
               wscale_gain=np.sqrt(2.0),
               activation_type='lrelu'):
        """Initializes the class with block settings.

        Inputs:
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          stride: Stride parameter for convolution operation.
          padding: Padding parameter for convolution operation.
          dilation: Dilation rate for convolution operation.
          add_bias: Whether to add bias onto the convolutional result.
          upsample: Whether to upsample the input tensor before convolution.
          wscale_gain: The gain factor for wscale layer.
          activation_type: Type of activation function. Support `linear`, `lrelu`
            and `tanh`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()
        self.pixel_norm = PixelNormLayer()
        self.upsample = ResolutionScalingLayer() if upsample else (lambda x: x)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=1,
                              bias=add_bias)
        self.wscale = WScaleLayer(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  gain=wscale_gain)
        if activation_type == 'linear':
          self.activate = (lambda x: x)
        elif activation_type == 'lrelu':
          self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation_type == 'tanh':
          self.activate = nn.Hardtanh()
        else:
          raise NotImplementedError(
              'Not implemented activation function: {}!'.format(activation_type))

    def forward(self, x):
        x = self.pixel_norm(x)
        x = self.upsample(x)
        x = self.conv(x)
        x = self.wscale(x)
        x = self.activate(x)
        return x

# Default path to the face generation model, from which to load parameters.
_MODEL_PATH = '/mnt/lustre/share/shenyujun/models/karras2018iclr-celebahq-1024x1024.pth'

# Maximum batch size to run model.
_MAX_BATCH_SIZE = 4

def convery_array_to_images(np_array):
    """Converts numpy array to images with data type `uint8`.

    This function assumes the input numpy array is with range [-1, 1], as well as
    with shape [batch_size, channel, height, width]. Here, channel = 3 for color
    image and channel = 1 for gray image.

    The return images are with data type `uint8`, lying in range [0, 255]. In
    addition, the return images are with shape [batch_size, height, width,
    channel], and the channel order will be `RGB` for color images.

    Inputs:
    np_array: The numpy array to convert.

    Returns:
    The converted images.

    Raises:
    ValueError: If this input is with wrong shape.
    """
    input_shape = np_array.shape
    if len(input_shape) != 4 or input_shape[1] not in [1, 3]:
        raise ValueError('Input `np_array` should be with shape [batch_size, '
                     'channel, height, width], where channel equals to 1 or 3. '
                    'But {} is received!'.format(input_shape))

    images = (np_array + 1.0) * 127.5
    images = np.clip(images.astype(np.uint8), 0, 255)
    images = images.transpose(0, 2, 3, 1)
    return images


def get_model(model_path=_MODEL_PATH, resolution=1024, use_gpu=True):
    """Gets the generator model in progressive GAN.

    Inputs:
    model_path: This field indicates the path to the well pre-trained model,
      from which to load parameters. If set as empty, no parameters will be
      loaded.
    use_gpu: Whether to run the model with GPU device. Default as True.

    Returns:
    A pytorch model for image generation. See `ProgressiveGANGenerator` in
      `pggan_generator.py`for more details.
    """
    model = ProgressiveGANGenerator(resolution=resolution)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    if use_gpu:
        model.cuda()
    return model


def generate_images(model, latent_vectors=None, num=1, use_gpu=True):
    """Generates images with progressive GAN model.

    This function outputs images based on the input latent vectors. The input
    latent vectors should be with shape [batch_size, latent_space_dim] or
    [latent_space_dim], where `latent_space_dim` should match the dimension of the
    input vector described in the given `model`. If no `latent_vectors` is fed
    into this function, it will randomly sample some noises, subject to N(0, 1)
    distribution, as inputs.

    Note that `num` can be larger than _MAX_BATCH_SIZE, but can not be too large,
    since each generated image is with extremely high resolution. Caching too many
    images may cause memory exhausted.

    Inputs:
    model: Model for image generation. Please use `get_model` function to load
      pre-trained model.
    latent_vectors: The latent vectors fed into the generation model.
    num: The number of images to generate. This field only has effect when
      `latent_vectors` is None.
    use_gpu: Whether to use GPU device. Default as True.

    Returns:
    A collection of generated images, with shape [batch_size, height, width,
      channel].

    Raises:
    ValueError: If the input pair (latent_vectors, num) is invalid.
    """
    latent_space_dim = model.layer1.conv.weight.shape[0]
    if latent_vectors is not None:
        input_shape = latent_vectors.shape
        if len(input_shape) == 1:
            latent_vectors = np.expand_dims(latent_vectors, axis=0)
        new_shape = latent_vectors.shape
        if len(new_shape) != 2 or new_shape[1] != latent_space_dim:
            raise ValueError('Input `latent_vectors` should be with shape '
                           '[batch_size, latent_space_dim] or [latent_space_dim], '
                           'where `latent_space_dim` equals to {}. But {} is '
                           'received!'.format(latent_space_dim, new_shape))
    else:
        if num <= 0 or not isinstance(num, int):
            raise ValueError('Input `num` should be a positive integer when '
                           '`latent_vectors` is set as None, '
                           'but {} is received!'.format(num))
        latent_vectors = np.random.randn(num, latent_space_dim).astype(np.float32)

    images = []
    num = latent_vectors.shape[0]
    finished_num = 0
    while finished_num != num:
        batch_size = min(num - finished_num, _MAX_BATCH_SIZE)
        x = latent_vectors[finished_num : finished_num + batch_size, :]
        x = torch.from_numpy(x).type(torch.FloatTensor)
        if use_gpu:
            x = x.cuda()
        model.eval()
        y = model(x)
        if use_gpu:
            y = y.cpu()
        images.append(y.detach().numpy())
        torch.cuda.empty_cache()
        finished_num += batch_size
        print('Generated %6d images.' % (finished_num))


def print_network(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('Total number of parameters: {:3.3f} M'.format(num_params / 1e6))


def from_pth_file(filename):
    '''
    Instantiate from a pth file.
    '''
    state_dict = torch.load(filename)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # Convert old version of parameter names
    if 'features.0.conv.weight' in state_dict:
        state_dict = state_dict_from_old_pt_dict(state_dict)
    sizes = sizes_from_state_dict(state_dict)
    result = ProgressiveGenerator(sizes=sizes)
    result.load_state_dict(state_dict)
    return result

###############################################################################
# Modules
###############################################################################

class ProgressiveGenerator(nn.Sequential):
    def __init__(self, resolution=None, sizes=None, modify_sequence=None,
            output_tanh=False):
        '''
        A pytorch progessive GAN generator that can be converted directly
        from either a tensorflow model or a theano model.  It consists of
        a sequence of convolutional layers, organized in pairs, with an
        upsampling and reduction of channels at every other layer; and
        then finally followed by an output layer that reduces it to an
        RGB [-1..1] image.

        The network can be given more layers to increase the output
        resolution.  The sizes argument indicates the fieature depth at
        each upsampling, starting with the input z: [input-dim, 4x4-depth,
        8x8-depth, 16x16-depth...].  The output dimension is 2 * 2**len(sizes)

        Some default architectures can be selected by supplying the
        resolution argument instead.

        The optional modify_sequence function can be used to transform the
        sequence of layers before the network is constructed.

        If output_tanh is set to True, the network applies a tanh to clamp
        the output to [-1,1] before output; otherwise the output is unclamped.
        '''
        assert (resolution is None) != (sizes is None)
        if sizes is None:
            sizes = {
                    8: [512, 512, 512],
                    16: [512, 512, 512, 512],
                    32: [512, 512, 512, 512, 256],
                    64: [512, 512, 512, 512, 256, 128],
                    128: [512, 512, 512, 512, 256, 128, 64],
                    256: [512, 512, 512, 512, 256, 128, 64, 32],
                    1024: [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
                }[resolution]
        # Follow the schedule of upsampling given by sizes.
        # layers are called: layer1, layer2, etc; then output_128x128
        sequence = []
        def add_d(layer, name=None):
            if name is None:
                name = 'layer%d' % (len(sequence) + 1)
            sequence.append((name, layer))
        add_d(NormConvBlock(sizes[0], sizes[1], kernel_size=4, padding=3))
        add_d(NormConvBlock(sizes[1], sizes[1], kernel_size=3, padding=1))
        for i, (si, so) in enumerate(zip(sizes[1:-1], sizes[2:])):
            add_d(NormUpscaleConvBlock(si, so, kernel_size=3, padding=1))
            add_d(NormConvBlock(so, so, kernel_size=3, padding=1))
        # Create an output layer.  During training, the progressive GAN
        # learns several such output layers for various resolutions; we
        # just include the last (highest resolution) one.
        dim = 4 * (2 ** (len(sequence) // 2 - 1))
        add_d(OutputConvBlock(sizes[-1], tanh=output_tanh),
                name='output_%dx%d' % (dim, dim))
        # Allow the sequence to be modified
        if modify_sequence is not None:
            sequence = modify_sequence(sequence)
        super().__init__(OrderedDict(sequence))

    def forward(self, x):
        # Convert vector input to 1x1 featuremap.
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return super().forward(x)


class ProgressiveGeneratorEncoder(nn.Sequential):
    def __init__(self, resolution=None, sizes=None, modify_sequence=None,
            output_tanh=False):
        '''
        A pytorch progessive GAN generator that can be converted directly
        from either a tensorflow model or a theano model.  It consists of
        a sequence of convolutional layers, organized in pairs, with an
        upsampling and reduction of channels at every other layer; and
        then finally followed by an output layer that reduces it to an
        RGB [-1..1] image.

        The network can be given more layers to increase the output
        resolution.  The sizes argument indicates the fieature depth at
        each upsampling, starting with the input z: [input-dim, 4x4-depth,
        8x8-depth, 16x16-depth...].  The output dimension is 2 * 2**len(sizes)

        Some default architectures can be selected by supplying the
        resolution argument instead.

        The optional modify_sequence function can be used to transform the
        sequence of layers before the network is constructed.

        If output_tanh is set to True, the network applies a tanh to clamp
        the output to [-1,1] before output; otherwise the output is unclamped.
        '''
        assert (resolution is None) != (sizes is None)
        if sizes is None:
            sizes = {
                    8: [512, 512, 512],
                    16: [512, 512, 512, 512],
                    32: [512, 512, 512, 512, 256],
                    64: [512, 512, 512, 512, 256, 128],
                    128: [512, 512, 512, 512, 256, 128, 64],
                    256: [512, 512, 512, 512, 256, 128, 64, 32],
                    1024: [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
                }[resolution]
        sizes += [3]
        sizes.reverse()
        # Follow the schedule of upsampling given by sizes.
        # layers are called: layer1, layer2, etc; then output_128x128
        sequence = []
        def add_d(layer, name=None):
            if name is None:
                name = 'layer%d' % (len(sequence) + 1)
            sequence.append((name, layer))

        add_d(NormConvBlock(sizes[0], sizes[1], kernel_size=3, padding=1))
        add_d(NormConvBlock(sizes[1], sizes[1], kernel_size=3, padding=1))
        for i, (si, so) in enumerate(zip(sizes[1:-2], sizes[2:-1])):
            add_d(NormDownConvBlock(si, so, kernel_size=3, padding=1))
            add_d(NormConvBlock(so, so, kernel_size=3, padding=1))
        # Create an output layer.  During training, the progressive GAN
        # learns several such output layers for various resolutions; we
        # just include the last (highest resolution) one.
#         dim = 4 * (2 ** (len(sequence) // 2 - 1))
#         add_d(OutputConvBlock(sizes[-1], tanh=output_tanh),
#                 name='output_%dx%d' % (dim, dim))
        # Allow the sequence to be modified
        add_d(NormConvBlock(sizes[-2], sizes[-1], kernel_size=4, padding=0))
        if modify_sequence is not None:
            sequence = modify_sequence(sequence)
        super().__init__(OrderedDict(sequence))

    def forward(self, x):
        # Convert vector input to 1x1 featuremap.
#         x = x.view(x.shape[0], x.shape[1], 1, 1)
        return super().forward(x)


class DoubleResolutionLayer(nn.Module):
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

class HalfResolutionLayer(nn.Module):
    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2)
        return x


class WScaleLayer(nn.Module):
    def __init__(self, size, fan_in, gain=np.sqrt(2)):
        super(WScaleLayer, self).__init__()
        self.scale = gain / np.sqrt(fan_in) # No longer a parameter
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size, x_size[2], x_size[3])
        return x


class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels,
                gain=np.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class NormUpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = DoubleResolutionLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels,
                gain=np.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class NormDownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormDownConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.down = HalfResolutionLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels,
                gain=np.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.down(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class OutputConvBlock(nn.Module):
    def __init__(self, in_channels, tanh=False):
        super().__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
                in_channels, 3, kernel_size=1, padding=0, bias=False)
        self.wscale = WScaleLayer(3, in_channels, gain=1)
        self.clamp = nn.Hardtanh() if tanh else (lambda x: x)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.wscale(x)
        x = self.clamp(x)
        return x

###############################################################################
# Conversion
###############################################################################

def from_tf_parameters(parameters):
    '''
    Instantiate from tensorflow variables.
    '''
    state_dict = state_dict_from_tf_parameters(parameters)
    sizes = sizes_from_state_dict(state_dict)
    result = ProgressiveGenerator(sizes=sizes)
    result.load_state_dict(state_dict)
    return result


def from_old_pt_dict(parameters):
    '''
    Instantiate from old pytorch state dict.
    '''
    state_dict = state_dict_from_old_pt_dict(parameters)
    sizes = sizes_from_state_dict(state_dict)
    result = ProgressiveGenerator(sizes=sizes)
    result.load_state_dict(state_dict)
    return result


def sizes_from_state_dict(params):
    '''
    In a progressive GAN, the number of channels can change after each
    upsampling.  This function reads the state dict to figure the
    number of upsamplings and the channel depth of each filter.
    '''
    sizes = []
    for i in itertools.count():
        pt_layername = 'layer%d' % (i + 1)
        try:
            weight = params['%s.conv.weight' % pt_layername]
        except KeyError:
            break
        if i == 0:
            sizes.append(weight.shape[1])
        if i % 2 == 0:
            sizes.append(weight.shape[0])
    return sizes


def state_dict_from_tf_parameters(parameters):
    '''
    Conversion from tensorflow parameters
    '''
    params = dict(parameters)
    result = {}
    sizes = []
    for i in itertools.count():
        resolution = 4 * (2 ** (i // 2))
        # Translate parameter names.  For example:
        # 4x4/Dense/weight -> layer1.conv.weight
        # 32x32/Conv0_up/weight -> layer7.conv.weight
        # 32x32/Conv1/weight -> layer8.conv.weight
        tf_layername = '%dx%d/%s' % (resolution, resolution,
                'Dense' if i == 0 else 'Conv' if i == 1 else
                'Conv0_up' if i % 2 == 0 else 'Conv1')
        pt_layername = 'layer%d' % (i + 1)
        # Stop looping when we run out of parameters.
        try:
            weight = torch.from_numpy(params['%s/weight' % tf_layername])
        except KeyError:
            break
        # Transpose convolution weights into pytorch format.
        if i == 0:
            # Convert dense layer to 4x4 convolution
            weight = weight.view(weight.shape[0], weight.shape[1] // 16,
                   4, 4).permute(1, 0, 2, 3).flip(2, 3)
            sizes.append(weight.shape[0])
        elif i % 2 == 0:
            # Convert inverse convolution to convolution
            weight = weight.permute(2, 3, 0, 1).flip(2, 3)
        else:
            # Ordinary Conv2d conversion.
            weight = weight.permute(3, 2, 0, 1)
            sizes.append(weight.shape[1])
        result['%s.conv.weight' % (pt_layername)] = weight
        # Copy bias vector.
        bias = torch.from_numpy(params['%s/bias' % tf_layername])
        result['%s.wscale.b' % (pt_layername)] = bias
    # Copy just finest-grained ToRGB output layers.  For example:
    # ToRGB_lod0/weight -> output.conv.weight
    i -= 1
    resolution = 4 * (2 ** (i // 2))
    tf_layername = 'ToRGB_lod0'
    pt_layername = 'output_%dx%d' % (resolution, resolution)
    result['%s.conv.weight' % pt_layername] = torch.from_numpy(
            params['%s/weight' % tf_layername]).permute(3, 2, 0, 1)
    result['%s.wscale.b' % pt_layername] = torch.from_numpy(
            params['%s/bias' % tf_layername])
    # Return parameters
    return result


def state_dict_from_old_pt_dict(params):
    '''
    Conversion from the old pytorch model layer names.
    '''
    result = {}
    sizes = []
    for i in itertools.count():
        old_layername = 'features.%d' % i
        pt_layername = 'layer%d' % (i + 1)
        try:
            weight = params['%s.conv.weight' % (old_layername)]
        except KeyError:
            break
        if i == 0:
            sizes.append(weight.shape[0])
        if i % 2 == 0:
            sizes.append(weight.shape[1])
        result['%s.conv.weight' % (pt_layername)] = weight
        result['%s.wscale.b' % (pt_layername)] = params[
                '%s.wscale.b' % (old_layername)]
    # Copy the output layers.
    i -= 1
    resolution = 4 * (2 ** (i // 2))
    pt_layername = 'output_%dx%d' % (resolution, resolution)
    result['%s.conv.weight' % pt_layername] = params['output.conv.weight']
    result['%s.wscale.b' % pt_layername] = params['output.wscale.b']
    # Return parameters and also network architecture sizes.
    return result


def z_dataset_for_model(model, size=100, seed=1):
    return TensorDataset(z_sample_for_model(model, size, seed))


def z_sample_for_model(model, size=100, seed=1):
    # If the model is marked with an input shape, use it.
    if hasattr(model, 'input_shape'):
        sample = standard_z_sample(size, model.input_shape[1], seed=seed).view(
                (size,) + model.input_shape[1:])
        return sample
    # Examine first conv in model to determine input feature size.
    first_layer = [c for c in model.modules()
            if isinstance(c, (torch.nn.Conv2d, torch.nn.ConvTranspose2d,
                torch.nn.Linear))][0]
    # 4d input if convolutional, 2d input if first layer is linear.
    if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        sample = standard_z_sample(
                size, first_layer.in_channels, seed=seed)[:,:,None,None]
    else:
        sample = standard_z_sample(
                size, first_layer.in_features, seed=seed)
    return sample


def standard_z_sample(size, depth, seed=1, device=None):
    '''
    Generate a standard set of random Z as a (size, z_dimension) tensor.
    With the same random seed, it always returns the same z (e.g.,
    the first one is always the same regardless of the size.)
    '''
    # Use numpy RandomState since it can be done deterministically
    # without affecting global state
    rng = np.random.RandomState(seed)
    result = torch.from_numpy(rng.standard_normal(size * depth).reshape(size, depth)).float()
    if device is not None:
        result = result.to(device)
    return result


