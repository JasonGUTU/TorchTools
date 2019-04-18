from math import sqrt, ceil
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import PIL
from PIL import Image

from ..Functions.functional import to_pil_image
from collections import OrderedDict


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


def calculate_parameters(model):
    parameters = 0
    for weight in model.parameters():
        p = 1
        for dim in weight.size():
            p *= dim
        parameters += p
    return parameters


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def FeatureMapsVisualization(feature_maps, instan_norm=False):
    """
    visualize feature maps
    :param feature_maps: must be 4D tensor with B equals to 1 or 3D tensor N * H * W
    :return: PIL.Image of feature maps
    """
    if len(feature_maps.size()) == 4:
        feature_maps = feature_maps.view(feature_maps.size()[1:])
    if not instan_norm:
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())
    maps_number = feature_maps.size()[0]
    feature_H = feature_maps.size()[1]
    feature_W = feature_maps.size()[2]
    W_n = ceil(sqrt(maps_number))
    H_n = ceil(maps_number / W_n)
    map_W = W_n * feature_W
    map_H = H_n * feature_H
    MAP = Image.new('L', (map_W, map_H))
    for i in range(maps_number):
        map_t = feature_maps[i]
        if instan_norm:
            map_t = (map_t - map_t.min()) / (map_t.max() - map_t.min())
        map_t = map_t.view((1, ) + map_t.size())
        map_pil = to_pil_image(map_t)
        n_row = i % W_n
        n_col = i // W_n
        MAP.paste(map_pil, (n_row * feature_W, n_col * feature_H))
    return MAP


def ModelToSequential(model, seq_output=True):
    Sequential_list = list()
    for sub in model.children():
        if isinstance(sub, torch.nn.modules.container.Sequential):
            Sequential_list.extend(ModelToSequential(sub, seq_output=False))
        else:
            Sequential_list.append(sub)
    if seq_output:
        return nn.Sequential(*Sequential_list)
    else:
        return Sequential_list


# def KernelsVisualization(kernels, instan_norm=False):
#     """
#     visualize feature maps
#     :param feature_maps: must be 4D tensor
#     :return: PIL.Image of feature maps
#     """
#     if not instan_norm:
#         feature_maps = (kernels - kernels.min()) / (kernels.max() - kernels.min())
#     kernels_out = kernels.size()[0]
#     kernels_in = kernels.size()[1]
#     feature_H = kernels.size()[2]
#     feature_W = kernels.size()[3]
#     W_n = ceil(sqrt(kernels_in))
#     H_n = ceil(kernels_in / W_n)
#     big_W_n = ceil(sqrt(kernels_out))
#     big_H_n = ceil(kernels_out / W_n)
#     map_W = W_n * feature_W
#     map_H = H_n * feature_H
#     MAP = Image.new('L', (map_W, map_H))
#     for i in range(maps_number):
#         map_t = feature_maps[i]
#         if instan_norm:
#             map_t = (map_t - map_t.min()) / (map_t.max() - map_t.min())
#         map_t = map_t.view((1, ) + map_t.size())
#         map_pil = to_pil_image(map_t)
#         n_row = i % W_n
#         n_col = i // W_n
#         MAP.paste(map_pil, (n_row * feature_W, n_col * feature_H))
#     return MAP

def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary



