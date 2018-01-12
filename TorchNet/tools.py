import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_parameters(model):
    parameters = 0
    for weight in model.parameters():
        p = 1
        for dim in weight.size():
            p *= dim
        parameters += p
    return parameters







