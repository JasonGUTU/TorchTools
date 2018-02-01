import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MCTHuberLoss(nn.Module):
    """
    The Huber Loss used in MCT
    """
    def __init__(self, hpyer_lambda, epsilon=0.01):
        super(MCTHuberLoss, self).__init__()
        self.epsilon = epsilon
        self.lamb = hpyer_lambda
        self.sobel = nn.Conv2d(2, 4, 3, stride=1, padding=0, groups=2)
        weight = np.array([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                  [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]],
                  [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                  [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]], dtype=np.float32)
        bias = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.sobel.weight.data = torch.from_numpy(weight)
        self.sobel.bias.data = torch.from_numpy(bias)

    def forward(self, flows):
        Grad_Flow = self.sobel(flows)
        return torch.sqrt(torch.sum(Grad_Flow * Grad_Flow) + self.epsilon) * self.lamb





