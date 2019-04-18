import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .VGG import vgg19


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

    def _sobel(self, flows):
        return self.sobel(flows)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1,:]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self,t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CropMarginLoss(nn.Module):
    def __init__(self, loss=nn.MSELoss, crop=5):
        super(CropMarginLoss, self).__init__()
        self.loss = loss()
        self.crop = crop

    def forward(self, input, target):
        return self.loss(input[:, :, self.crop: -self.crop, self.crop: -self.crop], target[:, :, self.crop: -self.crop, self.crop: -self.crop])


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VGGLoss(nn.Module):
    """
    VGG(
    (features): Sequential(
    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)  # 5 x 5
    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)  # 14 x 14
    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)  # 48 x 48
    (18): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (19): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)  # 116 x 116
    (27): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    )
    """
    def __init__(self, vgg_path, layers='5', input='RGB', loss='l1', activate='before'):
        super(VGGLoss, self).__init__()
        self.input = input
        vgg = vgg19()
        vgg.load_state_dict(torch.load(vgg_path))
        self.layers = [int(l) for l in layers]
        layers_dict = [0, 4, 9, 18, 27, 36] if activate == 'after' else [0, 3, 8, 17, 26, 35]
        self.vgg = []
        if loss == 'l1':
            self.loss_model = nn.L1Loss()
        elif loss == 'l2':
            self.loss_model = nn.MSELoss()
        else:
            raise Exception('Do not support this loss.')

        i = 0
        for j in self.layers:
            self.vgg.append(nn.Sequential(*list(vgg.features.children())[layers_dict[i]:layers_dict[j]]))
            i = j

    def cuda(self, device=None):
        for Seq in self.vgg:
            Seq.cuda()
        self.loss_model.cuda()

    def forward(self, input, target):
        if self.input == 'RGB':
            input_R, input_G, input_B = torch.split(input, 1, dim=1)
            target_R, target_G, target_B = torch.split(target, 1, dim=1)
            input_BGR = torch.cat([input_B, input_G, input_R], dim=1)
            target_BGR = torch.cat([target_B, target_G, target_R], dim=1)
        else:
            input_BGR = input
            target_BGR = target

        input_list = [input_BGR]
        target_list = [target_BGR]

        for Sequential in self.vgg:
            input_list.append(Sequential(input_list[-1]))
            target_list.append(Sequential(target_list[-1]))

        loss = []
        for i in range(len(self.layers)):
            loss.append(self.loss_model(input_list[i + 1], target_list[i + 1].detach()))
        if len(loss) != 1:
            return sum(loss)
        else:
            return loss[0]


class ContextualLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, epsilon=1e-5, similarity='cos'):
        super(ContextualLoss, self).__init__()
        self.sigma = sigma
        self.similarity = similarity
        self.b = b
        self.e = epsilon

    def cos_similarity(self, image_features, target_features):
        # N, V, C
        if_vec = image_features.view((image_features.size()[0], image_features.size()[1], -1)).permute(0, 2, 1)
        tf_vec = target_features.view((target_features.size()[0], target_features.size()[1], -1)).permute(0, 2, 1)
        # Centre by T
        tf_mean = torch.mean(tf_vec, dim=1, keepdim=True)
        ifc_vec = if_vec - tf_mean
        tfc_vec = tf_vec - tf_mean
        # L2-norm normalization
        ifc_vec_l2 = torch.div(ifc_vec, torch.sqrt(torch.sum(ifc_vec * ifc_vec, dim=2, keepdim=True)))
        tfc_vec_l2 = torch.div(tfc_vec, torch.sqrt(torch.sum(tfc_vec * tfc_vec, dim=2, keepdim=True)))
        # cross dot
        feature_cos_similarity_matrix = 1 - torch.bmm(ifc_vec_l2, tfc_vec_l2.permute(0, 2, 1))
        return feature_cos_similarity_matrix

    def L2_similarity(self, image_features, target_features):
        pass

    def relative_distances(self, feature_similarity_matrix):
        relative_dist = feature_similarity_matrix / (torch.min(feature_similarity_matrix, dim=2, keepdim=True)[0] + self.e)
        return relative_dist

    def weighted_average_distances(self, relative_distances_matrix):
        weights_before_normalization = torch.exp((self.b - relative_distances_matrix) / self.sigma)
        weights_sum = torch.sum(weights_before_normalization, dim=2, keepdim=True)
        weights_normalized = torch.div(weights_before_normalization, weights_sum)
        return weights_normalized

    def CX(self, feature_similarity_matrix):
        CX_i_j = self.weighted_average_distances(self.relative_distances(feature_similarity_matrix))
        CX_j_i = CX_i_j.permute(0, 2, 1)
        max_i_on_j = torch.max(CX_j_i, dim=1)[0]
        CS = torch.mean(max_i_on_j, dim=1)
        CX = - torch.log(CS)
        CX_loss = torch.mean(CX)
        return CX_loss

    def forward(self, image_features, target_features):
        if self.similarity == 'cos':
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
        elif self.similarity == 'l2':
            feature_similarity_matrix = self.L2_similarity(image_features, target_features)
        else:
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
        return self.CX(feature_similarity_matrix)


class VGGContextualLoss(nn.Module):
    def __init__(self, vgg_path, layer=3, input='RGB', sigma=0.1, b=1.0, epsilon=1e-5, similarity='cos', activate='before'):
        super(VGGContextualLoss, self).__init__()
        self.input = input
        vgg = vgg19()
        vgg.load_state_dict(torch.load(vgg_path))
        self.layer = layer  # layers in [1, 2, 3, 4, 5]
        self.CXLoss = ContextualLoss(sigma=sigma, b=b, epsilon=epsilon, similarity=similarity)
        layers_dict = [0, 4, 9, 18, 27, 36] if activate == 'after' else [0, 3, 8, 17, 26, 35]
        self.sequential = nn.Sequential(*list(vgg.features.children())[0:layers_dict[layer-1]])

    def forward(self, image, target):
        if self.input == 'RGB':
            input_R, input_G, input_B = torch.split(image, 1, dim=1)
            target_R, target_G, target_B = torch.split(target, 1, dim=1)
            image_BGR = torch.cat([input_B, input_G, input_R], dim=1)
            target_BGR = torch.cat([target_B, target_G, target_R], dim=1)
        else:
            image_BGR = image
            target_BGR = target

        image_features = self.sequential(image_BGR)
        target_features = self.sequential(target_BGR)
        return self.CXLoss(image_features, target_features)

    def cuda(self, device=None):
        self.sequential.cuda()




