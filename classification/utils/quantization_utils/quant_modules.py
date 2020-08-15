#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .quant_utils import *
import sys


class QuantAct(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True,
                 integer_only=True):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.momentum = 0.99
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.act_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def forward(self, x):
        """
        quantize given activation x
        """
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min,
                                          self.x_max)
            return quant_act
        else:
            return x


class Quant_Linear(Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Linear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        else:
            w = self.weight
        return F.linear(x, weight=w, bias=self.bias)


class Quant_Conv2d(Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)






# Integer Only Implementation
class QuantAct_Int(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True,
                 integer_only=True):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct_Int, self).__init__()
        self.activation_bit = activation_bit
        self.momentum = 0.99
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def forward(self, x):
        """
        quantize given activation x
        """
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        if not self.full_precision_flag:
            return x, self.activation_bit, self.x_min, self.x_max
        else:
            return x


class Quant_Linear_Int(Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear_Int, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit          = weight_bit

    def __repr__(self):
        s = super(Quant_Linear_Int, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            x, activation_bit, x_min, x_max = inputs
        else:
            x = inputs
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            new_quant_w, scale_w, zero_point_w = quantize_int(self.weight, self.weight_bit, w_min, w_max)
            new_quant_x, scale_x, zero_point_x = quantize_int(x, activation_bit, x_min, x_max)
            new_quant_b = linear_quantize(self.bias, scale_w*scale_x, 0, inplace=False)
            # TODO bias quantization
            mult_res = F.linear(new_quant_x.int(), weight=new_quant_w.int(), bias=self.bias)
            res = (zero_point_w*zero_point_x) - zero_point_w * new_quant_x.sum(-2) - zero_point_w * new_quant_x.sum(-1) + mult_res + new_quant_b

            print(f"new_quant_x:{new_quant_x.shape}")
            print(f"new_quant_w:{new_quant_w.shape}")
            print(f"scale_x:{scale_x.shape}")
            print(f"scale_w:{scale_w.shape}")
            print(f"zero_point_x:{zero_point_x.shape}")
            print(f"zero_point_w:{zero_point_w.shape}")
            print(f"multiplication result:{mult_res.shape}")
            exit()

            r_min, r_max = res.min(), res.max()
            scale_r, zero_point_r = asymmetric_linear_quantization_params(k, r_min, r_max)
            quant_r = dequantize_int(res, scale_r, scale_w, scale_x, zero_point_r)
            return quant_r

        else:
            w = self.weight
            return F.linear(x, weight=w, bias=self.bias)


class Quant_Conv2d_Int(Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d_Int, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Conv2d_Int, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            print(1)
            x, activation_bit, x_min, x_max = inputs
        else:
            print(0)
            x = inputs
            x_min, x_max = None, None
            activation_bit = 32

        """
        using quantized weights to forward activation x
        """

        # w = self.weight
        calc_x = nn.functional.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride, padding=self.padding).transpose(1, 2)
        calc_w = self.weight
        x_transform = calc_w.data.contiguous().view(self.out_channels, -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            print("-----w-----")
            new_quant_w, scale_w, zero_point_w = quantize_int(calc_w, self.weight_bit, w_min, w_max)
            print("-----x-----")
            new_quant_x, scale_x, zero_point_x = quantize_int(calc_x, activation_bit, x_min, x_max)
            new_quant_b = linear_quantize(self.bias, scale_w*scale_x, 0, inplace=False)

            # TODO bias quantization
            # DONT forget to transpose(1,2) on the result
            mult_res = torch.matmul(new_quant_x, new_quant_w.view(new_quant_w.size(0), -1).t())

            res = (zero_point_w*zero_point_x) - zero_point_w * new_quant_x.sum(-2) - zero_point_w * new_quant_x.sum(-1) + mult_res + new_quant_b

            r_min, r_max = res.min(), res.max()
            scale_r, zero_point_r = asymmetric_linear_quantization_params(k, r_min, r_max)
            quant_r = dequantize_int(res, scale_r, scale_w, scale_x, zero_point_r).transpose(1,2)

            return quant_r

        else:
            w = self.weight
            return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
