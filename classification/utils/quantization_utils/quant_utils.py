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

import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import torch.nn as nn


def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        if x_min is None or x_max is None or (sum(x_min == x_max) == 1
                                              and x_min.numel() == 1):
            x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        # raise NotImplementedError
        return grad_output.clone(), None, None, None




def quantize_int(x, k, x_min=None, x_max=None):
    if x_min is None or x_max is None or (sum(x_min == x_max) == 1 and x_min.numel() == 1):
        x_min, x_max = x.min(), x.max()
    scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
    quantfunc = LinearQuantizeModule()
    # new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    new_quant_x = quantfunc(x, scale, zero_point, inplace=False)
    n = 2**(k - 1)
    # new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    if len(scale.shape) == 0:
        return new_quant_x, torch.Tensor([scale]), torch.Tensor([zero_point])
    return new_quant_x, scale, zero_point


class LinearDequantizeModule(nn.Module):
    def __init__(self):
        super(LinearDequantizeModule, self).__init__()

    def forward(self, x, scale_x, scale_w):
        self.M = 1 / (scale_x * scale_w)
        M_0 = torch.round(self.M * 2**31)
        res = ((x * M_0) << 31)
        # print(f"M:{self.M}, M_0:{M_0}")
        # print(res.sum())
        # print((x*self.M).sum())
        # exit()
        return res

    def backward(self, grad_output):
        return self.M * grad_output.clone(), None, None, None


class LinearQuantizeModule(nn.Module):
    def __init__(self):
        super(LinearQuantizeModule, self).__init__()

    def forward(self, input, scale, zero_point, inplace=False):
        """
        Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
        input: single-precision input tensor to be quantized
        scale: scaling factor for quantization
        zero_pint: shift for quantization
        """

        # reshape scale and zeropoint for convolutional weights and activation
        self.scale      = scale
        self.zero_point = zero_point
        if len(input.shape) == 4:
            scale      = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(input.shape) == 2:
            scale      = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        # mapping single-precision input to integer values with the given scale and zeropoint
        if inplace:
            input.mul_(scale).sub_(zero_point).round_()
            return input
        return torch.round(scale * input - zero_point)

    def backward(self, grad_output):
        return self.scale * grad_output.clone(), None, None, None
