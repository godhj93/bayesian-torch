# Copyright (C) 2024 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Convolutional Layers with reparameterization estimator to perform variational
# inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after convolution operation, which is
# required to compute Evidence Lower Bound (ELBO).
#
# @authors: Ranganath Krishnan
#
# ======================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ..base_variational_layer import BaseVariationalLayer_, get_kernel_size
import math
from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver, MinMaxObserver
from torch.quantization.qconfig import QConfig
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import LowRankMultivariateNormal, kl_divergence
from termcolor import colored

__all__ = [
    'Conv1dReparameterization',
    'Conv2dReparameterization',
    'Conv3dReparameterization',
    'ConvTranspose1dReparameterization',
    'ConvTranspose2dReparameterization',
    'ConvTranspose3dReparameterization',
]


class Conv1dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv1d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(Conv1dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.rho_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False

    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        
        if self.bias:
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result
        
        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv1d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)
        
        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            return out, kl

        return out


class Conv2dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super(Conv2dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        kernel_size = get_kernel_size(kernel_size, 2)

        self.mu_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.rho_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False

    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv2d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)

        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation
            

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            return out, kl
            
        return out

class Conv2dReparameterization_Multivariate(BaseVariationalLayer_):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                prior_mean = None,
                prior_variance = None,
                posterior_mu_init=0,
                posterior_rho_init=-3.0,
                bias=False):
        """
        Implements Conv2d layer with reparameterization trick using multivariate Gaussian distribution.
        @Author: Heejung Shin, godhj@unist.ac.kr
        
        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int or tuple -> size of the convolving kernel,
            stride: int or tuple -> stride of the convolution. Default: 1,
            padding: int or tuple -> zero-padding added to both sides of the input. Default: 0,
            dilation: int or tuple -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(Conv2dReparameterization_Multivariate, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid out_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        kernel_size = self.kernel_size
        weight_size = out_channels * (in_channels // groups) * kernel_size[0] * kernel_size[1]

        if self.prior_mean is None:
            self.prior_mean = torch.zeros(weight_size)
        else:
            self.prior_mean = prior_mean.view(-1)
            
        if self.prior_variance is None:
            self.prior_cov_L = torch.zeros((weight_size, 1))
            self.prior_cov_B = torch.ones(weight_size) 
        else:
            raise NotImplementedError("Prior variance should be None")
            # self.prior_cov_L, self.prior_cov_B = self.prior_variance
        
        self.mu_kernel = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))        
        self.L_param = Parameter(torch.Tensor(weight_size, 1))
        self.B = torch.ones(weight_size) * 1e-20
        # self.B_param = Parameter(torch.Tensor(weight_size))
        
        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_channels), persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare = False

    def init_parameters(self):
        # self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=0.1)
        # self.L_param.data.normal_(mean=0, std=1)
        # self.logB_param.data.normal_(mean=0, std=0.01)
        
        # if self.bias:
        #     self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
        #     self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
        nn.init.xavier_normal_(self.mu_kernel)
        
        fan_in = self.in_channels
        fan_out = self.out_channels
        variance_L = 2.0 / (fan_in + fan_out)
        std_L = variance_L ** 0.5
        
        self.L_param.data.normal_(mean=0, std=std_L)
        

    def get_covariance_param(self):        
        '''
        L: covariance factor
        B: diagonal factor
        '''
        
        return self.L_param, self.B.to(self.L_param.device)

    def forward(self, input, return_kl=True):
        weight_shape = self.mu_kernel.shape

        mu_flat = self.mu_kernel.view(-1)
        L, B = self.get_covariance_param()# + (self.epsilon * torch.eye(mu_flat.size(0))).to(mu_flat.device)

        
        # Use MultivariateNormal for sampling
        mvn = LowRankMultivariateNormal(mu_flat, L, B)
        weight_flat = mvn.rsample() # Reparameterization trick
        weight = weight_flat.view(weight_shape)
        
        if return_kl:
            
            # prior_mvn = MultivariateNormal(self.prior_mean.to(mu_flat.device), self.prior_variance.to(mu_flat.device))
            prior_mvn = LowRankMultivariateNormal(self.prior_mean.to(mu_flat.device), self.prior_cov_L.to(mu_flat.device), self.prior_cov_B.to(mu_flat.device))

            # kl_weight = self.kl_div_multivariate_gaussian(mu_flat, cov_flat, self.prior_mean, self.prior_variance)
            kl_weight = kl_divergence(mvn, prior_mvn)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + sigma_bias * eps_bias
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            return out, kl
            
        return out
    
class Conv3dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 prior_mean,
                 prior_variance,
                 posterior_mu_init,
                 posterior_rho_init,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        """
        Implements Conv3d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super(Conv3dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias
        kernel_size = get_kernel_size(kernel_size, 3)
        self.mu_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.rho_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False

    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv3d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)
        
        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            return out, kl

        return out


class ConvTranspose1dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 output_padding=0,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements ConvTranspose1d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(ConvTranspose1dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_kernel = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.rho_kernel = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(in_channels, out_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(in_channels, out_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(in_channels, out_channels // groups, kernel_size),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False

    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv_transpose1d(input, weight, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.groups, self.dilation)

        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation
        
        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out


class ConvTranspose2dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 output_padding=0,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements ConvTranspose2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(ConvTranspose2dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias
        kernel_size = get_kernel_size(kernel_size, 2)
        self.mu_kernel = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.rho_kernel = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False

    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv_transpose2d(input, weight, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.groups, self.dilation)
        
        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out


class ConvTranspose3dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 output_padding=0,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements ConvTranspose3d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super(ConvTranspose3dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias
        kernel_size = get_kernel_size(kernel_size, 3)
        self.mu_kernel = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.rho_kernel = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False

    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv_transpose3d(input, weight, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.groups, self.dilation)
        
        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            return out, kl

        return out

