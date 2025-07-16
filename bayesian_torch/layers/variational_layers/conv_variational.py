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
import numpy as np

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
                 prior_type=None,
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
        self.prior_type = prior_type
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
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma, prior_type = self.prior_type)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma, prior_type = self.prior_type)

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
                                    self.prior_weight_mu, self.prior_weight_sigma, prior_type = self.prior_type)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma, prior_type = self.prior_type)

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
                 rank = 1,
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

        super(Conv2dReparameterization_Multivariate, self).__init__()
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
        # variance of weight --> sigma = log (1 + exp(rho))
        self.bias = bias

        kernel_size = get_kernel_size(kernel_size, 2)
        weight_size = out_channels * (in_channels // groups) * kernel_size[0] * kernel_size[1]
        self.weight_size = weight_size
        # self.mu_kernel = Parameter(
        #     torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
        #                  kernel_size[1]))
        
        
        self.mu_kernel = Parameter(torch.Tensor(weight_size))
        self.L_param = Parameter(torch.Tensor(weight_size, rank))
        # self.D_param = Parameter(torch.Tensor(1))
        self.D_param = torch.ones_like(self.mu_kernel) * 1e-10
        
        self.register_buffer(
            'prior_mean',
            torch.Tensor(weight_size),
            # torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
            #              kernel_size[1]),
            persistent=True)
        
        self.register_buffer(
            'prior_cov_L', 
            torch.Tensor(weight_size, 1), 
            persistent=True)
        
        self.register_buffer(
            'prior_cov_D', 
            torch.Tensor(weight_size), 
            persistent=True)

        self.init_parameters(weight_size)

        self.martern_prior = False
        self.BLOCK_MAT = self.covariance_matrix_by_filter((kernel_size[0], kernel_size[1]), sigma=1.0, lamb=1.0)
        
    def init_parameters(self, weight_size):
        
        # Set Multivariate Normal Prior as N(0, I)
        self.prior_mean.data.copy_(torch.zeros(weight_size))
        self.prior_cov_L.data.copy_(torch.zeros((weight_size, 1)))
        self.prior_cov_D.data.copy_(torch.ones(weight_size))

        self.mu_kernel.data.normal_(mean= 0 , std=0.1)
        self.L_param.data.normal_(mean= 0, std=0.1)
        # self.D_param.data.normal_(mean= 0, std=0.1)

    def kl_loss(self):
        
        return kl_divergence(self.variational_mvn, self.prior_mvn)
       
    def get_covariance_param(self):
        
        return self.L_param, self.D_param#.exp().log1p().expand_as(self.mu_kernel).to(self.L_param.device)
        
    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        
        L, D = self.get_covariance_param()
        # D = F.softplus(self.D_param.expand_as(self.mu_kernel))
        self.variational_mvn = LowRankMultivariateNormal(self.mu_kernel, L, D)
        
        weight = self.variational_mvn.rsample().view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        self.prior_mvn = LowRankMultivariateNormal(
            self.prior_mean.to(weight.device),
            self.prior_cov_L.to(weight.device),
            self.prior_cov_D.to(weight.device)
        )
        
        if return_kl:
            
            if self.martern_prior:
                kl_weight = self.martern_cov_kl_loss(
                    B = self.BLOCK_MAT,
                    n = self.in_channels * self.out_channels,
                    L = self.L_param.T,
                    d = self.D_param.exp().log1p(),
                    mu_q = self.mu_kernel.view(-1)
                )
                assert not torch.isnan(kl_weight), "KL divergence is NaN"
            else:
                kl_weight = kl_divergence(self.variational_mvn, self.prior_mvn)
       
        bias = None
        out = F.conv2d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)

       
        if return_kl:
           
            kl = kl_weight
            # Normalize KLD by the number of parameters in the layer as like in the original implementation (class Conv2dReparameterization)
            kl /= self.weight_size
            return out, kl
            
        return out
    
    def martern_cov_kl_loss(self, B, n, L, d, mu_q):
        '''
        B: Block matrix (torch.Tensor of shape (m, m))
        n: The number of blocks in the block diagonal matrix Sigma_p
        L: Covariance factor matrix of q (torch.Tensor of shape (k, D))
        d: Scalar value representing the diagonal elements of D (since D = d * I_D)
        mu_q: Mean vector of q (torch.Tensor of shape (D,))
        '''
        # Ensure all inputs are on the same device (e.g., GPU if available)
        device = B.device
        L = L.to(device)
        mu_q = mu_q.to(device)
        d = d.to(device)
        
        # Inverse of B
        B_inv = torch.inverse(B)
        # Trace of B_inv
        tr_B_inv = torch.trace(B_inv)
        
        # Total dimension D (should be equal to n * m)
        D_total = mu_q.shape[0]
        # Size of each block in B (assuming B is square)
        m = B.shape[0]
        # Ensure that D_total equals n times m
        assert D_total == n * m, f"Dimension mismatch: D_total ({D_total}) should be n ({n}) * m ({m})"
        k = L.shape[0]  # Rank of L
        
        # Reshape and permute L to shape (n, k, m)
        L = L.view(k, n, m).permute(1, 0, 2)  # Now L has shape (n, k, m)
        
        # Compute L_i.t() @ L_i for all i
        L_i_t = L.permute(0, 2, 1)  # Shape: (n, m, k)
        L_i_t_L_i = torch.bmm(L_i_t, L)  # Shape: (n, m, m)
        
        # Expand B_inv to shape (n, m, m)
        B_inv_expanded = B_inv.unsqueeze(0).expand(n, m, m)
        
        # Compute B_inv @ (L_i.t() @ L_i) for all i
        term1_matrices = torch.bmm(B_inv_expanded, L_i_t_L_i)  # Shape: (n, m, m)
        
        # Compute term1_i by summing the diagonal elements
        term1_i = term1_matrices.diagonal(dim1=1, dim2=2).sum(-1)  # Shape: (n,)
        Term1 = term1_i.sum()
        
        # Reshape mu_q to shape (n, m)
        mu_q = mu_q.view(n, m)
        
        # Compute mu_q_i @ B_inv for all i
        mu_q_B_inv = mu_q @ B_inv  # Shape: (n, m)
        
        # Compute term2_i
        term2_i = (mu_q_B_inv * mu_q).sum(-1)  # Shape: (n,)
        Term2 = term2_i.sum()
        
        # Compute Term3: d * n * Tr(B_inv)
        Term3 = d * n * tr_B_inv
        
        # Compute constants
        term_const = -D_total + n * torch.logdet(B) - D_total * torch.log(d)
        
        # Compute the log-determinant term
        # LLT is L @ L.T (shape: k x k)
        L_flat = L.permute(1, 0, 2).reshape(k, -1)  # Flatten L back to shape (k, D_total)
        LLT = L_flat @ L_flat.t()  # Shape: (k, k)
        # Compute the determinant of (I_k + (1/d) * LLT)
        I_k = torch.eye(k, device=device)  # Identity matrix of size k x k
        det_term_matrix = I_k + (1 / d) * LLT
        # Compute the log-determinant
        term_logdet = torch.logdet(det_term_matrix)
        
        # Final KL divergence calculation
        kl = 0.5 * (Term1 + Term2 + Term3 + term_const - term_logdet)
        
        return kl
    
    def covariance_matrix_by_filter(self, filter_size, sigma=1.0, lamb=1.0):
        """
        고정된 필터 크기에 대해 공분산 행렬을 계산하는 함수.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 필터 좌표 생성
        coords = torch.tensor([(float(i), float(j)) for i in range(filter_size[0]) for j in range(filter_size[1])], device=device)
        n = coords.shape[0]

        # 좌표 간 거리 계산
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # 크기: [n, n, 2]
        dist = torch.norm(diff, dim=2)  # 크기: [n, n]

        # 공분산 계산
        cov_matrix = (sigma ** 2) * torch.exp(-dist / lamb)  # 크기: [n, n]

        return cov_matrix
    
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

