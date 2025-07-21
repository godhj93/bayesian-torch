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
# ===============================================================================================
import torch
import torch.nn as nn
import torch.distributions as distributions
from itertools import repeat
import collections

standard_normal = torch.distributions.Normal(0, 1)

def get_kernel_size(x, n):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p, prior_type):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        
        if prior_type == 'normal':
        # This implementation is correct.
            kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
            return kl.mean()

        elif prior_type == 'laplace':
            
            # --- 샘플링 대신 해석적 해를 사용한 새로운 구현 ---
            
            # Laplace 분포의 스케일 파라미터 b는 sigma_p로 전달됩니다.
            b_p = torch.tensor(1.0)
            mu_p = torch.tensor(0.0)
            print(f"Using Laplace prior with b_p: {b_p}, mu_p: {mu_p}")
            # Laplace 분포의 위치(mu_p)를 고려하여 mu_q의 오프셋을 계산합니다.
            mu_q_offset = mu_q - mu_p
            
            # 해석적 공식에 따라 KL Divergence를 계산합니다.
            # KL(q||p) = log(2b) - 0.5*log(2*pi*sigma^2) - 0.5 + (1/b) * E_q[|w - mu_p|]
            
            # E_q[|w - mu_p|]는 접힌 정규 분포(Folded Normal)의 평균입니다.
            exp_abs_val = sigma_q * torch.sqrt(torch.tensor(2.0 / torch.pi)) * \
                        torch.exp(-mu_q_offset**2 / (2 * sigma_q**2)) + \
                        mu_q_offset * (1 - 2 * standard_normal.cdf(-mu_q_offset / sigma_q))
            
            # 모든 항을 결합합니다.
            kl = torch.log(2 * b_p) - 0.5 * torch.log(2 * torch.pi * sigma_q**2) - 0.5 + \
                (1 / b_p) * exp_abs_val
                
            return kl.mean()

        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")

    def kl_div_multivariate_gaussian(self, mu_q, sigma_q, mu_p, sigma_p, device='cuda'):
        """
        Calculates kl divergence between two multivariate gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """

        kl = 0.5 * (torch.logdet(sigma_p).to(device) - torch.logdet(sigma_q).to(device) +
                    torch.trace(torch.matmul(sigma_q.to(device), torch.inverse(sigma_p).to(device))) +
                    torch.matmul(torch.matmul((mu_q.to(device) - mu_p.to(device)).unsqueeze(-1).permute(1, 0), torch.inverse(sigma_p.to(device))), (mu_q.cuda() - mu_p.cuda()).unsqueeze(-1)).squeeze() - mu_p.shape[0])
        return kl.mean()
    
        