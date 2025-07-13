import torch
from torch import nn
from torch.nn import functional as F
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization

HYPO_A = 1.0
HYPO_B = 1.0

class LinearReparameterizationHierarchical(LinearReparameterization):
    """
    사전 분산(prior variance)에 대해 Inverse-Gamma 사전 분포를 사용하는
    계층적 베이즈 선형 레이어.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True,
                 prior_variance_hypo_a=HYPO_A,
                 prior_variance_hypo_b=HYPO_B):

        # 부모 클래스의 생성자를 호출하여 기본 파라미터들(mu, rho)을 초기화합니다.
        # 이 때, 부모 클래스의 prior_variance는 사용되지 않으므로 1.0으로 둡니다.
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         prior_mean=prior_mean,
                         prior_variance=prior_variance,
                         posterior_mu_init=posterior_mu_init,
                         posterior_rho_init=posterior_rho_init,
                         bias=bias)
        raise NotImplementedError("Do Not Use this Class. Use LinearReparameterizationHierarchical_Weightwise instead.")
        # 1. 역감마 사전분포의 하이퍼파라미터(a_0, b_0)를 저장합니다.
        # 이 값들은 고정된 상수입니다.
        self.prior_variance_hypo_a = torch.tensor(float(prior_variance_hypo_a))
        self.prior_variance_hypo_b = torch.tensor(float(prior_variance_hypo_b))

        # 2. q(σ_p^2) = Inv-Gamma(a_q, b_q)의 변분 파라미터(a_q, b_q)를 정의합니다.
        # 이 파라미터들은 학습을 통해 최적화되어야 하므로 nn.Parameter로 등록합니다.
        # torch.rand(1)를 더해 초기값을 약간 다르게 설정합니다.
        self.log_a_q = nn.Parameter(torch.log(torch.tensor(1.0) + torch.rand(1)))
        self.log_b_q = nn.Parameter(torch.log(torch.tensor(1.0) + torch.rand(1)))

    def kl_loss(self):
        """
        Empirical Bayes Prior를 사용하도록 재정의된 KL-Divergence 계산 메소드.
        사전 분포의 평균으로 self.prior_weight_mu를 사용합니다.
        """
        a_q = torch.exp(self.log_a_q)
        b_q = torch.exp(self.log_b_q)

        self.prior_variance_hypo_a = self.prior_variance_hypo_a.to(self.mu_weight.device)
        self.prior_variance_hypo_b = self.prior_variance_hypo_b.to(self.mu_weight.device)
        
        # --- 항 A: 가중치에 대한 KL-Divergence (수정된 부분) ---
        mu_W = self.mu_weight
        sigma_W = torch.log1p(torch.exp(self.rho_weight))
        
        # 사전 분포의 평균과 표준편차의 기댓값 계산
        # prior_weight_mu는 main 스크립트에서 설정한 DNN 가중치입니다.
        prior_mu_W = self.prior_weight_mu.to(mu_W.device)
        E_inv_sigma_p_sq = a_q / b_q
        E_log_sigma_p_sq = torch.log(b_q) - torch.digamma(a_q)

        # KL(q(W) || p(W|σ²))의 기댓값 계산
        # p(W|σ²) = N(prior_mu_W, σ²)를 사용
        kl_W = 0.5 * torch.sum(
            E_log_sigma_p_sq                                               # E[log σ_p²]
            - torch.log(sigma_W**2)                                        # - log σ_q²
            + E_inv_sigma_p_sq * (sigma_W**2 + (mu_W - prior_mu_W)**2)      # E[1/σ_p²] * (σ_q² + (μ_q - μ_p)²)
            - 1
        )
        kl_W_sum = kl_W

        if self.bias:
            mu_b = self.mu_bias
            sigma_b = torch.log1p(torch.exp(self.rho_bias))
            prior_mu_b = self.prior_bias_mu.to(mu_b.device) # 편향의 사전 분포 평균 (기본값 0)

            kl_b = 0.5 * torch.sum(
                E_log_sigma_p_sq
                - torch.log(sigma_b**2)
                + E_inv_sigma_p_sq * (sigma_b**2 + (mu_b - prior_mu_b)**2)
                - 1
            )
            kl_W_sum += kl_b

        # --- 항 B: 분산에 대한 KL-Divergence (기존과 동일) ---
        a_0 = self.prior_variance_hypo_a
        b_0 = self.prior_variance_hypo_b

        kl_sigma_p_sq = (a_q - a_0) * torch.digamma(a_q) \
                        - torch.lgamma(a_q) \
                        + torch.lgamma(a_0) \
                        + a_0 * (torch.log(b_q) - torch.log(b_0)) \
                        + (b_0 - b_q) * (a_q / b_q)

        return kl_W_sum + kl_sigma_p_sq
    
    def forward(self, input, return_kl=True):
        
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        tmp_result = sigma_weight * eps_weight
        weight = self.mu_weight + tmp_result

        if return_kl:
            # kl_weight = self.kl_div(self.mu_weight, sigma_weight,
            #                         self.prior_weight_mu, self.prior_weight_sigma)
            kl_weight = self.kl_loss()
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.linear(input, weight, bias)

        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_weight = self.qint_quant[1](self.mu_weight) # weight
            eps_weight = self.qint_quant[2](eps_weight) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation


        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out


class Conv2dReparameterizationHierarchical(Conv2dReparameterization):
    """
    사전 분산(prior variance)에 대해 Inverse-Gamma 사전 분포를 사용하는
    계층적 베이즈 2D 컨볼루션 레이어.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True, # use_bias -> bias
                 prior_variance_hypo_a=HYPO_A,
                 prior_variance_hypo_b=HYPO_B):

        # 부모 클래스의 생성자에 모든 인자를 전달합니다.
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         prior_mean=prior_mean,
                         prior_variance=prior_variance,
                         posterior_mu_init=posterior_mu_init,
                         posterior_rho_init=posterior_rho_init,
                         bias=bias)

        raise NotImplementedError("Do Not Use this Class. Use Conv2dReparameterizationHierarchical_Weightwise instead.")
        self.prior_variance_hypo_a = torch.tensor(float(prior_variance_hypo_a))
        self.prior_variance_hypo_b = torch.tensor(float(prior_variance_hypo_b))
        
        self.log_a_q = nn.Parameter(torch.log(torch.tensor(1.0) + torch.rand(1)))
        self.log_b_q = nn.Parameter(torch.log(torch.tensor(1.0) + torch.rand(1)))

    def kl_loss(self):
        """
        Empirical Bayes Prior를 사용하도록 재정의된 KL-Divergence 계산 메소드.
        사전 분포의 평균으로 self.prior_weight_mu를 사용합니다.
        """
        a_q = torch.exp(self.log_a_q)
        b_q = torch.exp(self.log_b_q)

        self.prior_variance_hypo_a = self.prior_variance_hypo_a.to(self.mu_kernel.device)
        self.prior_variance_hypo_b = self.prior_variance_hypo_b.to(self.mu_kernel.device)

        # --- 항 A: 가중치에 대한 KL-Divergence (수정된 부분) ---
        mu_kernel = self.mu_kernel
        sigma_kernel = torch.log1p(torch.exp(self.rho_kernel))

        prior_mu_kernel = self.prior_weight_mu.to(mu_kernel.device)
        E_inv_sigma_p_sq = a_q / b_q
        E_log_sigma_p_sq = torch.log(b_q) - torch.digamma(a_q)

        kl_kernel = 0.5 * torch.sum(
            E_log_sigma_p_sq
            - torch.log(sigma_kernel**2)
            + E_inv_sigma_p_sq * (sigma_kernel**2 + (mu_kernel - prior_mu_kernel)**2)
            - 1
        )
        kl_params_sum = kl_kernel

        if self.bias:
            mu_bias = self.mu_bias
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            prior_mu_bias = self.prior_bias_mu.to(mu_bias.device)

            kl_bias = 0.5 * torch.sum(
                E_log_sigma_p_sq
                - torch.log(sigma_bias**2)
                + E_inv_sigma_p_sq * (sigma_bias**2 + (mu_bias - prior_mu_bias)**2)
                - 1
            )
            kl_params_sum += kl_bias

        # --- 항 B: 분산에 대한 KL-Divergence (기존과 동일) ---
        a_0 = self.prior_variance_hypo_a
        b_0 = self.prior_variance_hypo_b

        kl_sigma_p_sq = (a_q - a_0) * torch.digamma(a_q) \
                        - torch.lgamma(a_q) \
                        + torch.lgamma(a_0) \
                        + a_0 * (torch.log(b_q) - torch.log(b_0)) \
                        + (b_0 - b_q) * (a_q / b_q)

        return kl_params_sum + kl_sigma_p_sq
        
    def forward(self, input, return_kl=True):
            if self.dnn_to_bnn_flag:
                return_kl = False

            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            tmp_result = sigma_weight * eps_kernel
            weight = self.mu_kernel + tmp_result

            if return_kl:
                # kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                #                         self.prior_weight_mu, self.prior_weight_sigma)
                kl_weight = self.kl_loss()
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
        
class LinearReparameterizationHierarchical_Weightwise(LinearReparameterization):
    """
    '각 가중치'가 자신의 사전 분산(prior variance)에 대해
    '각각 다른' Inverse-Gamma 사전 분포를 사용하는 계층적 베이즈 선형 레이어.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True): # a_0, b_0 인자 제거

        super().__init__(in_features=in_features,
                         out_features=out_features,
                         prior_mean=prior_mean,
                         prior_variance=prior_variance,
                         posterior_mu_init=posterior_mu_init,
                         posterior_rho_init=posterior_rho_init,
                         bias=bias)

        # a_q, b_q를 가중치(mu_weight)와 동일한 모양의 파라미터로 선언
        self.log_a_q_weight = nn.Parameter(torch.full_like(self.mu_weight, 0.0))
        self.log_b_q_weight = nn.Parameter(torch.full_like(self.mu_weight, 0.0))

        # if self.bias:
        #     self.log_a_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 0.0))
        #     self.log_b_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 0.0))
            
        # a_0, b_0 텐서는 외부에서 설정될 것으로 가정합니다.
        # 예: self.prior_hypo_a_weight = torch.ones_like(self.mu_weight)
        # Default 
        self.prior_hypo_a_weight = torch.ones_like(self.mu_weight)
        self.prior_hypo_b_weight = torch.ones_like(self.mu_weight)
        
        
    def kl_loss(self):
        """
        가중치별 a_0, b_0를 사용하도록 재정의된 KL-Divergence.
        """
        total_kl = 0.0

        # --- 가중치(weight)에 대한 KL Divergence 계산 ---
        mu_W = self.mu_weight
        sigma_W_sq = torch.log1p(torch.exp(self.rho_weight))**2
        a_q_W = torch.exp(self.log_a_q_weight)
        b_q_W = torch.exp(self.log_b_q_weight)
        prior_mu_W = self.prior_weight_mu.to(mu_W.device)
        
        # --- 핵심 변경 사항: a_0, b_0를 스칼라가 아닌 텐서 속성에서 가져옴 ---
        # 이 속성들은 main 스크립트에서 설정해주어야 합니다.
        a_0_W = self.prior_hypo_a_weight.to(mu_W.device)
        b_0_W = self.prior_hypo_b_weight.to(mu_W.device)

        # 항 A_i
        E_log_sigma_p_sq_W = torch.log(b_q_W) - torch.digamma(a_q_W)
        E_inv_sigma_p_sq_W = a_q_W / b_q_W
        kl_A_W = 0.5 * (E_log_sigma_p_sq_W - torch.log(sigma_W_sq) + E_inv_sigma_p_sq_W * (sigma_W_sq + (mu_W - prior_mu_W)**2) - 1)

        # 항 B_i
        # 모든 항이 텐서이므로 element-wise로 계산됩니다.
        kl_B_W = (a_q_W - a_0_W) * torch.digamma(a_q_W) - torch.lgamma(a_q_W) + torch.lgamma(a_0_W) + a_0_W * (torch.log(b_q_W) - torch.log(b_0_W)) + (b_0_W - b_q_W) * E_inv_sigma_p_sq_W
        
        total_kl += torch.sum(kl_A_W + kl_B_W)
        # if self.bias:
        #     mu_b = self.mu_bias
        #     sigma_b_sq = torch.log1p(torch.exp(self.rho_bias))**2
        #     a_q_b = torch.exp(self.log_a_q_bias)
        #     b_q_b = torch.exp(self.log_b_q_bias)
        #     prior_mu_b = self.prior_bias_mu.to(mu_b.device)
        #     a_0_b = self.prior_hypo_a_bias.to(mu_b.device)
        #     b_0_b = self.prior_hypo_b_bias.to(mu_b.device)
            
        #     E_log_sigma_p_sq_b = torch.log(b_q_b) - torch.digamma(a_q_b)
        #     E_inv_sigma_p_sq_b = a_q_b / b_q_b
            
        #     kl_A_b = 0.5 * (E_log_sigma_p_sq_b - torch.log(sigma_b_sq) + E_inv_sigma_p_sq_b * (sigma_b_sq + (mu_b - prior_mu_b)**2) - 1)
        #     kl_B_b = (a_q_b - a_0_b) * torch.digamma(a_q_b) - torch.lgamma(a_q_b) + torch.lgamma(a_0_b) + a_0_b * (torch.log(b_q_b) - torch.log(b_0_b)) + (b_0_b - b_q_b) * E_inv_sigma_p_sq_b

        #     total_kl += torch.sum(kl_A_b + kl_B_b)
        
        # if self.mu_bias is not None:
        #     sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        #     total_kl += self.kl_div(self.mu_bias, sigma_bias,
        #                       self.prior_bias_mu, self.prior_bias_sigma)

        return total_kl


    def forward(self, input, return_kl=True):
        
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        tmp_result = sigma_weight * eps_weight
        weight = self.mu_weight + tmp_result

        if return_kl:
            # kl_weight = self.kl_div(self.mu_weight, sigma_weight,
            #                         self.prior_weight_mu, self.prior_weight_sigma)
            kl_weight = self.kl_loss()

        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.linear(input, weight, bias)

        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_weight = self.qint_quant[1](self.mu_weight) # weight
            eps_weight = self.qint_quant[2](eps_weight) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation


        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out
    
class Conv2dReparameterizationHierarchical_Weightwise(Conv2dReparameterization):
    """
    '각 가중치'가 자신의 사전 분산(prior variance)에 대해
    고유한 Inverse-Gamma 사전 분포를 사용하는 계층적 베이즈 2D 컨볼루션 레이어 (MFVI).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True,
                 prior_variance_hypo_a=1.0,
                 prior_variance_hypo_b=1.0):

        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         prior_mean=prior_mean,
                         prior_variance=prior_variance,
                         posterior_mu_init=posterior_mu_init,
                         posterior_rho_init=posterior_rho_init,
                         bias=bias)

        self.prior_variance_hypo_a = torch.ones_like(self.mu_kernel) * prior_variance_hypo_a
        self.prior_variance_hypo_b = torch.ones_like(self.mu_kernel) * prior_variance_hypo_b

        # --- 핵심 변경 사항 ---
        self.log_a_q_kernel = nn.Parameter(torch.full_like(self.mu_kernel, 0.0))
        self.log_b_q_kernel = nn.Parameter(torch.full_like(self.mu_kernel, 0.0))

        if self.bias:
            self.log_a_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 0.0))
            self.log_b_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 0.0))

    def kl_loss(self):
        """
        MFVI와 Empirical Bayes Prior를 사용하도록 재정의된 KL-Divergence.
        모든 계산은 element-wise로 수행된 후 합산됩니다.
        """
        self.prior_variance_hypo_a = self.prior_variance_hypo_a.to(self.mu_kernel.device)
        self.prior_variance_hypo_b = self.prior_variance_hypo_b.to(self.mu_kernel.device)
        
        total_kl = 0.0

        # --- 커널(kernel)에 대한 KL Divergence 계산 ---
        mu_W = self.mu_kernel
        sigma_W_sq = torch.log1p(torch.exp(self.rho_kernel))**2
        a_q_W = torch.exp(self.log_a_q_kernel)
        b_q_W = torch.exp(self.log_b_q_kernel)
        prior_mu_W = self.prior_weight_mu.to(mu_W.device)

        # 항 A_i (가중치 KL)
        E_log_sigma_p_sq_W = torch.log(b_q_W) - torch.digamma(a_q_W)
        E_inv_sigma_p_sq_W = a_q_W / b_q_W
        kl_A_W = 0.5 * (E_log_sigma_p_sq_W - torch.log(sigma_W_sq) + E_inv_sigma_p_sq_W * (sigma_W_sq + (mu_W - prior_mu_W)**2) - 1)

        # 항 B_i (분산 KL)
        a_0 = self.prior_variance_hypo_a
        b_0 = self.prior_variance_hypo_b
        kl_B_W = (a_q_W - a_0) * torch.digamma(a_q_W) - torch.lgamma(a_q_W) + torch.lgamma(a_0) + a_0 * (torch.log(b_q_W) - torch.log(b_0)) + (b_0 - b_q_W) * E_inv_sigma_p_sq_W
        
        total_kl += torch.sum(kl_A_W + kl_B_W)

        # --- 편향(bias)에 대한 KL Divergence 계산 ---
        if self.bias:
            mu_b = self.mu_bias
            sigma_b_sq = torch.log1p(torch.exp(self.rho_bias))**2
            a_q_b = torch.exp(self.log_a_q_bias)
            b_q_b = torch.exp(self.log_b_q_bias)
            prior_mu_b = self.prior_bias_mu.to(mu_b.device)

            E_log_sigma_p_sq_b = torch.log(b_q_b) - torch.digamma(a_q_b)
            E_inv_sigma_p_sq_b = a_q_b / b_q_b
            
            kl_A_b = 0.5 * (E_log_sigma_p_sq_b - torch.log(sigma_b_sq) + E_inv_sigma_p_sq_b * (sigma_b_sq + (mu_b - prior_mu_b)**2) - 1)
            kl_B_b = (a_q_b - a_0) * torch.digamma(a_q_b) - torch.lgamma(a_q_b) + torch.lgamma(a_0) + a_0 * (torch.log(b_q_b) - torch.log(b_0)) + (b_0 - b_q_b) * E_inv_sigma_p_sq_b

            total_kl += torch.sum(kl_A_b + kl_B_b)

        return total_kl
    
    def forward(self, input, return_kl=True):
            if self.dnn_to_bnn_flag:
                return_kl = False

            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            tmp_result = sigma_weight * eps_kernel
            weight = self.mu_kernel + tmp_result

            if return_kl:
                # kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                #                         self.prior_weight_mu, self.prior_weight_sigma)
                kl_weight = self.kl_loss()
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
        
if __name__ == "__main__":
    # 테스트용 코드
    layer = LinearReparameterizationHierarchical(10, 5)
    print("KL Loss:", layer.kl_loss().item())
    print("Prior Variance Hypo a:", layer.prior_variance_hypo_a.item())
    print("Prior Variance Hypo b:", layer.prior_variance_hypo_b.item())
    print("Log a_q:", layer.log_a_q.item())
    print("Log b_q:", layer.log_b_q.item())
    
    