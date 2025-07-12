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
        계층적 모델에 맞게 재정의된 KL-Divergence 계산 메소드.
        ELBO = E[log P(D|W)] - KL_divergence
        여기서는 KL_divergence 부분만 계산합니다.
        """
        # 양수 제약을 위해 파라미터에 exp를 취해줍니다.
        a_q = torch.exp(self.log_a_q)
        b_q = torch.exp(self.log_b_q)

        # 디바이스 동기화
        self.prior_variance_hypo_a = self.prior_variance_hypo_a.to(self.mu_weight.device)
        self.prior_variance_hypo_b = self.prior_variance_hypo_b.to(self.mu_weight.device)
        # --------------------------------------------------------------------
        # 항 A: 가중치에 대한 KL-Divergence (E_q(σ^2)[D_KL(q(W)||p(W|σ^2))])
        # --------------------------------------------------------------------

        # 1. q(W)의 파라미터
        mu_W = self.mu_weight
        sigma_W = torch.log1p(torch.exp(self.rho_weight)) # σ = log(1+exp(ρ))
        kl_W_sum = 0

        if self.bias:
            mu_b = self.mu_bias
            sigma_b = torch.log1p(torch.exp(self.rho_bias))

        # 2. E[1/σ_p^2] 와 E[log σ_p^2] 계산
        # q(σ_p^2) = Inv-Gamma(a_q, b_q)
        E_inv_sigma_p_sq = a_q / b_q
        E_log_sigma_p_sq = torch.log(b_q) - torch.digamma(a_q)

        # 3. 가중치(kernel)에 대한 KL 계산
        # (μ_q^2 + σ_q^2)
        sum_mu_sq_plus_sigma_sq_W = torch.sum(mu_W**2 + sigma_W**2)
        # log σ_q^2
        sum_log_sigma_sq_W = torch.sum(torch.log(sigma_W**2))

        # 식: 0.5 * [ E[1/σ^2] * Σ(μ_w^2+σ_w^2) - D + D*E[log σ^2] - Σ log σ_w^2 ]
        D_W = mu_W.numel() # 가중치의 총 개수
        kl_W = 0.5 * (E_inv_sigma_p_sq * sum_mu_sq_plus_sigma_sq_W \
                      - D_W \
                      + D_W * E_log_sigma_p_sq \
                      - sum_log_sigma_sq_W)
        kl_W_sum += kl_W

        # 4. 편향(bias)에 대한 KL 계산 (사용하는 경우)
        if self.bias:
            sum_mu_sq_plus_sigma_sq_b = torch.sum(mu_b**2 + sigma_b**2)
            sum_log_sigma_sq_b = torch.sum(torch.log(sigma_b**2))
            D_b = mu_b.numel() # 편향의 총 개수
            kl_b = 0.5 * (E_inv_sigma_p_sq * sum_mu_sq_plus_sigma_sq_b \
                          - D_b \
                          + D_b * E_log_sigma_p_sq \
                          - sum_log_sigma_sq_b)
            kl_W_sum += kl_b


        # --------------------------------------------------------------------
        # 항 B: 분산에 대한 KL-Divergence (D_KL(q(σ_p^2)||p(σ_p^2)))
        # --------------------------------------------------------------------
        a_0 = self.prior_variance_hypo_a
        b_0 = self.prior_variance_hypo_b

        # 식: (a_q-a_0)ψ(a_q) - logΓ(a_q) + logΓ(a_0) + a_0(log b_q - log b_0) + (b_0-b_q) * a_q/b_q
        kl_sigma_p_sq = (a_q - a_0) * torch.digamma(a_q) \
                        - torch.lgamma(a_q) \
                        + torch.lgamma(a_0) \
                        + a_0 * (torch.log(b_q) - torch.log(b_0)) \
                        + (b_0 - b_q) * (a_q / b_q)

        # --------------------------------------------------------------------
        # 최종 KL Loss: 두 항의 합
        # --------------------------------------------------------------------
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

        self.prior_variance_hypo_a = torch.tensor(float(prior_variance_hypo_a))
        self.prior_variance_hypo_b = torch.tensor(float(prior_variance_hypo_b))
        
        self.log_a_q = nn.Parameter(torch.log(torch.tensor(1.0) + torch.rand(1)))
        self.log_b_q = nn.Parameter(torch.log(torch.tensor(1.0) + torch.rand(1)))

    def kl_loss(self):
        """
        계층적 모델에 맞게 재정의된 KL-Divergence 계산 메소드.
        """
        # 양수 제약을 위해 파라미터에 exp를 취해줍니다.
        a_q = torch.exp(self.log_a_q)
        b_q = torch.exp(self.log_b_q)

        # 디바이스 동기화
        self.prior_variance_hypo_a = self.prior_variance_hypo_a.to(self.mu_kernel.device)
        self.prior_variance_hypo_b = self.prior_variance_hypo_b.to(self.mu_kernel.device)

        # --------------------------------------------------------------------
        # 항 A: 가중치에 대한 KL-Divergence (E_q(σ^2)[D_KL(q(W)||p(W|σ^2))])
        # --------------------------------------------------------------------
        
        # q(W)의 파라미터
        mu_kernel = self.mu_kernel
        sigma_kernel = torch.log1p(torch.exp(self.rho_kernel))
        kl_params_sum = 0

        # E[1/σ_p^2] 와 E[log σ_p^2] 계산
        E_inv_sigma_p_sq = a_q / b_q
        E_log_sigma_p_sq = torch.log(b_q) - torch.digamma(a_q)

        # 컨볼루션 커널(kernel)에 대한 KL 계산
        sum_mu_sq_plus_sigma_sq_kernel = torch.sum(mu_kernel**2 + sigma_kernel**2)
        sum_log_sigma_sq_kernel = torch.sum(torch.log(sigma_kernel**2))
        D_kernel = mu_kernel.numel()
        kl_kernel = 0.5 * (E_inv_sigma_p_sq * sum_mu_sq_plus_sigma_sq_kernel \
                           - D_kernel \
                           + D_kernel * E_log_sigma_p_sq \
                           - sum_log_sigma_sq_kernel)
        kl_params_sum += kl_kernel

        # 편향(bias)에 대한 KL 계산 (사용하는 경우)
        if self.bias:
            mu_bias = self.mu_bias
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            
            sum_mu_sq_plus_sigma_sq_bias = torch.sum(mu_bias**2 + sigma_bias**2)
            sum_log_sigma_sq_bias = torch.sum(torch.log(sigma_bias**2))
            D_bias = mu_bias.numel()
            kl_bias = 0.5 * (E_inv_sigma_p_sq * sum_mu_sq_plus_sigma_sq_bias \
                             - D_bias \
                             + D_bias * E_log_sigma_p_sq \
                             - sum_log_sigma_sq_bias)
            kl_params_sum += kl_bias

        # --------------------------------------------------------------------
        # 항 B: 분산에 대한 KL-Divergence (D_KL(q(σ_p^2)||p(σ_p^2)))
        # --------------------------------------------------------------------
        a_0 = self.prior_variance_hypo_a
        b_0 = self.prior_variance_hypo_b

        kl_sigma_p_sq = (a_q - a_0) * torch.digamma(a_q) \
                        - torch.lgamma(a_q) \
                        + torch.lgamma(a_0) \
                        + a_0 * (torch.log(b_q) - torch.log(b_0)) \
                        + (b_0 - b_q) * (a_q / b_q)

        # --------------------------------------------------------------------
        # 최종 KL Loss: 두 항의 합
        # --------------------------------------------------------------------
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
    고유한 Inverse-Gamma 사전 분포를 사용하는 계층적 베이즈 선형 레이어.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True,
                 prior_variance_hypo_a=1.0,
                 prior_variance_hypo_b=1.0):

        super().__init__(in_features=in_features,
                         out_features=out_features,
                         prior_mean=prior_mean,
                         prior_variance=prior_variance,
                         posterior_mu_init=posterior_mu_init,
                         posterior_rho_init=posterior_rho_init,
                         bias=bias)

        # 고정된 하이퍼파라미터 a_0, b_0
        self.prior_variance_hypo_a = torch.tensor(float(prior_variance_hypo_a))
        self.prior_variance_hypo_b = torch.tensor(float(prior_variance_hypo_b))

        # --- 핵심 변경 사항 ---
        # a_q, b_q를 가중치(mu_kernel)와 동일한 모양의 파라미터로 선언
        self.log_a_q_kernel = nn.Parameter(torch.full_like(self.mu_kernel, 1.0))
        self.log_b_q_kernel = nn.Parameter(torch.full_like(self.mu_kernel, 1.0))

        if self.bias:
            # 편향(bias)에 대해서도 동일한 모양의 파라미터 선언
            self.log_a_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 1.0))
            self.log_b_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 1.0))

    def kl_loss(self):
        # 디바이스 동기화
        self.prior_variance_hypo_a = self.prior_variance_hypo_a.to(self.mu_kernel.device)
        self.prior_variance_hypo_b = self.prior_variance_hypo_b.to(self.mu_kernel.device)

        # --- 가중치(kernel)에 대한 KL Divergence 계산 ---
        # 1. q(W)와 q(σ_p^2)의 파라미터
        mu_W = self.mu_kernel
        sigma_W = torch.log1p(torch.exp(self.rho_kernel))
        a_q_W = torch.exp(self.log_a_q_kernel)
        b_q_W = torch.exp(self.log_b_q_kernel)

        # 2. 항 A: E_q(σ^2)[D_KL(q(W)||p(W|σ^2))]
        E_inv_sigma_p_sq_W = a_q_W / b_q_W
        E_log_sigma_p_sq_W = torch.log(b_q_W) - torch.digamma(a_q_W)

        kl_A_W = 0.5 * (E_inv_sigma_p_sq_W * (mu_W**2 + sigma_W**2)
                        - 1
                        + E_log_sigma_p_sq_W
                        - torch.log(sigma_W**2))

        # 3. 항 B: D_KL(q(σ_p^2)||p(σ_p^2))
        a_0 = self.prior_variance_hypo_a
        b_0 = self.prior_variance_hypo_b
        kl_B_W = (a_q_W - a_0) * torch.digamma(a_q_W) \
                 - torch.lgamma(a_q_W) \
                 + torch.lgamma(a_0) \
                 + a_0 * (torch.log(b_q_W) - torch.log(b_0)) \
                 + (b_0 - b_q_W) * (a_q_W / b_q_W)

        # 4. 가중치에 대한 최종 KL Loss (모든 원소에 대해 합산)
        total_kl = torch.sum(kl_A_W + kl_B_W)

        # --- 편향(bias)에 대한 KL Divergence 계산 ---
        if self.bias:
            mu_b = self.mu_bias
            sigma_b = torch.log1p(torch.exp(self.rho_bias))
            a_q_b = torch.exp(self.log_a_q_bias)
            b_q_b = torch.exp(self.log_b_q_bias)

            E_inv_sigma_p_sq_b = a_q_b / b_q_b
            E_log_sigma_p_sq_b = torch.log(b_q_b) - torch.digamma(a_q_b)

            kl_A_b = 0.5 * (E_inv_sigma_p_sq_b * (mu_b**2 + sigma_b**2)
                            - 1
                            + E_log_sigma_p_sq_b
                            - torch.log(sigma_b**2))

            kl_B_b = (a_q_b - a_0) * torch.digamma(a_q_b) \
                     - torch.lgamma(a_q_b) \
                     + torch.lgamma(a_0) \
                     + a_0 * (torch.log(b_q_b) - torch.log(b_0)) \
                     + (b_0 - b_q_b) * (a_q_b / b_q_b)

            total_kl += torch.sum(kl_A_b + kl_B_b)

        return total_kl


class Conv2dReparameterizationHierarchical_Weightwise(Conv2dReparameterization):
    """
    '각 가중치'가 자신의 사전 분산(prior variance)에 대해
    고유한 Inverse-Gamma 사전 분포를 사용하는 계층적 베이즈 2D 컨볼루션 레이어.
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

        self.prior_variance_hypo_a = torch.tensor(float(prior_variance_hypo_a))
        self.prior_variance_hypo_b = torch.tensor(float(prior_variance_hypo_b))

        # --- 핵심 변경 사항 ---
        self.log_a_q_kernel = nn.Parameter(torch.full_like(self.mu_kernel, 1.0))
        self.log_b_q_kernel = nn.Parameter(torch.full_like(self.mu_kernel, 1.0))

        if self.bias:
            self.log_a_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 1.0))
            self.log_b_q_bias = nn.Parameter(torch.full_like(self.mu_bias, 1.0))

    def kl_loss(self):
        # kl_loss 계산 로직은 Linear 버전과 완전히 동일합니다.
        # 코드를 복사하여 붙여넣습니다.
        self.prior_variance_hypo_a = self.prior_variance_hypo_a.to(self.mu_kernel.device)
        self.prior_variance_hypo_b = self.prior_variance_hypo_b.to(self.mu_kernel.device)

        mu_W = self.mu_kernel
        sigma_W = torch.log1p(torch.exp(self.rho_kernel))
        a_q_W = torch.exp(self.log_a_q_kernel)
        b_q_W = torch.exp(self.log_b_q_kernel)

        E_inv_sigma_p_sq_W = a_q_W / b_q_W
        E_log_sigma_p_sq_W = torch.log(b_q_W) - torch.digamma(a_q_W)
        
        kl_A_W = 0.5 * (E_inv_sigma_p_sq_W * (mu_W**2 + sigma_W**2) - 1 + E_log_sigma_p_sq_W - torch.log(sigma_W**2))

        a_0 = self.prior_variance_hypo_a
        b_0 = self.prior_variance_hypo_b
        kl_B_W = (a_q_W - a_0) * torch.digamma(a_q_W) - torch.lgamma(a_q_W) + torch.lgamma(a_0) + a_0 * (torch.log(b_q_W) - torch.log(b_0)) + (b_0 - b_q_W) * (a_q_W / b_q_W)

        total_kl = torch.sum(kl_A_W + kl_B_W)

        if self.bias:
            mu_b = self.mu_bias
            sigma_b = torch.log1p(torch.exp(self.rho_bias))
            a_q_b = torch.exp(self.log_a_q_bias)
            b_q_b = torch.exp(self.log_b_q_bias)

            E_inv_sigma_p_sq_b = a_q_b / b_q_b
            E_log_sigma_p_sq_b = torch.log(b_q_b) - torch.digamma(a_q_b)

            kl_A_b = 0.5 * (E_inv_sigma_p_sq_b * (mu_b**2 + sigma_b**2) - 1 + E_log_sigma_p_sq_b - torch.log(sigma_b**2))

            kl_B_b = (a_q_b - a_0) * torch.digamma(a_q_b) - torch.lgamma(a_q_b) + torch.lgamma(a_0) + a_0 * (torch.log(b_q_b) - torch.log(b_0)) + (b_0 - b_q_b) * (a_q_b / b_q_b)
            
            total_kl += torch.sum(kl_A_b + kl_B_b)

        return total_kl
    
if __name__ == "__main__":
    # 테스트용 코드
    layer = LinearReparameterizationHierarchical(10, 5)
    print("KL Loss:", layer.kl_loss().item())
    print("Prior Variance Hypo a:", layer.prior_variance_hypo_a.item())
    print("Prior Variance Hypo b:", layer.prior_variance_hypo_b.item())
    print("Log a_q:", layer.log_a_q.item())
    print("Log b_q:", layer.log_b_q.item())
    
    