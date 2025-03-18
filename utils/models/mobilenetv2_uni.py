import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers import Conv2dReparameterization, LinearReparameterization

# --- Bayesian Inverted Residual Block ---
class InvertedResidual_uni(nn.Module):
    """
    MobileNetV2의 Inverted Residual Block의 Bayesian 버전.
    
    Args:
        inp (int): 입력 채널 수.
        oup (int): 출력 채널 수.
        stride (int): stride (1 또는 2).
        expand_ratio (int): 확장 비율.
        prior_mean, prior_variance, posterior_mu_init, posterior_rho_init: 
            Bayesian layer에 필요한 파라미터들.
    """
    def __init__(self, inp, oup, stride, expand_ratio, 
                 prior_mean=0.0, prior_variance=1.0, 
                 posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super(InvertedResidual_uni, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be 1 or 2"
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)

        # 확장 단계 (expand_ratio != 1일 때)
        if expand_ratio != 1:
            self.expansion_conv = Conv2dReparameterization(
                inp, hidden_dim, kernel_size=1, bias=False,
                prior_mean=prior_mean, prior_variance=prior_variance,
                posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
            )
            self.expansion_bn = nn.BatchNorm2d(hidden_dim)
        else:
            self.expansion_conv = None
            hidden_dim = inp

        # Depthwise convolution (groups = hidden_dim)
        self.depthwise_conv = Conv2dReparameterization(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
            groups=hidden_dim, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)

        # Projection (linear bottleneck)
        self.projection_conv = Conv2dReparameterization(
            hidden_dim, oup, kernel_size=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.projection_bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        kl_sum = 0
        identity = x

        out = x
        if self.expansion_conv is not None:
            out, kl = self.expansion_conv(out)
            kl_sum += kl
            out = self.expansion_bn(out)
            out = F.relu6(out, inplace=True)
        out, kl = self.depthwise_conv(out)
        kl_sum += kl
        out = self.depthwise_bn(out)
        out = F.relu6(out, inplace=True)
        out, kl = self.projection_conv(out)
        kl_sum += kl
        out = self.projection_bn(out)

        if self.use_res_connect:
            out = out + identity
        return out, kl_sum

# --- Bayesian MobileNetV2 ---
class MobileNetv2_uni(nn.Module):
    """
    Bayesian MobileNetV2 (Uni variational) 모델.
    
    Args:
        num_classes (int): 분류할 클래스 수.
        width_mult (float): 채널 수 스케일링 계수.
        prior_mean, prior_variance, posterior_mu_init, posterior_rho_init: 
            Bayesian layer에 필요한 파라미터들.
    """
    def __init__(self, num_classes=1000, width_mult=1.0,
                 prior_mean=0.0, prior_variance=1.0, 
                 posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super(MobileNetv2_uni, self).__init__()
        # MobileNetV2 설정: [expansion, output_channels, num_blocks, stride]
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        
        # 첫 번째 Bayesian conv layer (3x3)
        self.first_conv = Conv2dReparameterization(
            3, input_channel, kernel_size=3, stride=2, padding=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.first_bn = nn.BatchNorm2d(input_channel)
        self.first_relu = nn.ReLU6(inplace=True)
        
        # Inverted Residual blocks
        self.inverted_residuals = nn.ModuleList()
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.inverted_residuals.append(
                    InvertedResidual_uni(
                        input_channel, output_channel, stride, expand_ratio=t,
                        prior_mean=prior_mean, prior_variance=prior_variance,
                        posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
                    )
                )
                input_channel = output_channel
        
        # 마지막 conv layer (1x1)
        self.last_conv = Conv2dReparameterization(
            input_channel, self.last_channel, kernel_size=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.last_bn = nn.BatchNorm2d(self.last_channel)
        self.last_relu = nn.ReLU6(inplace=True)
        
        # Classifier
        self.classifier = LinearReparameterization(
            self.last_channel, num_classes,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
            
    def forward(self, x):
        kl_sum = 0
        out, kl = self.first_conv(x)
        kl_sum += kl
        out = self.first_bn(out)
        out = self.first_relu(out)
        
        for block in self.inverted_residuals:
            out, kl = block(out)
            kl_sum += kl
        
        out, kl = self.last_conv(out)
        kl_sum += kl
        out = self.last_bn(out)
        out = self.last_relu(out)
        # Global average pooling
        out = out.mean([2, 3])
        out, kl = self.classifier(out)
        kl_sum += kl
        
        return out, kl_sum

# -------------------------------------------------------------------
# 사용 예시
if __name__ == "__main__":
    # 예: CIFAR-10 (클래스 10) 또는 ImageNet (클래스 1000)용 Bayesian MobileNetv2_uni
    model = MobileNetv2_uni(num_classes=10, width_mult=1.0)
    print(model)
    # 배치 크기를 4로 하여 forward 시 KL divergence 합산값도 확인 가능
    x = torch.randn(4, 3, 32, 32)
    output, kl_total = model(x)
    
    # Check the number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Output shape:", output.shape)
    print("Total KL divergence:", kl_total)
