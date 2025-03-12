import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers import Conv2dReparameterization, LinearReparameterization

# Bayesian DenseLayer (Bottleneck 구조)
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate,
                 prior_mean, prior_variance, posterior_mu_init, posterior_rho_init):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # 1x1 convolution (bottleneck)
        self.conv1 = Conv2dReparameterization(
            in_channels, bn_size * growth_rate,
            kernel_size=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        # 3x3 convolution
        self.conv2 = Conv2dReparameterization(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, padding=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out, kl1 = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out, kl2 = self.conv2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out, (kl1 + kl2)

# Bayesian DenseBlock: 여러 DenseLayer를 연결하며 각 층의 출력을 concatenate
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate,
                 prior_mean, prior_variance, posterior_mu_init, posterior_rho_init):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate, growth_rate, bn_size, drop_rate,
                prior_mean, prior_variance, posterior_mu_init, posterior_rho_init
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        kl_sum = 0
        for layer in self.layers:
            new_feature, kl = layer(torch.cat(features, 1))
            kl_sum += kl
            features.append(new_feature)
        return torch.cat(features, 1), kl_sum

# Bayesian Transition Layer: 1x1 conv (채널 압축) + 평균 풀링 (다운샘플링)
class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels,
                 prior_mean, prior_variance, posterior_mu_init, posterior_rho_init):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = Conv2dReparameterization(
            in_channels, out_channels, kernel_size=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        out, kl = self.conv(x)
        out = self.pool(out)
        return out, kl

# Bayesian DenseNet-BC-30 (3개의 DenseBlock, 각 블록에 10개의 DenseLayer → 총 30 layer)
class DenseNet_BC_30_uni(nn.Module):
    def __init__(self, growth_rate=8, block_config=(10, 10, 10),
                 num_init_features=16, bn_size=4, drop_rate=0, num_classes=10,
                 prior_mean=0.0, prior_variance=1.0,
                 posterior_mu_init=0.0, posterior_rho_init=-3.0):
        """
        Args:
            growth_rate (int): 각 DenseLayer가 추가하는 채널 수.
            block_config (tuple): 각 DenseBlock 내 DenseLayer의 수 (예: (10, 10, 10) → 총 30 layer).
            num_init_features (int): 초기 convolution layer의 출력 채널 수.
            bn_size (int): bottleneck 내부에서 1x1 conv의 출력 채널 수 = bn_size * growth_rate.
            drop_rate (float): dropout 확률.
            num_classes (int): 분류할 클래스 수.
            prior_mean (float): Bayesian layer의 prior 평균.
            prior_variance (float): Bayesian layer의 prior 분산.
            posterior_mu_init (float): posterior 평균 초기값.
            posterior_rho_init (float): posterior rho 초기값.
        """
        super(DenseNet_BC_30_uni, self).__init__()
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # 초기 Bayesian convolution layer
        self.features = nn.Sequential(
            Conv2dReparameterization(
                in_channels=3,
                out_channels=num_init_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            )
        )
        num_features = num_init_features

        # Dense Block 1
        self.denseblock1 = _DenseBlock(
            num_layers=block_config[0],
            in_channels=num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            drop_rate=drop_rate,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )
        num_features = num_features + block_config[0] * growth_rate
        self.transition1 = _Transition(
            in_channels=num_features,
            out_channels=int(num_features * 0.5),
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )
        num_features = int(num_features * 0.5)

        # Dense Block 2
        self.denseblock2 = _DenseBlock(
            num_layers=block_config[1],
            in_channels=num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            drop_rate=drop_rate,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )
        num_features = num_features + block_config[1] * growth_rate
        self.transition2 = _Transition(
            in_channels=num_features,
            out_channels=int(num_features * 0.5),
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )
        num_features = int(num_features * 0.5)

        # Dense Block 3
        self.denseblock3 = _DenseBlock(
            num_layers=block_config[2],
            in_channels=num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            drop_rate=drop_rate,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )
        num_features = num_features + block_config[2] * growth_rate

        self.norm_final = nn.BatchNorm2d(num_features)
        self.classifier = LinearReparameterization(
            in_features=num_features,
            out_features=num_classes,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )

    def forward(self, x):
        kl_sum = 0
        # 초기 Bayesian conv layer
        out, kl = self.features[0](x)
        kl_sum += kl

        # Dense Block 1 및 Transition 1
        out, kl_block = self.denseblock1(out)
        kl_sum += kl_block
        out, kl_trans = self.transition1(out)
        kl_sum += kl_trans

        # Dense Block 2 및 Transition 2
        out, kl_block = self.denseblock2(out)
        kl_sum += kl_block
        out, kl_trans = self.transition2(out)
        kl_sum += kl_trans

        # Dense Block 3
        out, kl_block = self.denseblock3(out)
        kl_sum += kl_block

        out = self.norm_final(out)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out, kl_lin = self.classifier(out)
        kl_sum += kl_lin

        return out, kl_sum

# 헬퍼 함수: Bayesian DenseNet-BC-30 모델 생성
def densenet_bc_30_uni(num_classes=10,
                       growth_rate=8,
                       block_config=(10, 10, 10),
                       num_init_features=16,
                       bn_size=4,
                       drop_rate=0,
                       prior_mean=0.0,
                       prior_variance=1.0,
                       posterior_mu_init=0.0,
                       posterior_rho_init=-3.0):
    return DenseNet_BC_30_uni(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=num_init_features,
        bn_size=bn_size,
        drop_rate=drop_rate,
        num_classes=num_classes,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init
    )

# 사용 예시
if __name__ == "__main__":
    # Bayesian DenseNet-BC-30 BNN 모델 생성 (CIFAR-10, 10 클래스)
    model = densenet_bc_30_uni(num_classes=10)
    print(model)
    x = torch.randn(1, 3, 32, 32)  # CIFAR-10 크기의 임의 입력
    output, kl_total = model(x)
    
    # Check the number of parameters
    print(f"The number of parameters: {(sum(p.numel() for p in model.parameters())):,}")
    print("Output shape:", output.shape)
    print("Total KL divergence:", kl_total)
