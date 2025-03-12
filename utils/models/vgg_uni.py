import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers import Conv2dReparameterization, LinearReparameterization

class VGG7_uni(nn.Module):
    def __init__(self, num_classes=10,
                 prior_mean=0.0, prior_variance=1.0,
                 posterior_mu_init=0.0, posterior_rho_init=-3.0):
        """
        Bayesian VGG7 모델 (BNN)
        구조:
          - Block1: [Conv2d, ReLU, Conv2d, ReLU, MaxPool]
          - Block2: [Conv2d, ReLU, Conv2d, ReLU, MaxPool]
          - Block3: [Conv2d, ReLU]
          - Classifier: [Flatten, Linear, ReLU, Linear]
          
        각 Bayesian 계층은 KL divergence 값을 함께 반환합니다.
        
        Args:
            num_classes (int): 분류할 클래스 수 (기본: 10, CIFAR-10 기준)
            prior_mean (float): 각 Bayesian layer의 prior 평균
            prior_variance (float): 각 Bayesian layer의 prior 분산
            posterior_mu_init (float): 각 Bayesian layer의 posterior 평균 초기값
            posterior_rho_init (float): 각 Bayesian layer의 posterior rho 초기값
        """
        super(VGG7_uni, self).__init__()
        
        # Block 1
        self.conv1 = Conv2dReparameterization(
            in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.conv2 = Conv2dReparameterization(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv3 = Conv2dReparameterization(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.conv4 = Conv2dReparameterization(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv5 = Conv2dReparameterization(
            in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        
        # Classifier 부분
        # CIFAR-10 입력(32x32) -> Block1: 32->16, Block2: 16->8, Block3: 8->8, 채널: 256
        self.fc1 = LinearReparameterization(
            in_features=256 * 8 * 8, out_features=512, bias=True,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.fc2 = LinearReparameterization(
            in_features=512, out_features=num_classes, bias=True,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        kl_sum = 0
        
        # Block 1
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.relu(out)
        out, kl = self.conv2(out)
        kl_sum += kl
        out = self.relu(out)
        out = self.pool1(out)
        
        # Block 2
        out, kl = self.conv3(out)
        kl_sum += kl
        out = self.relu(out)
        out, kl = self.conv4(out)
        kl_sum += kl
        out = self.relu(out)
        out = self.pool2(out)
        
        # Block 3
        out, kl = self.conv5(out)
        kl_sum += kl
        out = self.relu(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Classifier
        out, kl = self.fc1(out)
        kl_sum += kl
        out = self.relu(out)
        out, kl = self.fc2(out)
        kl_sum += kl
        
        return out, kl_sum

if __name__ == "__main__":
    # Bayesian VGG7 모델 생성 (CIFAR-10 기준)
    model = VGG7_uni(num_classes=10)
    print(model)
    
    # CIFAR-10 크기 입력 (1, 3, 32, 32)
    x = torch.randn(1, 3, 32, 32)
    output, kl_total = model(x)
    
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Output shape:", output.shape)
    print("Total KL divergence:", kl_total)
