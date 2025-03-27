import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import LinearReparameterization, Conv2dReparameterization

class LeNet5_uni(nn.Module):
    def __init__(self, num_classes=10,
                 prior_mean=0.0, prior_variance=1.0,
                 posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super(LeNet5_uni, self).__init__()

        self.conv1 = Conv2dReparameterization(
            in_channels=3, out_channels=6, kernel_size=5,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2dReparameterization(
            in_channels=6, out_channels=16, kernel_size=5,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = LinearReparameterization(
            in_features=16 * 5 * 5, out_features=120,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.fc2 = LinearReparameterization(
            in_features=120, out_features=84,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.fc3 = LinearReparameterization(
            in_features=84, out_features=num_classes,
            prior_mean=prior_mean, prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )

    def forward(self, x):
        kl_sum = 0

        x, kl = self.conv1(x)
        kl_sum += kl
        x = self.pool1(F.relu(x))

        x, kl = self.conv2(x)
        kl_sum += kl
        x = self.pool2(F.relu(x))

        x = x.view(x.size(0), -1)  # Flatten

        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)

        x, kl = self.fc2(x)
        kl_sum += kl
        x = F.relu(x)

        x, kl = self.fc3(x)
        kl_sum += kl

        return x, kl_sum

# 테스트용 예제
if __name__ == "__main__":
    model = BayesianLeNet5(num_classes=10)
    print(model)
    dummy_input = torch.randn(1, 3, 32, 32)
    output, kl_total = model(dummy_input)
    print("Output shape:", output.shape)
    print("Total KL divergence:", kl_total)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
