import torch
import torch.nn as nn
import sys
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization


class SimpleFCNN_UNI(nn.Module):
    """Bayesian version of ``SimpleFCNN`` (UNI prior on weights). Returns (logits, KL)."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_units: int = 100,
        num_classes: int = 10,
        *,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
    ):
        super().__init__()
        dims = [input_dim, hidden_units, hidden_units, num_classes]
        self.layers = nn.ModuleList(
            [
                LinearReparameterization(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    bias=True,
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                )
                for i in range(len(dims) - 1)
            ]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        kl_sum = 0.0
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x, kl = layer(x)
            kl_sum += kl
            if i < len(self.layers) - 1:
                x = self.relu(x)
        return x, kl_sum
    

if __name__ == "__main__":
    import torch

    # sanity check
    dummy = torch.randn(1, 1, 28, 28)

    print("\n=== Bayesian MLP (UNI) ===")
    uni = SimpleFCNN_UNI()
    logits_bnn, kl_total = uni(dummy)
    print(uni)
    print("Output shape:", logits_bnn.shape)
    print("Total KL divergence:", kl_total.item())

    # 추가: 파라미터 개수 출력
    total_params = sum(p.numel() for p in uni.parameters())
    trainable_params = sum(p.numel() for p in uni.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
