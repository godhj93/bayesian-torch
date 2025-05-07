import torch
import torch.nn as nn

class SimpleFCNN(nn.Module):
    """Deterministic 3‑layer fully‑connected network with 100 hidden units per layer.

    Paper setup: input (28×28 grayscale) → 100 → ReLU → 100 → ReLU → 100 → ReLU → num_classes.
    The default *input_dim* assumes MNIST (1×28×28); change as needed.
    """

    def __init__(self, input_dim: int = 28 * 28, hidden_units: int = 100, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units), nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(inplace=True),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
    
if __name__ == "__main__":
    import torch

    # sanity check
    dummy = torch.randn(1, 1, 28, 28)

    print("\n=== Deterministic MLP ===")
    dnn = SimpleFCNN()
    logits = dnn(dummy)
    print(dnn)
    print("Output shape:", logits.shape)

    # 파라미터 개수 출력
    total_params = sum(p.numel() for p in dnn.parameters())
    trainable_params = sum(p.numel() for p in dnn.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
