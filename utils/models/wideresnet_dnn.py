import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Public API -------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Export named WideResNet variants when using `from wide_resnet import *`.
__all__ = [
    # depth 10
    "wrn10_1", "wrn10_2", "wrn10_4",
    # depth 16
    "wrn16_1", "wrn16_2", "wrn16_4",
    # depth 22
    "wrn22_1", "wrn22_2", "wrn22_4",
]

# -----------------------------------------------------------------------------
# Helper: 3×3 Conv --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3×3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


# -----------------------------------------------------------------------------
# WideResNet basic residual block ---------------------------------------------
# -----------------------------------------------------------------------------

class WideBasic(nn.Module):
    """Basic residual block used in WideResNet (two 3×3 convolutions)."""

    def __init__(
        self,
        in_planes: int,
        planes: int,
        dropout_rate: float,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = _conv3x3(in_planes, planes)
        self.dropout = (
            nn.Dropout(p=dropout_rate, inplace=True) if dropout_rate > 0 else nn.Identity()
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=stride)

        # Projection shortcut when resolution or channel dims change
        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


# -----------------------------------------------------------------------------
# WideResNet main class --------------------------------------------------------
# -----------------------------------------------------------------------------

class WideResNet(nn.Module):
    """Wide Residual Network (Zagoruyko & Komodakis, 2016).

    Args:
        depth (int): Total network depth. Must satisfy (depth - 4) % 6 == 0.
        widen_factor (int): Channel multiplier *k*.
        dropout_rate (float): Dropout probability inside WideBasic blocks.
        num_classes (int): Output class count.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if (depth - 4) % 6 != 0:
            raise ValueError("Depth must be 6n+4 (e.g., 10, 16, 22, 28, 40 …)")
        n = (depth - 4) // 6  # number of blocks per stage
        k = widen_factor

        # Base channels per stage for CIFAR-sized inputs: 16, 32, 64
        stages: Tuple[int, int, int] = (16 * k, 32 * k, 64 * k)

        self.in_planes = 16
        self.conv1 = _conv3x3(3, self.in_planes)

        self.layer1 = self._make_layer(stages[0], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(stages[1], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(stages[2], n, dropout_rate, stride=2)

        self.bn = nn.BatchNorm2d(stages[2])
        self.fc = nn.Linear(stages[2], num_classes)

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Stage builder
    # ------------------------------------------------------------------
    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasic(self.in_planes, planes, dropout_rate, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size(3))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# -----------------------------------------------------------------------------
# Internal generic factory -----------------------------------------------------
# -----------------------------------------------------------------------------

def _wideresnet(
    depth: int,
    widen_factor: int,
    dropout: float,
    num_classes: int,
) -> WideResNet:
    return WideResNet(depth, widen_factor, dropout, num_classes)


# -----------------------------------------------------------------------------
# Auto-generate convenience constructors ---------------------------------------
# -----------------------------------------------------------------------------
# We programmatically create functions wrn<depth>_<k> to avoid boilerplate.


def _make_constructor(depth: int, k: int):
    """Return a constructor function for WideResNet-depth-k."""

    def _fn(num_classes: int = 10, dropout: float = 0.0) -> WideResNet:  # noqa: D401
        return _wideresnet(depth, k, dropout, num_classes)

    _fn.__name__ = f"wrn{depth}_{k}"
    _fn.__qualname__ = _fn.__name__
    _fn.__doc__ = f"WideResNet-{depth}-{k}: depth={depth}, k={k}"
    return _fn


# Depths and widen factors we want to expose
for _d in (10, 16, 22):
    for _k in (1, 2, 4):
        globals()[f"wrn{_d}_{_k}"] = _make_constructor(_d, _k)


# -----------------------------------------------------------------------------
# Quick test (run as script) ----------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for name in __all__:
        fn = globals()[name]
        model = fn(num_classes=100)
        
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        # the number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {num_params:,} parameters")
        print(f"{name}: output shape", y.shape)
