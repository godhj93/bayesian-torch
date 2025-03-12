import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    """
    MobileNetV2의 Inverted Residual Block.
    
    Args:
        inp (int): 입력 채널 수.
        oup (int): 출력 채널 수.
        stride (int): stride (1 또는 2).
        expand_ratio (int): 확장 비율 (t).
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be 1 or 2"
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)
        
        layers = []
        if expand_ratio != 1:
            # 확장 단계: 1x1 conv
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # 깊이별(depthwise) 3x3 conv
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # 프로젝션 단계: 1x1 conv (linear bottleneck)
        layers.append(nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_dnn(nn.Module):
    """
    MobileNetV2 DNN 모델.
    
    Args:
        num_classes (int): 분류할 클래스 수.
        width_mult (float): 채널 수에 곱해지는 계수. (기본 1.0)
    """
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2_dnn, self).__init__()
        # inverted residual block 설정: [expansion factor, output channels, number of blocks, stride]
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # 첫 번째 conv layer (3x3)
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        
        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]
        
        # Inverted Residual blocks 생성
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        # 마지막 conv layer: 1x1 conv
        features.append(nn.Conv2d(input_channel, self.last_channel, kernel_size=1, bias=False))
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(self.last_channel, num_classes)
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        # global average pooling
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

if __name__ == "__main__":
    model = MobileNetV2_dnn(num_classes=10, width_mult=1.0)
    print(model)
    x = torch.randn(8, 3, 32, 32)  # CIFAR-10 크기 (32x32) 입력 (단, 원래 MobileNetV2는 큰 이미지에서 동작)
    output = model(x)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Output shape:", output.shape)
