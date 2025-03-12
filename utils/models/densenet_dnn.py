import torch
import torch.nn as nn
import torch.nn.functional as F

# DenseLayer (Bottleneck 구조)
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # Bottleneck: 1x1 conv, 출력 채널: bn_size * growth_rate
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        # 3x3 conv, 출력 채널: growth_rate, padding=1로 spatial size 유지
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

# DenseBlock: 여러 DenseLayer를 연결하며, 각 단계의 출력을 concatenate
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

# Transition Layer: 채널 수 압축과 다운샘플링을 수행
class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x

# DenseNet-BC-30 모델 (3개의 DenseBlock, 각 블록에 10개의 DenseLayer)
class DenseNet_BC_30(nn.Module):
    def __init__(self, growth_rate=8, block_config=(10, 10, 10),
                 num_init_features=16, bn_size=4, drop_rate=0, num_classes=10):
        """
        Args:
            growth_rate (int): 각 DenseLayer가 추가하는 채널 수 (여기서는 8)
            block_config (tuple of int): 각 DenseBlock 내 DenseLayer의 수 (예: (10, 10, 10) → 총 30 layer)
            num_init_features (int): 초기 convolution layer의 출력 채널 수 (여기서는 16)
            bn_size (int): bottleneck 내부에서 1x1 conv의 출력 채널 수 = bn_size * growth_rate
            drop_rate (float): dropout 확률
            num_classes (int): 분류할 클래스 수
        """
        super(DenseNet_BC_30, self).__init__()
        
        # 초기 convolution layer (3x3)
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        )
        num_features = num_init_features

        # Dense Block 1
        self.denseblock1 = _DenseBlock(num_layers=block_config[0],
                                        in_channels=num_features,
                                        growth_rate=growth_rate,
                                        bn_size=bn_size,
                                        drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate
        self.transition1 = _Transition(in_channels=num_features,
                                       out_channels=int(num_features * 0.5))
        num_features = int(num_features * 0.5)

        # Dense Block 2
        self.denseblock2 = _DenseBlock(num_layers=block_config[1],
                                        in_channels=num_features,
                                        growth_rate=growth_rate,
                                        bn_size=bn_size,
                                        drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate
        self.transition2 = _Transition(in_channels=num_features,
                                       out_channels=int(num_features * 0.5))
        num_features = int(num_features * 0.5)

        # Dense Block 3
        self.denseblock3 = _DenseBlock(num_layers=block_config[2],
                                        in_channels=num_features,
                                        growth_rate=growth_rate,
                                        bn_size=bn_size,
                                        drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate

        # 최종 BatchNorm 및 분류기
        self.norm_final = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        x = self.norm_final(x)
        x = F.relu(x, inplace=True)
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 헬퍼 함수: DenseNet-BC-30 모델 생성
def densenet_bc_30(num_classes=10):
    return DenseNet_BC_30(
        growth_rate=8,
        block_config=(10, 10, 10),
        num_init_features=16,
        bn_size=4,
        drop_rate=0,
        num_classes=num_classes
    )

if __name__ == '__main__':
    
    model = densenet_bc_30()
    print(model)
    
    # Check the number of parameters
    print(f"The number of parameters: {(sum(p.numel() for p in model.parameters())):,}")
    
    # run
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.size())