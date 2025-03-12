import torch
import torch.nn as nn

class VGG7(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG7, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 2개의 Conv 계층 + MaxPooling
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 2개의 Conv 계층 + MaxPooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 1개의 Conv 계층
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 입력 이미지가 32x32일 경우,
        # Block 1: 32 -> 16, Block 2: 16 -> 8, Block 3: 8 -> 8 (변화 없음)
        # 따라서 특징 맵 크기는 256 x 8 x 8가 됨.
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # 모델 생성 및 요약 출력
    model = VGG7(num_classes=10)
    print(model)
    
    # 더미 입력을 통한 출력 테스트
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)
