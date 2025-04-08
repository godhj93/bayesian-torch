import torch
import torch.nn as nn
import torchvision
from torchvision.models import MobileNet_V2_Weights

class MobileNetV2_dnn(nn.Module):
    """
    MobileNetV2 DNN 모델 (사전 훈련 가중치 사용).

    Args:
        num_classes (int): 분류할 클래스 수.
        freeze_backbone (bool): 특징 추출기 동결 여부.
    """
    def __init__(self, num_classes=100, freeze_backbone=False):
        super().__init__()
        
        print("Using ImageNet1K Pretrained MobileNetV2")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.base_model = torchvision.models.mobilenet_v2(weights=weights)

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # MobileNetV2의 기본 classifier는 nn.Sequential으로 구성되어 있으며,
        # 마지막 Linear 계층에서 in_features를 가져올 수 있습니다.
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

if __name__ == "__main__":
    # 클래스 사용 예시
    model_class_based = MobileNetV2_dnn(num_classes=100, freeze_backbone=False)
    print("\nModel created using class structure:")
    print(model_class_based.base_model.classifier[1])
