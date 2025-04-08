import torch
import torch.nn as nn
import torchvision
from torchvision.models import MobileNet_V2_Weights # MobileNetV2 가중치 import

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
        # MobileNetV2 가중치 로드
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.base_model = torchvision.models.mobilenet_v2(weights=weights)

        if freeze_backbone:
            # 특징 추출기(backbone)의 파라미터 동결
            for param in self.base_model.features.parameters(): # MobileNetV2는 'features' 부분 동결
                param.requires_grad = False

        # MobileNetV2의 마지막 분류 레이어 ('classifier') 수정
        # MobileNetV2의 분류기는 보통 Sequential([Dropout, Linear]) 형태
        num_ftrs = self.base_model.classifier[1].in_features # 마지막 Linear 레이어의 입력 특징 수 가져오기
        # 마지막 Linear 레이어를 새로운 클래스 수에 맞게 교체
        self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":

    model_class_based_mobilenet = MobileNetV2_dnn(num_classes=150, freeze_backbone=True) # 다른 num_classes와 freeze 설정 예시
    print("\nModel created using class structure (MobileNetV2):")
    # MobileNetV2는 마지막 레이어가 'classifier' 내부에 있음
    print(model_class_based_mobilenet.base_model.classifier)