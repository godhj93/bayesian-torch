import torch
import torch.nn as nn
import torchvision
from torchvision.models import MobileNet_V2_Weights
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

class MobileNetV2_uni(nn.Module):
    """
    MobileNetV2 DNN 모델 (사전 훈련 가중치 사용 & Bayesian 처리).
    
    Args:
        num_classes (int): 분류할 클래스 수.
        freeze_backbone (bool): 특징 추출기 동결 여부.
    """
    def __init__(self, num_classes=100, freeze_backbone=False):
        super().__init__()
        
        # BNN 변환 시 사용될 사전 정의 파라미터
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",  # Flipout 또는 Reparameterization 선택
            "moped_enable": False,         # True일 경우, 사전 훈련된 dnn 가중치로부터 mu/sigma를 초기화
            "moped_delta": 0.5,
        }

        print("Using ImageNet1K Pretrained MobileNetV2")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.base_model = torchvision.models.mobilenet_v2(weights=weights)
        
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # MobileNetV2의 기본 classifier는 nn.Sequential로 구성되어 있고,
        # 두 번째 계층이 실제 Linear 분류 계층임.
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        # DNN -> BNN 변환 적용
        dnn_to_bnn(self.base_model, const_bnn_prior_parameters)
        
    def forward(self, x):
        output = self.base_model(x)
        kl = get_kl_loss(self.base_model)
        return output, kl


if __name__ == "__main__":
    # 클래스 사용 예시
    model_class_based = MobileNetV2_uni(num_classes=100, freeze_backbone=False)
    print("\nModel created using class structure:")
    print(model_class_based.base_model.classifier[1])
    
    # 더미 입력을 통한 테스트
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        prediction, kl = model_class_based(dummy_input)
        print("\nPrediction shape:", prediction.shape)
        print("KL divergence :", kl)
