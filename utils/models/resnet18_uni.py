# import torchvision

# class ResNet18_dnn(torchvision.models.ResNet):
#     """
#     ResNet18 DNN 모델.
    
#     Args:
#         num_classes (int): 분류할 클래스 수.
#     """
#     def __init__(self, num_classes=1000):
#         super(ResNet18_dnn, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        
#     def forward(self, x):
#         return super(ResNet18_dnn, self).forward(x)
    
    
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

class ResNet18_uni(nn.Module):
    """
    ResNet18 DNN 모델 (사전 훈련 가중치 사용).

    Args:
        num_classes (int): 분류할 클래스 수.
        freeze_backbone (bool): 특징 추출기 동결 여부.
    """
    def __init__(self, num_classes=100, freeze_backbone=False):
        super().__init__()
        
        const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    
    
        print("Using ImageNet1K Pretrained ResNet18")
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.base_model = torchvision.models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

        dnn_to_bnn(self.base_model, const_bnn_prior_parameters)
        
    def forward(self, x):
        
        output = self.base_model(x)
        kl = get_kl_loss(self.base_model)
        return output, kl


if __name__ == "__main__":
        
    # 클래스 사용 예시
    model_class_based = ResNet18_uni(num_classes=100, freeze_backbone=False)
    print("\nModel created using class structure:")
    print(model_class_based.base_model)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        
        prediction = model_class_based(dummy_input)
        print("\nPrediction shape:", prediction[0].shape)
        print("KL divergence :", prediction[1])