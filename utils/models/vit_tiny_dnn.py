import torch
import torch.nn as nn
from typing import Type
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Attention as TimmAttention
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from utils.models.dynamic_tanh import DynamicTanh, convert_gelu_to_relu, convert_ln_to_dyt, convert_ln_to_rms
class ViT_Tiny_dnn(nn.Module):
    def __init__(self, num_classes=100, norm = 'layernorm', model = 'original'):
        super().__init__()
        
        # load ViT backbone
        if model == 'original':
            self.base_model = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=192 // 2,
                depth=6 // 2,
                num_heads=3,
                mlp_ratio=4.0,
                num_classes=num_classes
            )
        elif model == 'nano':
            self.base_model = VisionTransformer(
                img_size=32,
                patch_size=8,       # 큰 패치로 시퀀스 길이를 16으로 단축
                embed_dim=96,       # 임베딩 차원을 낮게 설정
                depth=4,            # 얕은 깊이
                num_heads=3,        # 헤드 수 감소
                mlp_ratio=2.0,      # MLP 확장 비율 축소 (기본값 4.0)
                num_classes=num_classes
            )
        elif model == 'micro':
            self.base_model = VisionTransformer(
                img_size=32,
                patch_size=4,       # 작은 패치로 더 많은 특징 학습 (시퀀스 길이 64)
                embed_dim=192,      # 표준적인 경량 모델의 임베딩 차원
                depth=6,            # 중간 수준의 깊이
                num_heads=3,        # 임베딩 차원에 맞춰 헤드 수 조절
                mlp_ratio=2.0,      # MLP 확장 비율 축소
                num_classes=num_classes
            )
        elif model == 'pico':
            self.base_model = VisionTransformer(
        img_size=32,
        patch_size=4,
        embed_dim=256,      # 표현력을 위해 임베딩 차원 확장
        depth=7,            # 모델 깊이 추가
        num_heads=4,        # 확장된 임베딩 차원에 맞춰 헤드 수 증가
        mlp_ratio=3.0,      # 성능을 위해 MLP 비율을 소폭 상향
        num_classes=100
    )
        # replace head
        num_ftrs = self.base_model.head.in_features
        self.base_model.head = nn.Linear(num_ftrs, num_classes)
       
        if norm == 'dyt':
            convert_ln_to_dyt(self.base_model)
            print("Converted LayerNorm to DynamicTanh")

    def forward(self, x):
        out = self.base_model(x)
        return out

# usage
if __name__ == '__main__':
    model = ViT_Tiny_dnn(num_classes=10)
    print(model)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32)
        prediction = model(dummy_input)
        print("Prediction shape:", prediction.shape)


'''
기존에 시도했던 모델 구조
# load ViT backbone 
        self.base_model = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=192//2,
            depth=6//2,
            num_heads=3,
            mlp_ratio=4.0,
            num_classes=num_classes
        )

ViT-Nano
VisionTransformer(
                img_size=32,
                patch_size=8,       # 큰 패치로 시퀀스 길이를 16으로 단축
                embed_dim=96,       # 임베딩 차원을 낮게 설정
                depth=4,            # 얕은 깊이
                num_heads=3,        # 헤드 수 감소
                mlp_ratio=2.0,      # MLP 확장 비율 축소 (기본값 4.0)
                num_classes=100
            )
            
ViT-MiCro
VisionTransformer(
    img_size=32,
    patch_size=4,       # 작은 패치로 더 많은 특징 학습 (시퀀스 길이 64)
    embed_dim=192,      # 표준적인 경량 모델의 임베딩 차원
    depth=6,            # 중간 수준의 깊이
    num_heads=3,        # 임베딩 차원에 맞춰 헤드 수 조절
    mlp_ratio=2.0,      # MLP 확장 비율 축소
    num_classes=100
)

ViT-Pico
VisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=256,      # 표현력을 위해 임베딩 차원 확장
    depth=7,            # 모델 깊이 추가
    num_heads=4,        # 확장된 임베딩 차원에 맞춰 헤드 수 증가
    mlp_ratio=3.0,      # 성능을 위해 MLP 비율을 소폭 상향
    num_classes=100
)
'''