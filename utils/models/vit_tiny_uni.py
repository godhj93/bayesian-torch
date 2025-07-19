import torch
import torch.nn as nn
from typing import Type
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Attention as TimmAttention
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from utils.models.dynamic_tanh import DynamicTanh, convert_gelu_to_relu, convert_ln_to_dyt, convert_ln_to_rms

class ViT_Tiny_uni(nn.Module):
    def __init__(self, num_classes=10, norm = 'layernorm', model = 'original'):
        super().__init__()
        const_bnn_prior_parameters = {
            'prior_mu': 0.0,
            'prior_sigma': 1.0,
            'posterior_mu_init': 0.0,
            'posterior_rho_init': -3.0,
            'type': 'Reparameterization',
            'moped_enable': False,
            'moped_delta': 0.5,
        }

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
        num_classes=num_classes
    )

        num_ftrs = self.base_model.head.in_features
        self.base_model.head = nn.Linear(num_ftrs, num_classes)

        # Apply BNN conversion first
        dnn_to_bnn(self.base_model, const_bnn_prior_parameters)

       
        # Restore original Conv2d in patch embedding
        old_proj = self.base_model.patch_embed.proj
        conv2d_params = {
            "in_channels": old_proj.in_channels,
            "out_channels": old_proj.out_channels,
            "kernel_size": old_proj.kernel_size,
            "stride": old_proj.stride,
            "padding": old_proj.padding,
            "dilation": old_proj.dilation,
            "groups": old_proj.groups,
            "bias": old_proj.bias is not None,
            # "padding_mode": old_proj.padding_mode
        }
        self.base_model.patch_embed.proj = nn.Conv2d(**conv2d_params)
        
        if norm == 'dyt':
            convert_ln_to_dyt(self.base_model)
        
    def forward(self, x):
        out = self.base_model(x)
        kl = get_kl_loss(self.base_model)
        return out, kl


if __name__ == '__main__':
    model = ViT_Tiny_uni(num_classes=10)
    print(model)

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32)
        pred, kl = model(dummy_input)
        print("Prediction shape:", pred.shape)
        print("KL Divergence:", kl.item())
