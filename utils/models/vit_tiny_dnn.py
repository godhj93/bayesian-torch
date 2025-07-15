import torch
import torch.nn as nn
from typing import Type
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Attention as TimmAttention
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from utils.models.dynamic_tanh import DynamicTanh, convert_gelu_to_relu, convert_ln_to_dyt, convert_ln_to_rms
class ViT_Tiny_dnn(nn.Module):
    def __init__(self, num_classes=10, norm = 'layernorm'):
        super().__init__()
        # BNN prior settings
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
        self.base_model = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=192//2,
            depth=6//2,
            num_heads=3,
            mlp_ratio=4.0,
            num_classes=num_classes
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
