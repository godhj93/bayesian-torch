import torch
import torch.nn as nn
from typing import Type
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Attention as TimmAttention
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # separate linear projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViT_Tiny_dnn(nn.Module):
    def __init__(self, num_classes=10):
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
       
        # function to replace attention
        def _replace_attn(module: nn.Module):
            for name, child in list(module.named_children()):
                if isinstance(child, TimmAttention):
                    # extract config
                    dim       = child.qkv.in_features
                    num_heads = child.num_heads
                    qkv_bias  = child.qkv.bias is not None
                    proj_bias = child.proj.bias is not None
                    attn_p    = child.attn_drop.p if hasattr(child.attn_drop, 'p') else 0.0
                    proj_p    = child.proj_drop.p if hasattr(child.proj_drop, 'p') else 0.0
                    qk_norm   = not isinstance(child.q_norm, nn.Identity)
                    norm_layer = type(child.q_norm) if qk_norm else nn.LayerNorm
                    # build custom
                    new_attn = Attention(
                        dim=dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        proj_bias=proj_bias,
                        attn_drop=attn_p,
                        proj_drop=proj_p,
                        norm_layer=norm_layer
                    )
                    # copy weights
                    qkv_w = child.qkv.weight.data
                    if qkv_bias:
                        qkv_b = child.qkv.bias.data
                    d = dim
                    new_attn.q.weight.data.copy_(qkv_w[:d, :])
                    new_attn.k.weight.data.copy_(qkv_w[d:2*d, :])
                    new_attn.v.weight.data.copy_(qkv_w[2*d:3*d, :])
                    if qkv_bias:
                        new_attn.q.bias.data.copy_(qkv_b[:d])
                        new_attn.k.bias.data.copy_(qkv_b[d:2*d])
                        new_attn.v.bias.data.copy_(qkv_b[2*d:3*d])
                    # copy proj
                    new_attn.proj.weight.data.copy_(child.proj.weight.data)
                    new_attn.proj.bias.data.copy_(child.proj.bias.data)
                    setattr(module, name, new_attn)
                else:
                    _replace_attn(child)
        # apply replacement
        _replace_attn(self.base_model)
        
      

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
