import torch
import torch.nn as nn
from typing import Type
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Attention as TimmAttention
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization


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
        use_plain_qk: bool = False,  # << NEW
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # selectively replace q, k with plain Linear
        if use_plain_qk:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = LinearReparameterization(dim, dim, bias=qkv_bias)
            self.k = LinearReparameterization(dim, dim, bias=qkv_bias)

        self.v = LinearReparameterization(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearReparameterization(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kl = 0
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v, kld = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v, kld = self.v(x)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x, kld = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViT_Tiny_uni(nn.Module):
    def __init__(self, num_classes=10):
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

        self.base_model = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=192 // 2,
            depth=6 // 2,
            num_heads=3,
            mlp_ratio=4.0,
            num_classes=num_classes
        )

        num_ftrs = self.base_model.head.in_features
        self.base_model.head = nn.Linear(num_ftrs, num_classes)

        # Apply BNN conversion first
        dnn_to_bnn(self.base_model, const_bnn_prior_parameters)

        def _replace_attn(module: nn.Module):
            for name, child in list(module.named_children()):
                if isinstance(child, TimmAttention) or isinstance(child, Attention):
                    # safer way to extract dim
                    try:
                        parent_block = module  # usually Block
                        dim = parent_block.norm1.normalized_shape[0]
                    except AttributeError:
                        print(f"Could not extract 'dim' from {module}. Skipping.")
                        continue

                    num_heads = child.num_heads if hasattr(child, 'num_heads') else 3
                    qkv_bias = True
                    proj_bias = True
                    attn_p = child.attn_drop.p if hasattr(child.attn_drop, 'p') else 0.0
                    proj_p = child.proj_drop.p if hasattr(child.proj_drop, 'p') else 0.0
                    qk_norm = not isinstance(child.q_norm, nn.Identity)
                    norm_layer = type(child.q_norm) if qk_norm else nn.LayerNorm

                    new_attn = Attention(
                        dim=dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        proj_bias=proj_bias,
                        attn_drop=attn_p,
                        proj_drop=proj_p,
                        norm_layer=norm_layer,
                        use_plain_qk=True
                    )

                    setattr(module, name, new_attn)
                else:
                    _replace_attn(child)


        _replace_attn(self.base_model)

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
