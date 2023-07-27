import torch
import torch.nn as nn

from .vit import Attention
import networks.vit_utils as utils


class MaskClassifierLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., output_size=1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = utils.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = utils.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=output_size, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskClassifier(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., output_size=1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.layer_1 = MaskClassifierLayer(
        #     dim=dim,
        #     num_heads=num_heads,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     drop=drop,
        #     attn_drop=attn_drop,
        #     output_size=dim,
        #     drop_path=drop_path,
        #     act_layer=act_layer,
        #     norm_layer=norm_layer
        # )

        self.layer_2 = MaskClassifierLayer(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            output_size=output_size,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

    def forward(self, x):
        # x = self.layer_1(x)
        x = self.layer_2(x)
        return x
