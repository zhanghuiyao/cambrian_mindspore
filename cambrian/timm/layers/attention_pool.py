import numpy as np
from typing import Optional

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from cambrian.timm.layers.mlp import Mlp


class AttentionPoolLatent(nn.Cell):
    """ Attention pooling w/ latent query
    """

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = 'token',
            norm_layer: Optional[nn.Cell] = None,
            drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type

        if pos_embed == 'abs':
            spatial_len = self.feat_size
            self.pos_embed = Parameter(Tensor(np.zeros((spatial_len, in_features)), ms.float32), name='pos_embed')
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = Parameter(Tensor(np.zeros((1, self.latent_len, embed_dim)), ms.float32), name="latent")

        self.q = nn.Dense(embed_dim, embed_dim, has_bias=qkv_bias)
        self.kv = nn.Dense(embed_dim, embed_dim * 2, has_bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Dense(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(p=drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))

        # TODO: init weight
        # self.init_weights()

    def construct(self, x):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = ops.broadcast_to(self.latent, (B, -1, -1))
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).swapdims(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        # attention
        q = q * self.scale
        attn = ops.BatchMatMul()(q, k.swapdims(-2, -1))
        attn = attn.softmax(axis=-1)
        x = ops.BatchMatMul()(attn, v)

        x = x.swapdims(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x
