import re

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from cambrian.model.multimodal_projector.projectors import CAbstractor


class IdentityMap(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Cell):
    def __init__(self, channels):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels]
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.SequentialCell([
            nn.Dense(channels, channels),
            nn.GELU(),
            nn.Dense(channels, channels)
        ])

    def construct(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    

class SEMLP(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.se = nn.SequentialCell([
            nn.Dense(in_channels, in_channels, has_bias=False),
            nn.GELU(),
            nn.Dense(in_channels, in_channels, has_bias=False),
            nn.Sigmoid()
        ])

        self.proj = nn.SequentialCell([
            nn.Dense(in_channels, out_channels),
            nn.GELU(),
            nn.Dense(out_channels, out_channels)
        ])

    def construct(self, x):
        global_x = ops.mean(x, 1, keep_dims=True)
        weight = self.se(global_x)
        x = x * weight + x
        return self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Dense(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Dense(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Dense(config.hidden_size, config.hidden_size))
        return nn.SequentialCell(modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    if projector_type == 'se_mlp':
        return SEMLP(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == 'CAbstractor':
        return CAbstractor(config.mm_hidden_size, config.hidden_size)

    raise ValueError(f'Unknown projector type: {projector_type}')
