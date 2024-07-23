import functools
import types
from typing import Type

from mindspore import nn, ops

from cambrian.timm.layers import LayerNorm, LayerNorm2d, RmsNorm, FrozenBatchNorm2d


_NORM_MAP = dict(
    batchnorm=nn.BatchNorm2d,
    batchnorm2d=nn.BatchNorm2d,
    batchnorm1d=nn.BatchNorm1d,
    groupnorm=nn.GroupNorm,
    groupnorm1=nn.GroupNorm,
    layernorm=LayerNorm,
    layernorm2d=LayerNorm2d,
    rmsnorm=RmsNorm,
    frozenbatchnorm2d=FrozenBatchNorm2d,
)
_NORM_TYPES = {m for n, m in _NORM_MAP.items()}


def get_norm_layer(norm_layer):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    norm_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace('_', '')
        norm_layer = _NORM_MAP[layer_name]
    else:
        norm_layer = norm_layer

    if norm_kwargs:
        norm_layer = functools.partial(norm_layer, **norm_kwargs)  # bind/rebind args
    return norm_layer
