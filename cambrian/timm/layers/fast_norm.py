from typing import Tuple, Optional

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter


def rms_norm(
    x: Tensor,
    normalized_shape: Tuple[int],
    weight: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    _dtype = x.dtype
    x = x.to(ms.float32)

    norm_ndim = len(normalized_shape)

    dims = tuple(range(-1, -norm_ndim - 1, -1))
    v = ops.var(x, axis=dims, keepdims=True)

    x = x * ops.rsqrt(v + eps)
    if weight is not None:
        x = x * weight.to(x.dtype)

    x = x.to(_dtype)
    return x
