import numbers
from typing import Tuple

import numpy as np
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common import initializer as init

from cambrian.timm.layers.fast_norm import rms_norm


class RmsNorm(nn.Cell):
    """ RmsNorm w/ fast (apex) norm if available
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, channels, eps=1e-6, affine=True, dtype=None) -> None:
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        if self.elementwise_affine:
            self.weight = Parameter(Tensor(np.zeros(self.normalized_shape), **factory_kwargs), name='weight')
        else:
            self.weight = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            self.weight.set_data(init.initializer(1.0, self.weight.shape, self.weight.dtype))

    def construct(self, x: Tensor) -> Tensor:
        # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
        # Since there is no built-in PyTorch impl, always use APEX RmsNorm if is installed.
        x = rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return x


class FrozenBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(FrozenBatchNorm2d, self).__init__(*args, **kwargs)

        self.moving_mean.requires_grad = False
        self.moving_variance.requires_grad = False
        self.gamma.requires_grad = False
        self.beta.requires_grad = False
