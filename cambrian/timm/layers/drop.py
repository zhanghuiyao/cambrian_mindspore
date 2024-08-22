import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, constexpr


@constexpr(reuse_result=False)
def _bernoulli(shape, p):
    _uniform_samples = np.random.uniform(0., 1., shape)
    bernoulli_tensor = np.ones(shape)
    bernoulli_tensor[_uniform_samples > p] = 0.
    return Tensor(bernoulli_tensor, ms.float32)


def _bak_bernoulli(shape, p):
    _uniform_samples = ops.uniform(shape, ops.zeros((), ms.float32), ops.ones((), ms.float32))
    bernoulli_tensor = ops.select(
        ops.less_equal(_uniform_samples, p),
        ops.ones((), ms.float32),
        ops.zeros((), ms.float32)
    )

    return bernoulli_tensor


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    # FIXME: `ops.bernoulli` has bug on MindSpore 2.3.0 + jit_level O2
    # random_tensor = ops.bernoulli(ops.zeros(shape), p=keep_prob)
    random_tensor = _bernoulli(shape, p=keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div(keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
