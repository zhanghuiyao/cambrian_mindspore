from functools import partial
from collections import OrderedDict
from mindspore import nn, ops

import mindspore as ms


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


class SiLU32(nn.Cell):
    def __init__(self):
        super(SiLU32, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        _dtype = x.dtype
        x = x.to(ms.float32)
        out = x * self.sigmoid(x)
        out = out.to(_dtype)
        return out


class QuickGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def __init__(self):
        super(QuickGELUActivation, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, input):
        return input * self.sigmoid(1.702 * input)


class LeakyReLU(nn.Cell):

    def __init__(self, negative_slope: float = 1e-2) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def construct(self, input):
        return ops.leaky_relu(input, self.negative_slope)


ACT2CLS = {
    "gelu": partial(nn.GELU, approximate=False),
    "quick_gelu": QuickGELUActivation,
    "leaky_relu": LeakyReLU,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLU32,
    "swish": SiLU32,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)
