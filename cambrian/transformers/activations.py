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


ACT2CLS = {
    "gelu": partial(nn.GELU, approximate=False),
    "quick_gelu": partial(nn.GELU, approximate=False),
    "leaky_relu": nn.LeakyReLU,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLU32,
    "swish": SiLU32,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)
