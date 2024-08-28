from functools import partial
from collections import OrderedDict
from mindspore import nn


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": partial(nn.GELU, approximate=False),
    "quick_gelu": partial(nn.GELU, approximate=False),
    "leaky_relu": nn.LeakyReLU,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)
