from typing import Union, Type
from functools import partial

from mindspore import nn, ops

from cambrian.timm.layers.activations import HardMish


_ACT_LAYER = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SeLU,
    gelu=partial(nn.GELU, approximate=False),
    gelu_tanh=partial(nn.GELU, approximate=True),
    quick_gelu=nn.FastGelu,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    hard_sigmoid=nn.HSigmoid,
    hard_swish=nn.HSwish,
    hard_mish=HardMish,
    identity=nn.Identity,
)


def get_act_layer(name: Union[Type[nn.Cell], str] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    if not name:
        return None

    return _ACT_LAYER[name]
