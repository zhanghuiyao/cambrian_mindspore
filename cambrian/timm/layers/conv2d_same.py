import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from typing import Optional, Tuple

from cambrian.timm.layers.padding import pad_same, get_padding_value


def conv2d_same(
        x,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
):
    x = pad_same(x, weight.shape[-2:], stride, dilation)

    return ops.conv2d(
        input=x,
        weight=weight,
        bias=bias,
        stride=stride,
        pad_mode="valid",
        padding=0,
        dilation=dilation,
        groups=groups
    )


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        assert padding == 0
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride,
            pad_mode='valid',
            padding=0,
            dilation=dilation,
            group=groups,
            has_bias=bias,
        )

    def construct(self, x):
        return conv2d_same(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if "bias" in kwargs:
            _bias = kwargs.pop("bias")
            kwargs["has_bias"] = _bias
        else:
            kwargs["has_bias"] = True

        if "groups" in kwargs:
            _groups = kwargs.pop("groups")
            kwargs["group"] = _groups

        return nn.Conv2d(in_chs, out_chs, kernel_size, pad_mode='pad', padding=padding, **kwargs)
