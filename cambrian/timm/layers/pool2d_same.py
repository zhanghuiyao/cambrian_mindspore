import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from cambrian.timm.layers.helpers import to_2tuple
from cambrian.timm.layers.padding import pad_same


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, pad_mode="valid", padding=0, ceil_mode=False, count_include_pad=True):
        assert padding == 0
        assert pad_mode == "valid"
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)

        super(AvgPool2dSame, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="valid",
            padding=(0, 0),
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

    def construct(self, x):
        x = pad_same(x, self.kernel_size, self.stride)

        return ops.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
