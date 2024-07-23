from enum import Enum
from typing import Union

from mindspore import Tensor


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


FormatT = Union[str, Format]


def nchw_to(x: Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(start_dim=2).swapdims(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(start_dim=2)
    return x
