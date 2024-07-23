from typing import Callable, Tuple, Type, Union

from mindspore import nn


LayerType = Union[str, Callable, Type[nn.Cell]]
PadType = Union[str, int, Tuple[int, int]]
