import collections.abc
import numpy as np
from itertools import repeat

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


class FrozenBatchNorm2d(nn.Cell):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps

        self.weight = Parameter(Tensor(np.ones(num_features), ms.float32), requires_grad=False, name="weight")
        self.bias = Parameter(Tensor(np.zeros(num_features), ms.float32), requires_grad=False, name="bias")
        self.running_mean = Parameter(Tensor(np.zeros(num_features), ms.float32), requires_grad=False, name="running_mean")
        self.running_var = Parameter(Tensor(np.ones(num_features), ms.float32), requires_grad=False, name="running_var")

    def construct(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        ori_dtype = x.dtype
        x = x.to(ms.float32)

        w = self.weight.reshape(1, -1, 1, 1).to(ms.float32)
        b = self.bias.reshape(1, -1, 1, 1).to(ms.float32)
        rv = self.running_var.reshape(1, -1, 1, 1).to(ms.float32)
        rm = self.running_mean.reshape(1, -1, 1, 1).to(ms.float32)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        out = x * scale + bias

        out = out.to(ori_dtype)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    is_match = True
    if module_match:
        is_match = name in module_match

    for name, cell in module.cells_and_names():
        if is_match and isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.moving_mean.requires_grad = False
            module.moving_variance.requires_grad = False
            module.gamma.requires_grad = False
            module.beta.requires_grad = False
        else:
            freeze_batch_norm_2d(cell)
