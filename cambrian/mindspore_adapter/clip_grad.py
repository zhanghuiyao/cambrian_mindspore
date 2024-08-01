import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F


_clip_grad = C.MultitypeFuncGraph("clip_grad")


@_clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor]: clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad, F.cast(F.tuple_to_array((-clip_value,)), dt), F.cast(F.tuple_to_array((clip_value,)), dt)
        )
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


apply_global_norm = C.MultitypeFuncGraph("_apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    x = x * clip_norm / global_norm
    x = F.cast(x, x_dtype)
    return x


class L2Norm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.l2_norm_1 = ops.LpNorm((0,))
        self.l2_norm_2 = ops.LpNorm((0, 1))
        self.l2_norm_3 = ops.LpNorm((0, 1, 2))
        self.l2_norm_4 = ops.LpNorm((0, 1, 2, 3))

    def construct(self, x):
        if x.ndim == 1:
            norm = self.l2_norm_1(x)
        elif x.ndim == 2:
            norm = self.l2_norm_2(x)
        elif x.ndim == 3:
            norm = self.l2_norm_3(x)
        else:
            norm = self.l2_norm_4(x)
        return norm


class _ClipByGlobalNormFix(nn.Cell):
    def __init__(self, clip_norm=1.0):
        super().__init__()
        self.clip_norm = Tensor([clip_norm], ms.float32)
        self.hyper_map = ops.HyperMap()
        self.greater_equal = ops.GreaterEqual()
        self.l2norm = L2Norm()

    def construct(self, x):
        norms = self.hyper_map(self.l2norm, x)
        norms_square = self.hyper_map(ops.square, norms)
        global_norm = ops.sqrt(ops.addn(norms_square)).astype(ms.float32)

        cond = self.greater_equal(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), x)
        return clip_x


hyper_map_op = ops.HyperMap()


def clip_grad(clip_norm, x):
    clip_value = hyper_map_op(F.partial(_clip_grad, 1, clip_norm), x)
    return clip_value


def clip_grad_global(clip_norm, x):
    clip_value = _ClipByGlobalNormFix(clip_norm)(x)
    return clip_value
