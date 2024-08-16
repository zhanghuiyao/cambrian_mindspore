import numpy as np

import mindspore as ms
from mindspore import ParameterTuple, Tensor, nn, ops
from mindspore.ops import functional as F
from mindspore.ops import operations as P


update_params = ops.MultitypeFuncGraph("update_params")
adamw_opt = ops.MultitypeFuncGraph("adamw_opt")


# @adamw_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool")
def _bak_adamw_opt(beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag):
    op_mul = P.Mul()
    op_square = P.Square()
    op_sqrt = P.Sqrt()
    op_cast = P.Cast()
    op_reshape = P.Reshape()
    op_shape = P.Shape()
    param_fp32 = op_cast(param, ms.float32)
    m_fp32 = op_cast(m, ms.float32)
    v_fp32 = op_cast(v, ms.float32)
    gradient_fp32 = op_cast(gradient, ms.float32)

    next_m = op_mul(beta1, m_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), ms.float32) - beta1, gradient_fp32)

    next_v = op_mul(beta2, v_fp32) + op_mul(
        op_cast(F.tuple_to_array((1.0,)), ms.float32) - beta2, op_square(gradient_fp32)
    )

    update = next_m / (eps + op_sqrt(next_v))
    if decay_flag:
        update = op_mul(weight_decay, param_fp32) + update

    update_with_lr = op_mul(lr, update)
    next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

    # next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
    next_param = F.depend(next_param, F.assign(m, op_cast(next_m, F.dtype(m))))
    next_param = F.depend(next_param, F.assign(v, op_cast(next_v, F.dtype(v))))

    return op_cast(next_param, F.dtype(param))


@adamw_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool")
def _adamw_opt(beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag):
    # op_mul = P.Mul()
    # op_square = P.Square()
    # op_sqrt = P.Sqrt()
    # op_cast = P.Cast()
    # op_reshape = P.Reshape()
    # op_shape = P.Shape()
    param_fp32 = ops.cast(param, ms.float32)
    m_fp32 = ops.cast(m, ms.float32)
    v_fp32 = ops.cast(v, ms.float32)
    gradient_fp32 = ops.cast(gradient, ms.float32)

    next_m = ops.mul(beta1, m_fp32) + ops.mul(ops.cast(F.tuple_to_array((1.0,)), ms.float32) - beta1, gradient_fp32)

    next_v = ops.mul(beta2, v_fp32) + ops.mul(
        ops.cast(F.tuple_to_array((1.0,)), ms.float32) - beta2, ops.square(gradient_fp32)
    )

    update = next_m / (eps + ops.sqrt(next_v))
    if decay_flag:
        update = ops.mul(weight_decay, param_fp32) + update

    update_with_lr = ops.mul(lr, update)
    next_param = param_fp32 - ops.reshape(update_with_lr, param_fp32.shape)

    # next_param = F.depend(next_param, F.assign(param, ops.cast(next_param, F.dtype(param))))
    next_param = F.depend(next_param, F.assign(m, ops.cast(next_m, F.dtype(m))))
    next_param = F.depend(next_param, F.assign(v, ops.cast(next_v, F.dtype(v))))

    return ops.cast(next_param, F.dtype(param))


@update_params.register("Tensor", "Tensor")
def update_params(param, update):
    success = ops.logical_not(ops.isnan(update))
    success = ops.depend(success, ops.assign(param, update))
    return success


class AdamWeightDecay(nn.Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecay, self).__init__(learning_rate, params, weight_decay)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self._param_init_op(self._parameters, prefix="adam_m", init="zeros")
        self.moments2 = self._param_init_op(self._parameters, prefix="adam_v", init="zeros")

    def _param_init_op(self, params, prefix, init="zeros"):
        news = []
        for p in params:
            new = p.clone(init)
            new.name = prefix + "." + p.name
            news.append(new)
        return ParameterTuple(news)

    @ms.jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    F.partial(adamw_opt, self.beta1, self.beta2, self.eps),
                    lr,
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                )
            else:
                optim_result = self.hyper_map(
                    F.partial(adamw_opt, self.beta1, self.beta2, self.eps, lr),
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                )
        else:
            optim_result = self.hyper_map(
                F.partial(adamw_opt, self.beta1, self.beta2, self.eps, lr, weight_decay),
                self._parameters,
                self.moments1,
                self.moments2,
                gradients,
                self.decay_flags,
            )

        success = self.hyper_map(update_params, self._parameters, optim_result)

        return success
