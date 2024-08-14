import numpy as np

import mindspore as ms
from mindspore import ParameterTuple, Tensor, nn, ops, context
from mindspore.common.initializer import initializer
from mindspore.communication.management import GlobalComm, get_group_size, get_rank
from mindspore.ops import functional as F
from mindspore.ops import operations as P


adamw_opt = ops.MultitypeFuncGraph("adamw_opt")
adamw_opt_split = ops.MultitypeFuncGraph("adamw_opt_split")
split_params = ops.MultitypeFuncGraph("split_params")
update_params = ops.MultitypeFuncGraph("update_params")
allreduce_and_split_op = ops.MultitypeFuncGraph("reduce_and_split_op")
reducescatter_and_split_op = ops.MultitypeFuncGraph("reducescatter_and_split_op")


@update_params.register("Tensor", "Tensor", "Function")
def update_params(param, update, all_gather):
    update = all_gather(update)
    success = ops.logical_not(ops.isnan(update))
    success = ops.depend(success, ops.assign(param, update))
    return success


@split_params.register("Number", "Number", "Tensor")
def split_params(shard_id, shard_size, param):
    if param.shape[0] % shard_size == 0:
        param = ops.Split(0, shard_size)(param)[shard_id]
    return param


@adamw_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool")
def _adamw_opt(beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag):
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


@adamw_opt_split.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Number", "Number")
def _adamw_opt_split(shard_id, shard_size, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag):
    op_mul = P.Mul()
    op_square = P.Square()
    op_sqrt = P.Sqrt()
    op_cast = P.Cast()
    op_reshape = P.Reshape()
    op_shape = P.Shape()

    if param.shape[0] % shard_size == 0:
        param = ops.Split(0, shard_size)(param)[shard_id]

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


@allreduce_and_split_op.register("Number", "Bool", "Function", "Number", "Number", "Tensor")
def _tensors_allreduce_and_split(degree, mean, all_reduce_op, shard_id, shard_size, grad):
    # allreduce
    grad = all_reduce_op(grad)
    if mean:
        grad = F.tensor_mul(grad, F.cast(degree, F.dtype(grad)))

    # split
    if grad.shape[0] % shard_size == 0:
        grad = ops.Split(0, shard_size)(grad)[shard_id]

    return grad


@reducescatter_and_split_op.register("Number", "Bool", "Function", "Function", "Number", "Tensor")
def _tensors_reducescatter_and_split(degree, mean, reduce_scatter_op, all_reduce_op, shard_size, grad):

    if grad.shape[0] % shard_size == 0:
        # allreduce and split on world size
        grad = reduce_scatter_op(grad)
    else:
        # allreduce
        grad = all_reduce_op(grad)

    if mean:
        grad = F.tensor_mul(grad, F.cast(degree, F.dtype(grad)))

    return grad


class AdamWeightDecayZeRO1(nn.Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0, shard_size=None):
        super(AdamWeightDecayZeRO1, self).__init__(learning_rate, params, weight_decay)
        self.map = ops.Map()
        self.rank = get_rank()
        self.group_size = get_group_size()

        # group for split
        if shard_size is None:
            comm_group = GlobalComm.WORLD_COMM_GROUP
            g_id = 0
            self.shard_id = self.rank
            self.shard_size = self.group_size
        else:
            assert shard_size > 1
            assert (shard_size <= self.group_size) and (self.group_size % shard_size == 0)
            from mindspore.communication import create_group

            g_id = self.rank // shard_size
            s_id, e_id = g_id * shard_size, (g_id + 1) * shard_size
            comm_group = f"sub_group_{g_id}"
            create_group(comm_group, [_i for _i in range(s_id, e_id)])
            self.shard_id = self.rank % shard_size
            self.shard_size = shard_size

        print(
            f"WARNING: {self.__class__.__name__} shard size setting to {self.shard_size}, "
            f"shard_id {self.shard_id}, group: {comm_group}"
        )

        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self._param_init_op(self._parameters, prefix="adam_m", init="zeros")
        self.moments2 = self._param_init_op(self._parameters, prefix="adam_v", init="zeros")
        self.all_gather_ops = self._init_all_gather_ops(self._parameters, group=comm_group)

        self.all_reduce_op = ops.AllReduce()
        self.mean = context.get_auto_parallel_context("gradients_mean")
        self.degree = context.get_auto_parallel_context("device_num")
        self.degree = 1. / self.degree

        total_num = len(self.all_gather_ops)
        split_num = sum([1 for _op in self.all_gather_ops if isinstance(_op, ops.AllGather)])
        unsplit_num = total_num - split_num
        print(
            f"WARNING: {self.__class__.__name__}, total param num: {total_num}, "
            f"split num: {split_num}, unsplit num: {unsplit_num}"
        )

    def _init_all_gather_ops(self, params, group):
        op_list = []
        for x in params:
            if x.split_op:
                op_list.append(ops.AllGather(group=group))
            else:
                op_list.append(ops.identity)
        return tuple(op_list)

    def _param_init_op(self, params, prefix, init="zeros"):
        news = []
        for p in params:
            s = p.shape
            if s[0] % self.shard_size == 0:
                s = list(s)
                s[0] = s[0] // self.shard_size
                s = tuple(s)
                new = ms.Parameter(initializer(init, shape=s, dtype=p.dtype), name=prefix + "." + p.name)
                setattr(p, "split_op", True)
            else:
                new = p.clone(init)
                new.name = prefix + "." + p.name
                setattr(p, "split_op", False)
                print(f"[WARNING] Split {new.name} fail, keep ori shape.")
            news.append(new)
        return ParameterTuple(news)

    def grad_reduce(self, grads):
        mean, degree, shard_id, shard_size = self.mean, self.degree, self.shard_id, self.shard_size
        return self.grad_allreduce_and_split(mean, degree, shard_id, shard_size, grads)

    def grad_allreduce_and_split(self, mean, degree, shard_id, shard_size, gradients):
        gradients = ops.HyperMap()(
            F.partial(allreduce_and_split_op, degree, mean, self.all_reduce_op, shard_id, shard_size),
            gradients
        )
        return gradients

    @ms.jit
    def bak_construct(self, split_gradients):
        gradients = split_gradients
        params = self.hyper_map(F.partial(split_params, self.shard_id, self.shard_size), self._parameters)
        # gradients = self.hyper_map(F.partial(split_params, self.shard_id, self.shard_size), gradients)
        # params = self.hyper_map(F.partial(split_params, self.shard_id, self.shard_size), self._parameters)

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
                    params,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                )
            else:
                optim_result = self.hyper_map(
                    F.partial(adamw_opt, self.beta1, self.beta2, self.eps, lr),
                    weight_decay,
                    params,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                )
        else:
            optim_result = self.hyper_map(
                F.partial(adamw_opt, self.beta1, self.beta2, self.eps, lr, weight_decay),
                params,
                self.moments1,
                self.moments2,
                gradients,
                self.decay_flags,
            )

        success = self.hyper_map(update_params, self._parameters, optim_result, self.all_gather_ops)

        return success

    @ms.jit
    def construct(self, split_gradients):
        gradients = split_gradients
        # params = self.hyper_map(F.partial(split_params, self.shard_id, self.shard_size), self._parameters)
        # gradients = self.hyper_map(F.partial(split_params, self.shard_id, self.shard_size), gradients)
        # params = self.hyper_map(F.partial(split_params, self.shard_id, self.shard_size), self._parameters)

        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    F.partial(adamw_opt_split, self.shard_id, self.shard_size, self.beta1, self.beta2, self.eps),
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
                    F.partial(adamw_opt_split, self.shard_id, self.shard_size, self.beta1, self.beta2, self.eps, lr),
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                )
        else:
            optim_result = self.hyper_map(
                F.partial(adamw_opt_split, self.shard_id, self.shard_size, self.beta1, self.beta2, self.eps, lr, weight_decay),
                self._parameters,
                self.moments1,
                self.moments2,
                gradients,
                self.decay_flags,
            )

        success = self.hyper_map(update_params, self._parameters, optim_result, self.all_gather_ops)

        return success


class AdamWeightDecayZeRO2(AdamWeightDecayZeRO1):

    def __init__(self, *args, **kwargs):
        super(AdamWeightDecayZeRO2, self).__init__(*args, **kwargs)
        self.reduce_scatter_op = ops.ReduceScatter()

    def grad_reduce(self, grads):

        mean, degree, shard_id, shard_size = self.mean, self.degree, self.shard_id, self.shard_size

        if self.group_size == self.shard_size:
            return self.grad_reducescatter_and_split(mean, degree, shard_id, shard_size, grads)
        else:
            return self.grad_allreduce_and_split(mean, degree, shard_id, shard_size, grads)

    def grad_reducescatter_and_split(self, mean, degree, shard_id, shard_size, gradients):
        gradients = ops.HyperMap()(
            F.partial(reducescatter_and_split_op, degree, mean, self.reduce_scatter_op, self.all_reduce_op, shard_size),
            gradients
        )
        return gradients
