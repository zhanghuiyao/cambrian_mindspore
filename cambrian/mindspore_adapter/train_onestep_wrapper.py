from typing import Dict

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, context, ParallelMode
from mindspore.boost.grad_accumulation import gradient_accumulation_op as _grad_accum_op
from mindspore.boost.grad_accumulation import gradient_clear_op as _grad_clear_op


def _is_pynative_parallel():
    parallel_mode = context.get_auto_parallel_context('parallel_mode')
    return context.get_context('mode') == context.PYNATIVE_MODE and parallel_mode in (
        context.ParallelMode.SEMI_AUTO_PARALLEL, context.ParallelMode.AUTO_PARALLEL)


def create_loss_scaler(ms_loss_scaler="static", scale_value=1024, scale_factor=2, scale_window=1000):
    if ms_loss_scaler == "dynamic":
        from mindspore.amp import DynamicLossScaler

        loss_scaler = DynamicLossScaler(scale_value=scale_value, scale_factor=scale_factor, scale_window=scale_window)
    elif ms_loss_scaler == "static":
        from mindspore.amp import StaticLossScaler

        loss_scaler = StaticLossScaler(scale_value=scale_value)
    elif ms_loss_scaler in ("none", "None"):
        from mindspore.amp import StaticLossScaler

        loss_scaler = StaticLossScaler(1.0)
    else:
        raise NotImplementedError(f"Not support ms_loss_scaler: {ms_loss_scaler}")

    return loss_scaler


def create_grad_reducer(trainable_parameters):
    use_reducer = context.get_auto_parallel_context("parallel_mode") in (
    ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL) \
                  or _is_pynative_parallel()

    if use_reducer:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(trainable_parameters, mean, degree)
    else:
        grad_reducer = nn.Identity()
    return grad_reducer


class TrainOneStepWrapper(nn.Cell):
    """TrainStep with ema and clip grad.

    Returns:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.
        loss (Tensor) -  A scalar, the loss value.
        overflow (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        loss scale (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    """

    def __init__(
        self,
        network: nn.Cell,
        optimizer: nn.Optimizer,
        ema: nn.Cell = None,
        drop_overflow_step: bool = True,
        scaler: str = "default",
        scaler_config: Dict = {},
        gradient_accumulation_steps: int = 1,
        clip_grad: str = "norm",
        clip_value: float = 1.0,
    ):
        super().__init__(auto_prefix=False)

        # grad and optimizer
        self.grad_fn = ops.value_and_grad(network, grad_position=None, weights=optimizer.parameters)
        self.optimizer = optimizer
        self.ema = ema

        # scaler and reducer
        assert "ms_loss_scaler" not in scaler_config
        if scaler.lower() == "default":
            if len(scaler_config) == 0:
                scaler_config = {"scale_value": 1024}
            scaler = create_loss_scaler("static", **scaler_config)
        elif scaler.lower() == "static":
            scaler = create_loss_scaler("static", **scaler_config)
        elif scaler.lower() in ("auto", "dynamic"):
            scaler = create_loss_scaler("dynamic", **scaler_config)
        elif scaler.lower() == "none":
            scaler = create_loss_scaler("none", **scaler_config)
        else:
            raise NotImplementedError
        self.scaler = scaler
        self.reducer = create_grad_reducer(self.network.trainable_params())
        self.all_finite = ms.amp.all_finite
        self.all_finite_reducer = ops.AllReduce()
        self.drop_overflow_step = Tensor(drop_overflow_step, ms.bool_)

        # clip grad
        assert clip_value > 0.0 and isinstance(clip_value, float), f"clip_value must be float > 0., but got {clip_value}"
        if clip_grad.lower() in ("global", "global_norm"):
            from cambrian.mindspore_adapter.clip_grad import clip_grad_global as clip_grad_global_
            clip_grad_fn = ops.partial(clip_grad_global_, clip_value)
        elif clip_grad.lower() in ("local", "local_norm", "norm"):
            from cambrian.mindspore_adapter.clip_grad import clip_grad as clip_grad_
            clip_grad_fn = ops.partial(clip_grad_, clip_value)
        elif clip_grad.lower() == "none":
            clip_grad_fn = None
        else:
            raise NotImplementedError
        self.clip_grad_fn = clip_grad_fn

        # grad accumulation
        assert gradient_accumulation_steps >= 1
        self.accum_steps = gradient_accumulation_steps
        if gradient_accumulation_steps > 1:
            self.hyper_map = ops.HyperMap()
            self.cur_accum_step = ms.Parameter(ms.Tensor(0, dtype=ms.int32), name="accum_step", requires_grad=False)
            self.accumulated_grads = optimizer.parameters.clone(prefix="accum_grad", init="zeros")

    def do_optim(self, loss, grads):

        if not self.accum_steps > 1:
            if self.clip_grad_fn is not None:
                grads = self.clip_grad_fn(grads)   # FIXME: compare with torch.nn.utils.clip_grad_norm_
            loss = ops.depend(loss, self.optimizer(grads))
            if self.ema is not None:
                self.ema.ema_update()
        else:
            loss = ops.depend(
                loss, self.hyper_map(ops.partial(_grad_accum_op, self.accum_steps), self.accumulated_grads, grads)
            )
            loss = ops.depend(loss, ops.assign_add(self.cur_accum_step, ms.Tensor(1, ms.int32)))
            if self.cur_accum_step % self.accum_steps == 0:
                if self.clip_grad_fn is not None:
                    grads = self.clip_grad_fn(self.accumulated_grads)
                    loss = ops.depend(loss, self.optimizer(grads))
                else:
                    loss = ops.depend(loss, self.optimizer(self.accumulated_grads))
                loss = ops.depend(loss, self.hyper_map(ops.partial(_grad_clear_op), self.accumulated_grads))
                loss = ops.depend(loss, ops.assign(self.cur_accum_step, ms.Tensor(0, ms.int32)))
                if self.ema is not None:
                    self.ema.ema_update()
            else:
                # update the learning rate, do not update the parameter
                loss = ops.depend(loss, self.optimizer.get_lr())

        return loss

    def construct(self, *args, **kwargs):
        loss, grads = self.grad_fn(*args, **kwargs)
        grads = self.reducer(grads)
        unscaled_grads = self.scaler.unscale(grads)
        finite = self.all_finite(unscaled_grads)
        finite = self.all_finite_reducer(finite)
        _do_optim = finite or (not self.drop_overflow_step)

        if _do_optim:
            loss = self.do_optim(loss, unscaled_grads)

        overflow_tag = not finite
        return self.scaler.unscale(loss), unscaled_grads, overflow_tag