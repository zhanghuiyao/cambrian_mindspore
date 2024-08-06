
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from cambrian.mindspore_adapter.train_onestep_wrapper import TrainOneStepWrapper


def _build_optimizer(model):
    return nn.Momentum(model.trainable_params(), learning_rate=0.1, momentum=0.9)


def loss_with_target_one(x):
    return ((x - 1.) ** 2).mean()


class NetWithLoss(nn.Cell):
    def __init__(self, model, loss_fn=loss_with_target_one, out_feature_index=0):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self.net = model
        self.loss_fn = loss_fn
        self.out_feature_index = out_feature_index

    def construct(self, *args, **kwargs):
        out = self.net(*args, **kwargs)
        if isinstance(out, (tuple, list)):
            out = out[self.out_feature_index]

        if self.loss_fn is not None:
            loss = self.loss_fn(out)
        else:
            loss = out

        return loss


def build_train_net(model, loss_fn=loss_with_target_one, out_feature_index=0):
    net_with_loss = NetWithLoss(model, loss_fn, out_feature_index=out_feature_index)
    optimizer = _build_optimizer(net_with_loss)
    train_net = TrainOneStepWrapper(
        net_with_loss,
        optimizer,
        clip_grad="global_norm",
        clip_value=1.0
    )

    return train_net
