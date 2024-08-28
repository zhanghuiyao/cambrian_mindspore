import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from functools import partial

from cambrian.timm.layers.helpers import to_2tuple


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=partial(nn.GELU, approximate=False),
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1, pad_mode="pad") if use_conv else nn.Dense

        self.fc1 = linear_layer(in_features, hidden_features, has_bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, has_bias=bias[1])
        self.drop2 = nn.Dropout(p=drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Cell):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1, pad_mode="pad") if use_conv else nn.Dense
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, has_bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, has_bias=bias[1])
        self.drop2 = nn.Dropout(p=drop_probs[1])

    def init_weights(self):
        # # override init of fc1 w/ gate portion set to weight near zero, bias=1
        # fc1_mid = self.fc1.bias.shape[0] // 2
        # nn.init.ones_(self.fc1.bias[fc1_mid:])
        # nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)
        raise NotImplementedError

    def construct(self, x):
        x = self.fc1(x)

        x1, x2 = x.chunk(2, axis=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nn.SiLU, gate_last=False)
