import numpy as np
from functools import partial

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter



def build_pos_embeds(
    num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    pos_emb = Parameter(Tensor(np.zeros((1, num_input_tokens, vision_hidden_size)), ms.float32), name="pos_emb")

    # TODO: weight init
    # nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)

    return pos_emb

def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Dense(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Dense(output_hidden_size, output_hidden_size))
    return nn.SequentialCell(layers)


class Projector(nn.Cell):
    """Base projector class"""

    def __init__(
        self,
        encoder_hidden_size: int,
        num_input_tokens: int,
        num_queries: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.num_input_tokens = num_input_tokens
        self.num_queries = num_queries
        self.hidden_size = 1024
        self.output_hidden_size = output_hidden_size

        # pos emb
        self.pos_emb = build_pos_embeds(num_input_tokens, encoder_hidden_size)
        # self.pos_emb = None

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.pos_emb is not None:
            x = x + self.pos_emb

        dtype = x.dtype
        # x = self._forward(x.to(torch.float32))  # (B, L, output_hidden_size)
        x = self._forward(x)

        return x.to(dtype)


class ConvProjector(Projector):
    def _forward(self, x):
        hw = int(x.size(1) ** 0.5)

        # rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        b, _, d = x.shape
        x = x.transpose(0, 2, 1).view(b, d, hw, hw)

        x = self.net(x)

        # x = rearrange(x, "b d h w -> b (h w) d")
        b, d, h, w = x.shape
        x = x.view(b, d, -1).transpose(0, 2, 1)

        x = self.readout(x)
        return x


class CAbstractor(nn.Cell):
    """Base projector class"""

    def __init__(
        self,
        encoder_hidden_size: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = 1024
        self.output_hidden_size = output_hidden_size

        # pos emb
        # self.pos_emb = build_pos_embeds(num_input_tokens, encoder_hidden_size)
        self.pos_emb = None

        self.downsamples = nn.Conv2d(
                    encoder_hidden_size,
                    self.hidden_size,
                    kernel_size=3,
                    stride=2,
                    pad_mode="pad",
                    padding=1,
                    has_bias=False,
                )
        
        self.readout = nn.SequentialCell([
            nn.Dense(self.hidden_size, output_hidden_size),
            nn.GELU(),
            nn.Dense(output_hidden_size, output_hidden_size)
        ])

    def construct(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.to(ms.float32)

        hw = int(x.shape[1] ** 0.5)

        # x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        b, _, d = x.shape
        x = x.transpose(0, 2, 1).view(b, d, hw, hw)

        x = self.downsamples(x)

        # x = rearrange(x, "b d h w -> b (h w) d")
        b, d, h, w = x.shape
        x = x.view(b, d, h * w).transpose(0, 2, 1)

        x = self.readout(x)

        return x.to(dtype)
