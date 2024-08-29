
import numpy as np
import collections.abc

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from typing import Union, Optional, Tuple, Set

from transformers import Dinov2Config

from cambrian.transformers.activations import ACT2FN
from cambrian.transformers.modeling_utils import PreTrainedModel


class Dinov2Embeddings(nn.Cell):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        self.cls_token = Parameter(Tensor(np.random.randn(1, 1, config.hidden_size), ms.float32), name='cls_token')
        self.mask_token = Parameter(Tensor(np.zeros((1, config.hidden_size)), ms.float32), name='mask_token')
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = Parameter(Tensor(np.random.randn(1, num_patches + 1, config.hidden_size), ms.float32), name='position_embeddings')
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        height = height // self.config.patch_size
        width = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(num_positions ** 0.5), int(num_positions ** 0.5), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = ops.interpolate(
            patch_pos_embed.to(dtype=ms.float32),
            scale_factor=(float(height / (num_positions ** 0.5)), float(width / (num_positions ** 0.5))),
            recompute_scale_factor=True,
            mode="bilinear", #"bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)

        # assert int(height) == patch_pos_embed.shape[-2] and int(width) == patch_pos_embed.shape[-1]
        # if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
        #     raise ValueError("Width or height does not match with the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return ops.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)

    def construct(self, pixel_values: Tensor, bool_masked_pos: Optional[Tensor] = None) -> Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = ops.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.broadcast_to((batch_size, -1, -1)).to(embeddings.dtype)

        embeddings = ops.cat((cls_tokens, embeddings), axis=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings


class Dinov2PatchEmbeddings(nn.Cell):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, has_bias=True, pad_mode="pad"
        )

    def construct(self, pixel_values: Tensor) -> Tensor:
        # num_channels = pixel_values.shape[1]
        # assert num_channels == self.num_channels, \
        #     f"Make sure that the channel dimension of the pixel values match with the one set in the configuration." \
        #     f"Expected {self.num_channels} but got {num_channels}."

        embeddings = self.projection(pixel_values).flatten(start_dim=2).swapdims(1, 2)
        return embeddings


class Dinov2SelfAttention(nn.Cell):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.qkv_bias)
        self.key = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.qkv_bias)
        self.value = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.qkv_bias)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = tuple(x.shape[:-1]) + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self, hidden_states, head_mask: Optional[Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapdims(-1, -2))

        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = tuple(context_layer.shape[:-2]) + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class Dinov2SelfOutput(nn.Cell):
    """
    The residual connection is defined in Dinov2Layer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Dinov2Attention(nn.Cell):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.attention = Dinov2SelfAttention(config)
        self.output = Dinov2SelfOutput(config)

    def prune_heads(self, heads: Set[int]) -> None:
        raise NotImplementedError

    def construct(
        self,
        hidden_states: Tensor,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:

        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Dinov2LayerScale(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = Parameter(Tensor(config.layerscale_value * np.ones(config.hidden_size), ms.float32), name='lambda1')

    def construct(self, hidden_state: Tensor) -> Tensor:
        return hidden_state * self.lambda1


def drop_path(input: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + ops.rand(shape, dtype=input.dtype)
    random_tensor.floor()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class Dinov2DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: Tensor) -> Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Dinov2MLP(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=True)

    def construct(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2SwiGLUFFN(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Dense(in_features, 2 * hidden_features, has_bias=True)
        self.weights_out = nn.Dense(hidden_features, out_features, has_bias=True)

    def construct(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, axis=-1)
        hidden = ops.silu(x1) * x2
        return self.weights_out(hidden)


class Dinov2Layer(nn.Cell):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.attention = Dinov2Attention(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = Dinov2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

        if config.use_swiglu_ffn:
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)
        self.layer_scale2 = Dinov2LayerScale(config)

    def construct(
        self,
        hidden_states: Tensor,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:

        breakpoint()

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class Dinov2Encoder(nn.Cell):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.CellList([Dinov2Layer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        hidden_states: Tensor,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            breakpoint()

            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        _last_hidden_state, _hidden_states, _attentions = \
            hidden_states, all_hidden_states, all_self_attentions

        return _last_hidden_state, _hidden_states, _attentions


class Dinov2Model(PreTrainedModel):
    config_class = Dinov2Config

    def __init__(self, config: Dinov2Config):
        super().__init__(config)

        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        # assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def construct(
        self,
        pixel_values: Optional[Tensor] = None,
        bool_masked_pos: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        # last_hidden_state, pooler_output, hidden_states, attentions
        return sequence_output, pooled_output, encoder_outputs[1], encoder_outputs[2]
