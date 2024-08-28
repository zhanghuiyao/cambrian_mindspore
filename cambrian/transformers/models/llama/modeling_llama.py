import math
import numpy as np
from typing import List, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from transformers import LlamaConfig, GenerationConfig, logging

from cambrian.transformers.activations import ACT2FN
from cambrian.transformers.cache_utils import Cache, StaticCache, DynamicCache
from cambrian.transformers.modeling_utils import PreTrainedModel
from cambrian.transformers.modeling_attn_mask_utils import DTYPE_FP16_MIN

from cambrian.mindspore_adapter.attention import FlashAttention2
from cambrian.constants import IGNORE_INDEX


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(Tensor(np.ones(hidden_size), ms.float32), name='weight')
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Cell):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2).astype(np.float32) / self.dim))
        self.inv_freq = Parameter(Tensor(inv_freq, ms.float32), requires_grad=False, name="inv_freq_buffer")
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    def construct(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].to(ms.float32).broadcast_to((position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].to(ms.float32)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = ops.matmul(inv_freq_expanded, position_ids_expanded).swapdims(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        cos, sin = cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        cos, sin = ops.stop_gradient(cos), ops.stop_gradient(sin)
        return cos, sin


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def construct(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.to(ms.float32) / self.scaling_factor
        cos, sin = super().construct(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def construct(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = ops.max(position_ids)[0] + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (ops.arange(0, self.dim, 2, dtype=ms.float32) / self.dim)
            )
            x = ops.depend(x, ops.assign(self.inv_freq, inv_freq))

        cos, sin = super().construct(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=config.mlp_bias)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=config.mlp_bias)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

        _name_list = [
            'pretraining_tp',
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

    def construct(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, axis=0)
            up_proj_slices = self.up_proj.weight.split(slice, axis=0)
            down_proj_slices = self.down_proj.weight.split(slice, axis=1)

            gate_proj = ops.cat(
                [ops.dense(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], axis=-1
            )
            up_proj = ops.cat([ops.dense(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], axis=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, axis=2)
            down_proj = [
                ops.dense(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

            breakpoint()
            x_gate = self.gate_proj(x)
            x_gate = self.act_fn(x_gate)
            x_up = self.up_proj(x)
            x = x_gate * x_up
            down_proj = self.down_proj(x)

        return down_proj


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.attention_bias)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.o_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=config.attention_bias)
        self._init_rope()

        _name_list = [
            'pretraining_tp',
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ):
        breakpoint()

        bsz, q_len, _ = hidden_states.shape

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, axis=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [ops.dense(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = ops.cat(query_states, axis=-1)

            key_states = [ops.dense(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = ops.cat(key_states, axis=-1)

            value_states = [ops.dense(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = ops.cat(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapdims(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapdims(2, 3)) / (self.head_dim ** 0.5)

        attn_weights = ops.cast(attn_weights, ms.float32)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights + ops.cast(causal_mask, attn_weights.dtype)
            # attn_weights = ops.clip(attn_weights, min=DTYPE_FP16_MIN)

        # upcast attention to fp32
        # attn_weights = ops.softmax(attn_weights, axis=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = ops.softmax(attn_weights, axis=-1).to(query_states.dtype)

        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)
        # assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.swapdims(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, axis=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, axis=1)
            attn_output = sum([ops.dense(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        # FIXME: level 0, bug when return None
        # if not output_attentions:
        #     attn_weights = None
        # return attn_output, attn_weights, past_key_value

        outputs = (attn_output,)
        if past_key_value is not None:
            outputs += (past_key_value,)

        # attn_output, past_key_value
        return outputs



class LlamaFlashAttention2(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.flash_attention = FlashAttention2(
            self.head_dim,
            self.num_heads,
            self.attention_dropout,
            input_layout="BNSD",
            dtype=ms.float16
        )

    def convert_mask_to_fa_format(self, attention_mask):

        if attention_mask is not None:
            if attention_mask.dtype == ms.bool_:
                # flip mask, since ms FA treats 1 as discard, 0 as retain.
                attention_mask = 1 - attention_mask
                attention_mask = attention_mask.to(ms.uint8)
            else:
                dtype_fp16_min = ops.full((), DTYPE_FP16_MIN, dtype=ms.float16)
                attention_mask = attention_mask.to(ms.float16)
                attention_mask = ops.select(
                    ops.equal(attention_mask, dtype_fp16_min),
                    ops.ones((), ms.uint8),
                    ops.zeros((), ms.uint8),
                )

        return attention_mask

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, axis=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [ops.dense(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = ops.cat(query_states, axis=-1)

            key_states = [ops.dense(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = ops.cat(key_states, axis=-1)

            value_states = [ops.dense(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = ops.cat(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapdims(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        # 1. flash attention
        if attention_mask is not None:  # no matter the length, we just slice it
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attention_mask = self.convert_mask_to_fa_format(attention_mask)
        attn_output = self.flash_attention(query_states, key_states, value_states, attention_mask)
        # assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)

        # 2. vanilla attention
        # attn_weights = ops.matmul(query_states, key_states.swapdims(2, 3)) / (self.head_dim ** 0.5)
        #
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask
        #
        # # upcast attention to fp32
        # attn_weights = ops.softmax(attn_weights, axis=-1, dtype=ms.float32).to(query_states.dtype)
        # attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = ops.matmul(attn_weights, value_states)
        # # assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.swapdims(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, axis=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, axis=1)
            attn_output = sum([ops.dense(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if past_key_value is not None:
            outputs += (past_key_value,)

        # attn_output, past_key_value
        return outputs


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": None,
}


class LlamaDecoderLayer(nn.Cell):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config._attn_implementation not in ["eager", "flash_attention_2"]:
            raise NotImplementedError
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.output_identity = nn.Identity()

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        breakpoint()

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = attn_output[0]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.output_identity(hidden_states)

        past_key_value = None if len(attn_output) == 1 else attn_output[1]
        outputs = (hidden_states,)
        if use_cache and past_key_value is not None:
            outputs += (past_key_value,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    _supports_flash_attn_2 = True


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # TODO: Initialize weights and apply final processing
        # self.post_init()

        _name_list = [
            'output_attentions', 'output_hidden_states', 'use_return_dict', 'use_cache',
            '_attn_implementation', 'pretraining_tp', 'vocab_size'
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        if not isinstance(value, nn.Embedding):
            raise NotImplementedError
        ori_name = value.embedding_table.name

        self.embed_tokens = value

        self.embed_tokens.embedding_table.name = ori_name

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
    ) -> Union[Tuple, ]:
        use_cache = use_cache if use_cache is not None else self.use_cache

        # if self.training:
        #     assert not use_cache
        # assert not output_attentions
        # assert not output_hidden_states
        # assert ((input_ids is None) and (inputs_embeds is not None)) or \
        #        ((input_ids is not None) and (inputs_embeds is None))
        # # assert (input_ids is None) ^ (inputs_embeds is None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
            cache_position = ops.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                # assert past_key_values is not None
                next_cache = layer_outputs[1]
                past_key_values[layer_idx] = next_cache

        hidden_states = self.norm(hidden_states)

        outputs = (hidden_states,)
        if past_key_values is not None and use_cache:
            outputs += (past_key_values,)

        # last_hidden_state, past_key_values, hidden_states, attentions
        return outputs

    def _update_causal_mask(
        self,
        attention_mask: Tensor,
        input_tensor: Tensor,
        cache_position: Tensor,
        past_key_values: Tuple[Tuple[Tensor]],
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self._attn_implementation == "flash_attention_2":
            # if attention_mask is not None and 0.0 in attention_mask:
            #     return attention_mask
            # return None
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self._attn_implementation == "sdpa" and not output_attentions:
            return None

        dtype, min_dtype = input_tensor.dtype, DTYPE_FP16_MIN
        sequence_length = input_tensor.shape[1]

        # if using_static_cache:
        if past_key_values:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and len(attention_mask.shape) == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing

            # if attention_mask.max() != 0:
            #     raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            # assert attention_mask.max() == 0

            causal_mask = attention_mask
        else:
            causal_mask = ops.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype
            )
            if sequence_length != 1:
                causal_mask = ops.triu(causal_mask, diagonal=1)

            _mask_position = ops.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask *= _mask_position
            causal_mask = causal_mask[None, None, :, :].broadcast_to((input_tensor.shape[0], 1, -1, -1))
            if attention_mask is not None:
                causal_mask = causal_mask  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = (padding_mask == 0)
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        _name_list = ['output_attentions', 'output_hidden_states', 'use_return_dict', 'pretraining_tp', 'vocab_size']
        for name in _name_list:
            setattr(self, name, getattr(config, name))

        # TODO: Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            # gradient_checkpointing_kwargs = {"mp_comm_recompute": True, "parallel_optimizer_comm_recompute": True}
            gradient_checkpointing_kwargs = {}

        from cambrian.mindspore_adapter import recompute_except_output

        # llama layers
        for decoder_layer in self.model.layers:
            assert isinstance(decoder_layer, LlamaDecoderLayer)
            for name, cell in decoder_layer.name_cells().items():
                if "output_identity" in name:
                    assert isinstance(cell, nn.Identity)
                    pass
                else:
                    # cell._recompute()
                    recompute_except_output(cell, **gradient_checkpointing_kwargs)
        recompute_except_output(self.model.embed_tokens, **gradient_checkpointing_kwargs)
        recompute_except_output(self.model.norm, **gradient_checkpointing_kwargs)

        # llama head
        # recompute_except_output(self.lm_head, **gradient_checkpointing_kwargs)

        logger.info(f"{self.__class__.__name__}: enable recompute done.")

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Union[Cache, List[Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
    ) -> Union[Tuple, ]:
        r"""
        Args:
            labels (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        # assert not output_attentions
        # assert not output_hidden_states
        # assert ((input_ids is None) and (inputs_embeds is not None)) or \
        #        ((input_ids is not None) and (inputs_embeds is None))
        # assert (input_ids is None) ^ (inputs_embeds is None)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        if use_cache:
            past_key_values = outputs[1]

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, axis=0)
            logits = [ops.dense(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = ops.cat(logits, axis=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.to(ms.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.cross_entropy_loss(shift_logits, shift_labels)

        if use_cache:
            return loss, logits, past_key_values
        else:
            return loss, logits

    def preprocess_input_before_generate_numpy(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray = None,
        position_ids: np.ndarray = None,
        attention_mask: np.ndarray = None,
    ):

        # init empty array
        bs = len(input_ids)
        padded_input_ids = np.zeros((bs, self.tokenizer_model_max_length), np.int32)
        padded_labels = np.full((bs, self.tokenizer_model_max_length), IGNORE_INDEX, np.int32)
        padded_position_ids = np.zeros((bs, self.tokenizer_model_max_length), np.int32)
        padded_attention_mask = np.zeros((bs, self.tokenizer_model_max_length), np.bool_)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.bool_)
        else:
            attention_mask = attention_mask.astype(np.bool_)
        if position_ids is None:
            position_ids = np.arange(0, input_ids.shape[1], dtype=np.int32)
        if labels is None:
            labels = np.full_like(input_ids, IGNORE_INDEX)

        masked_input_ids = []
        masked_labels = []
        masked_attention_mask = []
        for i in range(len(input_ids)):
            cur_input_ids, cur_labels, cur_attention_mask = input_ids[i], labels[i], attention_mask[i]
            active_len = int(cur_attention_mask.sum())
            # assert cur_attention_mask[:active_len].sum() == cur_attention_mask.sum()
            masked_input_ids.append(cur_input_ids[:active_len])
            masked_labels.append(cur_labels[:active_len])
            masked_attention_mask.append(cur_attention_mask[:active_len])
        input_ids = masked_input_ids
        labels = masked_labels
        attention_mask = masked_attention_mask

        for batch_idx, cur_input_ids in enumerate(input_ids):

            cur_len = cur_input_ids.shape[0]

            if self.tokenizer_padding_side == "right":
                padded_input_ids[batch_idx, :cur_len] = cur_input_ids[:]

                padded_labels[batch_idx, :cur_len] = labels[batch_idx][:]
                padded_attention_mask[batch_idx, :cur_len] = attention_mask[batch_idx][:]
                padded_position_ids[batch_idx, :cur_len] = np.arange(0, cur_len, dtype=position_ids.dtype)
            elif self.tokenizer_padding_side == "left":
                # padded_input_ids[batch_idx, -cur_len:] = cur_input_ids[:]
                #
                # padded_labels[batch_idx, -cur_len:] = labels[batch_idx][:]
                # padded_attention_mask[batch_idx, -cur_len:] = attention_mask[batch_idx][:]
                # padded_position_ids[batch_idx, -cur_len:] = np.arange(0, cur_len, dtype=position_ids.dtype)
                raise ValueError
            else:
                raise ValueError

        new_input_ids = Tensor(padded_input_ids)
        new_attention_mask = Tensor(padded_attention_mask)
        new_labels = None if _labels is None else Tensor(padded_labels)
        new_position_ids = None if _position_ids is None else Tensor(padded_position_ids)

        return new_input_ids, new_labels, new_position_ids, new_attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=False,
        position_ids=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                Tensor(past_key_values.get_max_length())
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else ops.minimum(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.to(ms.int32).cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            if not isinstance(input_ids, Tensor):
                input_ids = Tensor(input_ids, dtype=ms.int32)
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = ops.arange(past_length, past_length + input_length)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past
