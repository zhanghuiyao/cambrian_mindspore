import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, ParameterTuple

from cambrian.model.nn_functional import scaled_dot_product_attention


class CrossAttention(nn.Cell):

    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.SequentialCell([
            nn.LayerNorm([q_dim]), nn.Dense(q_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
        ])
        self.k_proj = nn.SequentialCell([
            nn.LayerNorm([kv_dim]), nn.Dense(kv_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
        ])
        self.v_proj = nn.SequentialCell([
            nn.LayerNorm([kv_dim]), nn.Dense(kv_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
        ])
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, q_dim, has_bias=attention_bias)

    def construct(
            self,
            vision_latents, queries, attention_mask
    ):

        bsz, q_len, _ = queries.shape
        bsz, v_len, _ = vision_latents.shape

        query_states = self.q_proj(queries)
        key_states = self.k_proj(vision_latents)
        value_states = self.v_proj(vision_latents)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapdims(1, 2)
        key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).swapdims(1, 2)
        value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).swapdims(1, 2)

        if attention_mask is not None:
            # if attention_mask.shape != (bsz, 1, q_len, v_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.shape}"
            #     )
            assert attention_mask.shape == (bsz, 1, q_len, v_len)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.swapdims(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


class MLP(nn.Cell):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.linear_1 = nn.Dense(d_in, d_hidden, has_bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Dense(d_hidden, d_out, has_bias=False)

    def construct(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class AggregationBlock(nn.Cell):
    def __init__(self, attention, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.attention = attention
        if attention:
            self.attention_layer = CrossAttention(q_dim, kv_dim, hidden_dim, num_heads, attention_bias)
        else:
            self.attention_layer = MLP(kv_dim, q_dim, q_dim)

    def construct(
        self,
        vision_latents, queries, attention_mask
    ):
        if self.attention:
            queries = self.attention_layer(vision_latents, queries, attention_mask)
        else:
            queries = self.attention_layer(vision_latents)

        return queries


class MultiKVCrossAttention(nn.Cell):

    def __init__(self, q_dim, kv_dim_list, hidden_dim, num_heads, attention_bias=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.SequentialCell([
            nn.LayerNorm([q_dim]), nn.Dense(q_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
        ])
        self.num_of_kvs = len(kv_dim_list)

        self.k_projs, self.v_projs = nn.CellList([]), nn.CellList([])
        for i, kv_dim in enumerate(kv_dim_list):
            # setattr(self, 'k_proj_{}'.format(i), nn.SequentialCell([
            #     nn.LayerNorm([kv_dim]), nn.Dense(kv_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
            # ]))
            # setattr(self, 'v_proj_{}'.format(i), nn.SequentialCell([
            #     nn.LayerNorm([kv_dim]), nn.Dense(kv_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
            # ]))
            self.k_projs.append(
                nn.SequentialCell([
                    nn.LayerNorm([kv_dim]), nn.Dense(kv_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
                ])
            )
            self.v_projs.append(
                nn.SequentialCell([
                    nn.LayerNorm([kv_dim]), nn.Dense(kv_dim, self.num_heads * self.head_dim, has_bias=attention_bias)
                ])
            )
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, q_dim, has_bias=attention_bias)

    def construct(
            self,
            queries, *vision_latents_attention_mask_list,
    ):

        vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
        attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]

        bsz, q_len, _ = queries.shape

        query_states = self.q_proj(queries)
        # key_states = ops.cat(
        #     [getattr(self, 'k_proj_{}'.format(i))(vision_latents_list[i]) for i in range(self.num_of_kvs)], axis=1)
        # value_states = ops.cat(
        #     [getattr(self, 'v_proj_{}'.format(i))(vision_latents_list[i]) for i in range(self.num_of_kvs)], axis=1)
        key_states = ops.cat(
            [self.k_projs[i](vision_latents_list[i]) for i in range(self.num_of_kvs)], axis=1)
        value_states = ops.cat(
            [self.v_projs[i](vision_latents_list[i]) for i in range(self.num_of_kvs)], axis=1)

        v_len = key_states.shape[1]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapdims(1, 2)
        key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).swapdims(1, 2)
        value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).swapdims(1, 2)

        # if kv_weight is not None:
        #     kv_weight = kv_weight.unsqueeze(1).broadcast_to((-1, self.num_heads, -1, -1))

        attention_mask = ops.cat(attention_mask_list, axis=-1)

        if attention_mask is not None:
            # if attention_mask.shape != (bsz, 1, q_len, v_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.shape}"
            #     )
            assert attention_mask.shape == (bsz, 1, q_len, v_len)

        if attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.swapdims(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


class VisionCrossAttentionLayer(nn.Cell):
    def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, hidden_dim = 1024, layer_idx=0):
        super().__init__()
        num_heads = 16
        self.num_of_kvs = len(kv_dim_list)

        self.proj_context = nn.Dense(context_dim, hidden_dim, has_bias=False)
        self.proj_in = nn.Dense(q_dim+hidden_dim, hidden_dim, has_bias=False)
        # if self.num_of_kvs > 1:
        #     self.weight_mlp = MLP(q_dim+hidden_dim, hidden_dim, self.num_of_kvs)
        #     self.tower_weight = Parameter(ops.zeros((self.num_of_kvs)), name='tower_weight')
        self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

        self.norm = nn.LayerNorm([hidden_dim])

        self.cross_attn = MultiKVCrossAttention(hidden_dim, kv_dim_list, hidden_dim, num_heads)
        self.kv_size_list = kv_size_list

        pos_embeds = []
        for i, kv_size in enumerate(kv_size_list):
            if kv_size > 1:
                # setattr(self, "pos_embed_{}".format(i),
                #         Parameter(Tensor(np.random.randn(kv_size**2, hidden_dim), ms.float32)))
                pos_embeds.append(
                    Parameter(Tensor(np.random.randn(kv_size ** 2, hidden_dim), ms.float32), name=f"pos_embed_{i}")
                )
            else:
                pos_embeds.append(
                    Parameter(Tensor(np.zeros(1), ms.float32), name=f"pos_embed_{i}_buffer", requires_grad=False)
                )
        self.pos_embeds = ParameterTuple(pos_embeds)

    def construct(
        self,
        queries,
        context_feature,
        *vision_latents_attention_mask_list,
    ) -> Tensor:

        residual = queries
        # queries = self.proj_in(queries)
        context_feature = self.proj_context(context_feature)
        # queries = queries + context_feature
        queries = ops.cat([queries, context_feature], -1)

        # if self.num_of_kvs > 1:
        #     kv_weight = self.weight_mlp(queries) # B * 1 * num_tower
        #     kv_weight = kv_weight + self.tower_weight.view(1, 1, -1)
        #     kv_weight = kv_weight.softmax(-1)
        #     kv_number_list = [size**2 for size in self.kv_size_list]
        #     kv_weight = ops.repeat_interleave(kv_weight, torch.tensor(kv_number_list).to(kv_weight.device), axis=-1)
        # else:
        #     kv_weight = None
        queries = self.proj_in(queries)

        vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
        attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]

        attention_mask_list_reshaped = []
        if attention_mask_list is not None:
            for attention_mask in attention_mask_list:
                attention_mask = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
                # attention_mask = attention_mask.broadcast_to((-1, -1, queries.shape[1], -1))
                attention_mask = ops.repeat_interleave(attention_mask.to(ms.int32), queries.shape[1], 2).to(attention_mask.dtype)
                attention_mask_list_reshaped.append(attention_mask)

        vision_latents_pos_list = []
        for i, vision_latents in enumerate(vision_latents_list):
            if vision_latents.shape[1] > 1:
                # vision_latents_pos_list.append(vision_latents + getattr(self, "pos_embed_{}".format(i))[None, :, :].to(vision_latents.dtype))
                vision_latents_pos_list.append(
                    vision_latents + self.pos_embeds[i][None, :, :].to(vision_latents.dtype)
                )
            else:
                vision_latents_pos_list.append(vision_latents)

        # Cross Attention
        attention_output = self.cross_attn(
            queries,
            *vision_latents_pos_list,
            *attention_mask_list_reshaped
        )

        # attention_output = (attention_output * combination_weight).sum(2)
        queries = queries + attention_output

        queries = self.norm(queries)

        queries = self.proj_out(queries)

        queries = queries + residual

        return queries


class VisionAggregationLayer(nn.Cell):
    def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, hidden_dim = 1024, layer_idx=0):
        super().__init__()
        num_heads = 16
        self.num_of_kvs = len(kv_dim_list)

        self.proj_context = nn.Dense(context_dim, hidden_dim, has_bias=False)
        self.proj_in = nn.Dense(q_dim+hidden_dim, hidden_dim, has_bias=False)

        self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

        self.norm = nn.LayerNorm([hidden_dim])

        if self.num_of_kvs > 1:
            self.weight_mlp = MLP(q_dim+hidden_dim, hidden_dim, self.num_of_kvs)

        pos_embeds, aggregates = [], []
        for i, kv_size in enumerate(kv_size_list):
            if kv_size > 1:
                # setattr(self, "pos_embed_{}".format(i), Parameter(Tensor(np.random.randn(kv_size**2, hidden_dim), ms.float32)))
                # setattr(self, "aggregate_{}".format(i), AggregationBlock(True, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))
                pos_embeds.append(Parameter(Tensor(np.random.randn(kv_size**2, hidden_dim), ms.float32), name=f"pos_embed_{i}"))
                aggregates.append(AggregationBlock(True, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))
            else:
                # setattr(self, "aggregate_{}".format(i), AggregationBlock(False, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))
                aggregates.append(AggregationBlock(False, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))
        self.pos_embeds = ParameterTuple(pos_embeds)
        self.aggregates = nn.CellList(aggregates)

    def construct(
        self,
        queries,
        context_feature,
        *vision_latents_attention_mask_list,
    ) -> Tensor:

        residual = queries
        # queries = self.proj_in(queries)
        context_feature = self.proj_context(context_feature)
        # queries = queries + context_feature
        queries = ops.cat([queries, context_feature], -1)

        if self.num_of_kvs > 1:
            combination_weight = self.weight_mlp(queries).softmax(-1) # B * 1 * num_tower
            combination_weight = combination_weight.unsqueeze(-1)
        else:
            combination_weight = 1

        queries = self.proj_in(queries)

        vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
        attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]

        attention_mask_list_reshaped = []
        if attention_mask_list is not None:
            for attention_mask in attention_mask_list:
                attention_mask = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
                attention_mask = attention_mask.broadcast_to((-1, -1, queries.shape[1], -1))
                attention_mask_list_reshaped.append(attention_mask)

        vision_latents_pos_list = []
        for i, vision_latents in enumerate(vision_latents_list):
            if vision_latents.shape[1] > 1:
                # vision_latents_pos_list.append(vision_latents + getattr(self, "pos_embed_{}".format(i))[None, :, :].to(vision_latents.dtype))
                vision_latents_pos_list.append(vision_latents + self.pos_embeds[i][None, :, :].to(vision_latents.dtype))
            else:
                vision_latents_pos_list.append(vision_latents)

        aggregated_vision_latents_list = []
        for i, (vision_latents, attention_mask) in enumerate(zip(vision_latents_pos_list, attention_mask_list_reshaped)):
            # aggregated_vision_latents_list.append(getattr(self, "aggregate_{}".format(i))(vision_latents, queries, attention_mask))
            aggregated_vision_latents_list.append(self.aggregates[i](vision_latents, queries, attention_mask))

        aggregated_vision_latents = ops.stack(aggregated_vision_latents_list, 2)

        queries = queries + (aggregated_vision_latents * combination_weight).sum(2)

        queries = self.norm(queries)

        queries = self.proj_out(queries)

        queries = queries + residual

        return queries


class VisionTokenSampler(nn.Cell):
    def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, num_of_layers=1, layer_type="joint"):
        super().__init__()
        assert layer_type in ['joint', 'sep']
        if layer_type == 'joint':
            self.layers = nn.CellList([
                VisionCrossAttentionLayer(q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, idx)
                for idx in range(num_of_layers)
            ])
        else:
            self.layers = nn.CellList([
                VisionAggregationLayer(q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, idx)
                for idx in range(num_of_layers)
            ])

    def construct(self, queries, context_feature, *vision_latents_attention_mask_list):
        for layer in self.layers:
            queries = layer(queries, context_feature, *vision_latents_attention_mask_list)
        return queries
