from typing import List, Optional, Tuple, Union

import mindspore as ms
import numpy as np
from mindspore import nn, ops, Tensor, Parameter, ParameterTuple

from transformers import GenerationConfig
from transformers.utils import logging

from cambrian.transformers.models.llama import LlamaConfig, LlamaModel, LlamaForCausalLM
from cambrian.transformers.cache_utils import Cache, DynamicCache, StaticCache
from cambrian.transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from cambrian.model.cambrian_arch import CambrianMetaModel, CambrianMetaForCausalLM
from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from cambrian.mindspore_adapter import auto_mixed_precision, recompute_except_output


logger = logging.get_logger(__name__)


EXPAND_FOR_BATCH = True


class CambrianConfig(LlamaConfig):
    model_type = "cambrian_llama"

    debug = "debug"


class CambrianLlamaModel(CambrianMetaModel, LlamaModel):
    config_class = CambrianConfig

    def __init__(self, config):
        super(CambrianLlamaModel, self).__init__(config)

        _name_list = [
            'output_attentions', 'output_hidden_states', 'use_cache',
            "connector_only", "start_of_vision_sampler_layers", "stride_of_vision_sampler_layers",
            "image_position", "image_token_len", "query_num_list", "mm_projector_type"
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_tower_aux_feature_list: Optional[List[Tensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[Tensor]] = None,
        final_vision_feature_size: Optional[List[tuple]] = None,
        global_context_feature: Optional[Tensor] = None,
    ) -> Union[Tuple]:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.use_cache

        # assert not output_attentions
        # assert not output_hidden_states

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        # assert not use_cache, "NotImplementedError"  # TODO
        # if use_cache:
        #     use_legacy_cache = not isinstance(past_key_values, Cache)
        #     if use_legacy_cache:
        #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        #     past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=ms.int32
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # embed positions
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):

            # zhy_test infer, breakpoint()
            # np.save(f"hidden_states_in_{i}.npy", hidden_states.asnumpy())
            # ops.TensorDump()(f"hidden_states_in_{i}.npy", hidden_states)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            # zhy_test infer, breakpoint()
            # np.save(f"hidden_states_out_{i}.npy", hidden_states.asnumpy())
            # ops.TensorDump()(f"hidden_states_out_{i}.npy", hidden_states)

            if not self.connector_only:

                cross_layers_start_idx = self.start_of_vision_sampler_layers
                cross_index_step = self.stride_of_vision_sampler_layers
                cross_layers_start_idx_list = [
                    cross_layers_start_idx + cross_index * cross_index_step
                    for cross_index in range(len(self.vision_sampler_layers))
                ]

                if vision_tower_aux_feature_list is not None and i in cross_layers_start_idx_list:
                    latent_query_start_idx = self.image_position

                    # if EXPAND_FOR_BATCH:
                    if self.training:
                        image_token_len_per_side = int(self.image_token_len ** 0.5)
                        latent_query_newline_num = self.image_token_len + image_token_len_per_side
                        latent_query_num = self.image_token_len
                        latent_query_with_newline = hidden_states[:, latent_query_start_idx:latent_query_start_idx+latent_query_newline_num, :]
                        bs = latent_query_with_newline.shape[0]
                        latent_query_with_newline = latent_query_with_newline.view(bs, image_token_len_per_side, image_token_len_per_side+1, -1)
                        latent_query = latent_query_with_newline[:, :, :-1, :]
                        newline_embd = latent_query_with_newline[:, :, -1:, :]
                        vision_tower_aux_feature_list = [vision_tower_aux_feature.to(latent_query.dtype) for vision_tower_aux_feature in vision_tower_aux_feature_list]
                        bs = latent_query.shape[0]
                        latent_query = latent_query.view(bs*latent_query_num, 1, -1)

                        latent_query = self.vision_sampler_layers[(i-cross_layers_start_idx)//cross_index_step](
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list
                        )

                        # latent_query = latent_query.view(bs, self.latent_query_num, -1)
                        latent_query = latent_query.view(bs, image_token_len_per_side, image_token_len_per_side, -1)
                        latent_query_with_newline = ops.cat([latent_query, newline_embd], 2).flatten(start_dim=1, end_dim=2)
                        hidden_states[
                            :,
                            latent_query_start_idx:latent_query_start_idx+latent_query_newline_num
                        ] = \
                            latent_query_with_newline[:, :, :]
                    else:

                        # 1. original implement
                        # bs = len(final_vision_feature_size)
                        # latent_query_num_list = []
                        # newline_embd_list = []
                        # latent_query_list = []
                        # for batch_i in range(bs):
                        #     cur_h, cur_w = final_vision_feature_size[batch_i]
                        #
                        #     cur_latent_query_num = cur_h*cur_w
                        #     cur_latent_query_newline_num = cur_h * (cur_w+1)
                        #     cur_latent_query_with_newline = \
                        #         hidden_states[
                        #             batch_i:batch_i+1,
                        #             latent_query_start_idx:latent_query_start_idx+cur_latent_query_newline_num,
                        #             :
                        #         ]
                        #
                        #     cur_latent_query_with_newline = cur_latent_query_with_newline.view(1, cur_h, cur_w+1, -1)
                        #     cur_latent_query = cur_latent_query_with_newline[:, :, :-1, :]
                        #     cur_newline_embd = cur_latent_query_with_newline[:, :, -1:, :]
                        #
                        #     latent_query_num_list.append(cur_latent_query_num)
                        #     latent_query_list.append(cur_latent_query.view(cur_latent_query_num, 1, -1))
                        #     newline_embd_list.append(cur_newline_embd)
                        # latent_query = ops.cat(latent_query_list, 0)
                        # latent_query = self.vision_sampler_layers[(i-cross_layers_start_idx)//cross_index_step](
                        #     latent_query,
                        #     global_context_feature,
                        #     *vision_tower_aux_feature_list,
                        #     *vision_tower_aux_attention_masks_list
                        # )
                        #
                        # latent_query = ops.split(latent_query, latent_query_num_list, 0)
                        # for batch_i in range(bs):
                        #     cur_h, cur_w = final_vision_feature_size[batch_i]
                        #     cur_latent_query = latent_query[batch_i]
                        #     cur_newline_embd = newline_embd_list[batch_i]
                        #     cur_latent_query_newline_num = cur_h * (cur_w+1)
                        #     cur_latent_query = cur_latent_query.view(1, cur_h, cur_w, -1)
                        #     cur_latent_query_with_newline = ops.cat([cur_latent_query, cur_newline_embd], 2).flatten(start_dim=1, end_dim=2)
                        #     hidden_states[
                        #         batch_i:batch_i+1,
                        #         latent_query_start_idx:latent_query_start_idx+cur_latent_query_newline_num
                        #     ] = \
                        #         cur_latent_query_with_newline[:, :, :]

                        # 2. new implement, for avoid dynamic shape
                        bs = len(final_vision_feature_size)
                        latent_query_num_list = []
                        newline_embd_list = []
                        latent_query_list = []

                        final_h = final_w = int(self.image_token_len ** 0.5)
                        max_latent_query_num = final_h * final_w
                        max_latent_query_newline_num = final_h * (final_w + 1)

                        for batch_i in range(bs):

                            cur_h, cur_w = final_vision_feature_size[batch_i]
                            cur_latent_query_num = cur_h * cur_w
                            cur_latent_query_newline_num = cur_h * (cur_w + 1)

                            padded_latent_query_with_newline = \
                                hidden_states[
                                    batch_i,
                                    latent_query_start_idx:latent_query_start_idx + max_latent_query_newline_num,
                                    :
                                ]

                            # cur_latent_query_with_newline = cur_latent_query_with_newline.view(1, cur_h, cur_w + 1, -1)
                            # cur_latent_query = cur_latent_query_with_newline[:, :, :-1, :]
                            # cur_newline_embd = cur_latent_query_with_newline[:, :, -1:, :]

                            # gather latent_query
                            _index_table = ops.arange(0, max_latent_query_num)
                            _gather_index_1 = _index_table + (_index_table // cur_w)
                            _gather_index_1 = ops.clip(_gather_index_1, Tensor(0, ms.int32), Tensor(cur_latent_query_newline_num - 1, ms.int32))
                            padded_latent_query = ops.gather(padded_latent_query_with_newline, _gather_index_1, axis=0)

                            # gather newline_embd
                            _gather_index_2 = ops.arange(1, final_h + 1) * (cur_w + 1) - 1
                            _gather_index_2 = ops.clip(_gather_index_2, Tensor(0, ms.int32), Tensor(cur_latent_query_newline_num - 1, ms.int32))
                            padded_newline_embd = ops.gather(padded_latent_query_with_newline, _gather_index_2, axis=0)

                            latent_query_num_list.append(cur_latent_query_num)
                            latent_query_list.append(padded_latent_query.view(max_latent_query_num, 1, -1))
                            newline_embd_list.append(padded_newline_embd.view(1, final_h, 1, -1))

                        latent_query = ops.cat(latent_query_list, 0)
                        latent_query = self.vision_sampler_layers[(i - cross_layers_start_idx) // cross_index_step](
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list
                        )

                        latent_query = latent_query.view(bs, max_latent_query_num, 1, -1)
                        for batch_i in range(bs):
                            cur_h, cur_w = final_vision_feature_size[batch_i]
                            cur_latent_query_newline_num = cur_h * (cur_w + 1)
                            padded_latent_query = latent_query[batch_i].view(max_latent_query_num, -1)   # (max_latent_query_num, -1)
                            padded_newline_embd = newline_embd_list[batch_i].view(final_h * 1, -1)       # (final_h, -1)
                            padded_latent_query_with_newline = ops.concat((padded_latent_query, padded_newline_embd), axis=0)

                            _index_table = ops.arange(0, max_latent_query_newline_num)
                            _gather_index_1 = _index_table - _index_table // (cur_w + 1)
                            _gather_index_2 = (_index_table + 1) // (cur_w + 1) - 1 + max_latent_query_num
                            _gather_index = ops.select(
                                (_index_table + 1) % (cur_w + 1) == 0,
                                _gather_index_2,
                                _gather_index_1
                            )
                            padded_latent_query_with_newline = ops.gather(padded_latent_query_with_newline, _gather_index, axis=0)
                            padded_latent_query_with_newline = padded_latent_query_with_newline.view(max_latent_query_newline_num, -1)


                            _index_table = ops.arange(0, hidden_states.shape[1])
                            _gather_index_query = _index_table - latent_query_start_idx + hidden_states.shape[1]
                            _gather_index = ops.select(
                                ops.logical_or(
                                    _index_table < latent_query_start_idx,
                                    _index_table >= latent_query_start_idx + cur_latent_query_newline_num
                                ),
                                _index_table,
                                _gather_index_query
                            )
                            hidden_states_with_latent_query = ops.concat((hidden_states[batch_i], padded_latent_query_with_newline), axis=0)
                            cur_hidden_states = ops.gather(hidden_states_with_latent_query, _gather_index, axis=0)
                            hidden_states[batch_i] = cur_hidden_states

            if use_cache:
                # next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
                next_cache = layer_outputs[1]
                past_key_values[i] = next_cache

        hidden_states = self.norm(hidden_states)

        # zhy_test infer, breakpoint()
        # np.save(f"hidden_states_out.npy", hidden_states.asnumpy())
        # ops.TensorDump()(f"hidden_states_out.npy", hidden_states)

        outputs = (hidden_states,)
        if use_cache and past_key_values is not None:
            outputs += (past_key_values,)

        # last_hidden_state, past_key_values, hidden_states, attentions
        return outputs


class CambrianLlamaForCausalLM(LlamaForCausalLM, CambrianMetaForCausalLM):
    config_class = CambrianConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = CambrianLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.use_cache = config.use_cache
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # preprocess input_ids
        self.tokenizer_model_max_length = config.tokenizer_model_max_length
        self.tokenizer_padding_side = config.tokenizer_padding_side

        # for inference
        self.vision_tower_aux_feature_list = None

        # for train
        self.loss_fct = nn.CrossEntropyLoss()

        # TODO: Initialize weights and apply final processing
        # self.post_init()

    def set_use_cache(self, use_cache: bool):
        self.use_cache = use_cache
        self.model.use_cache = use_cache

    def get_model(self):
        return self.model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            # gradient_checkpointing_kwargs = {"mp_comm_recompute": True, "parallel_optimizer_comm_recompute": True}
            gradient_checkpointing_kwargs = {}

        # 1. visual encoders
        if hasattr(self.model, "vision_tower_aux_list"):
            for vision_tower in self.model.vision_tower_aux_list:
                for name, cell in vision_tower.name_cells().items():
                    if "output_identity" in name:
                        assert isinstance(cell, nn.Identity)
                        continue
                    else:
                        recompute_except_output(cell, **gradient_checkpointing_kwargs)

        # 2. mm projector and vision samplers
        if hasattr(self.model, "mm_projector"):
            if isinstance(self.model.mm_projector, nn.SequentialCell):
                for cell in self.model.mm_projector.cell_list[:-1]:
                    recompute_except_output(cell)
            else:
                recompute_except_output(self.model.mm_projector, **gradient_checkpointing_kwargs)
        if hasattr(self.model, "mm_projector_auxes"):
            for cell in self.model.mm_projector_auxes:
                for sub_cell in cell.cell_list[:-1]:
                    recompute_except_output(sub_cell, **gradient_checkpointing_kwargs)
        if hasattr(self.model, "vision_samplers"):
            for vision_sampler in self.model.vision_samplers:
                for _, cell in vision_sampler.name_cells().items():
                    recompute_except_output(cell, **gradient_checkpointing_kwargs)

        # 3. llama layers
        from cambrian.transformers.models.llama.modeling_llama import LlamaDecoderLayer
        for decoder_layer in self.model.layers:
            assert isinstance(decoder_layer, LlamaDecoderLayer)
            for name, cell in decoder_layer.name_cells().items():
                if "output_identity" in name:
                    assert isinstance(cell, nn.Identity)
                    continue
                else:
                    # cell._recompute()
                    recompute_except_output(cell, **gradient_checkpointing_kwargs)
        recompute_except_output(self.model.embed_tokens, **gradient_checkpointing_kwargs)
        recompute_except_output(self.model.norm, **gradient_checkpointing_kwargs)

        # 4. vision sampler layers
        if hasattr(self.model, "vision_sampler_layers"):
            for cell in self.model.vision_sampler_layers:
                recompute_except_output(cell, **gradient_checkpointing_kwargs)

        # 5. cambrian head
        # recompute_except_output(self.lm_head, **gradient_checkpointing_kwargs)

        logger.info(f"{self.__class__.__name__}: enable recompute done.")

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[List[Tensor]] = None,
            image_aux_attention_masks_list: Optional[List[Tensor]] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            cache_position=None,
            final_vision_feature_size=None,  # for infer
    ):

        vision_tower_aux_feature_list = None
        vision_tower_aux_attention_masks_list = None
        final_vision_feature_size = final_vision_feature_size
        global_context_feature = None

        if inputs_embeds is None:
            (
                _,
                position_ids,
                attention_mask,
                inputs_embeds,
                labels,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                labels,
                images,
                image_aux_attention_masks_list,
                image_sizes
            )
            input_ids = None

            position_ids = ops.stop_gradient(position_ids)
            attention_mask = ops.stop_gradient(attention_mask)
            labels = ops.stop_gradient(labels)
            vision_tower_aux_attention_masks_list = ops.stop_gradient(vision_tower_aux_attention_masks_list)
            final_vision_feature_size = ops.stop_gradient(final_vision_feature_size)

        # training
        if self.training:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vision_tower_aux_feature_list=vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list,
                final_vision_feature_size=final_vision_feature_size,
                global_context_feature=global_context_feature,
            )

        # inference
        else:
            if self.vision_tower_aux_feature_list is not None:
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

                if inputs_embeds is not None:
                    vision_tower_aux_feature_list = self.vision_tower_aux_feature_list
                    vision_tower_aux_attention_masks_list = self.vision_tower_aux_attention_masks_list
                    # final_vision_feature_size = self.final_vision_feature_size
                    global_context_feature = self.global_context_feature

                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    vision_tower_aux_feature_list=vision_tower_aux_feature_list,
                    vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list,
                    final_vision_feature_size=final_vision_feature_size,
                    global_context_feature=global_context_feature,
                )
            else:
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        hidden_states = model_outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, axis=0)
            logits = [ops.dense(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = ops.cat(logits, axis=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.to(ms.float32)

        loss = ops.zeros((), dtype=logits.dtype)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = self.loss_fct(shift_logits, shift_labels)

        outputs = (loss, logits)
        if past_key_values is not None:
            past_key_values = model_outputs[1]
            outputs += (past_key_values,)

        # loss, logits, past_key_values, hidden_states, attentions
        return outputs

    def generate(
            self,
            inputs: Optional[np.ndarray] = None,
            images: Optional[Tuple[Tensor]] = None,
            image_sizes: Optional[Tuple] = None,
            position_ids = None,
            attention_mask = None,
            inputs_embeds = None,
            **kwargs,
    ) -> Union[Tuple, Tensor]:

        # if "inputs_embeds" in kwargs:
        #     raise NotImplementedError("`inputs_embeds` is not supported")
        assert inputs_embeds is None
        final_vision_feature_size = None

        # zhy_test infer
        # breakpoint()

        inputs, _, position_ids, attention_mask = \
            self.preprocess_input_before_generate_numpy(inputs, None, position_ids, attention_mask)

        # zhy_test infer
        # breakpoint()

        if images is not None:

            (
                inputs,
                position_ids,
                attention_mask,
                inputs_embeds,
                _,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            ) = self.prepare_inputs_labels_for_multimodal_(
                inputs,
                position_ids,
                attention_mask,
                None,
                *images,
                *image_sizes,
            )


            # Do pad
            max_per_context_token_len = self.model.image_token_len
            # final_vision_feature_size: ((cur_h, cur_w), ...)
            cur_context_token_len = sum([(_h * _w) for _h, _w in final_vision_feature_size])
            assert global_context_feature.shape[0] == cur_context_token_len
            bs, num_vision_tower = inputs_embeds.shape[0], len(vision_tower_aux_feature_list)
            context_shape, context_dtype = global_context_feature.shape, global_context_feature.dtype
            # (cur_h*cur_w+..., 1, -1) -> (bs, final_h*final_w, 1, -1)
            padded_global_context_feature = ops.ones((bs, max_per_context_token_len, *context_shape[1:]), dtype=context_dtype)
            # 4*(cur_h*cur_w+..., factor, -1) -> 4*(bs, final_h*final_w, factor, -1)
            padded_vision_tower_aux_feature_list, padded_vision_tower_aux_attention_masks_list = [], []
            for i in range(num_vision_tower):
                _shape1, _dtype1 = vision_tower_aux_feature_list[i].shape, vision_tower_aux_feature_list[i].dtype
                _shape2, _dtype2 = vision_tower_aux_attention_masks_list[i].shape, vision_tower_aux_attention_masks_list[i].dtype
                padded_vision_tower_aux_feature_list.append(ops.ones((bs, max_per_context_token_len, *_shape1[1:]), dtype=_dtype1))
                padded_vision_tower_aux_attention_masks_list.append(ops.ones((bs, max_per_context_token_len, *_shape2[1:]), dtype=_dtype2))
            _cur_pos = 0
            for idx_bs in range(bs):
                cur_h, cur_w = final_vision_feature_size[idx_bs]
                _size = cur_w * cur_h
                padded_global_context_feature[idx_bs, :_size] = global_context_feature[_cur_pos:_cur_pos+_size]
                for idx_vision in range(num_vision_tower):
                    padded_vision_tower_aux_feature_list[idx_vision][idx_bs, :_size] = vision_tower_aux_feature_list[idx_vision][_cur_pos:_cur_pos+_size]
                    padded_vision_tower_aux_attention_masks_list[idx_vision][idx_bs, :_size] = vision_tower_aux_attention_masks_list[idx_vision][_cur_pos:_cur_pos+_size]
                _cur_pos = _cur_pos + _size
            global_context_feature = padded_global_context_feature.view(-1, *context_shape[1:])
            vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list = [], []
            for i in range(num_vision_tower):
                _shape1, _shape2 = padded_vision_tower_aux_feature_list[i].shape, padded_vision_tower_aux_attention_masks_list[i].shape
                vision_tower_aux_feature_list.append(padded_vision_tower_aux_feature_list[i].view(-1, *_shape1[2:]))
                vision_tower_aux_attention_masks_list.append(padded_vision_tower_aux_attention_masks_list[i].view(-1, *_shape2[2:]))

            # _dtype = vision_tower_aux_feature_list[0].dtype
            # vision_tower_aux_feature_list = [
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_feature_list_0_pt.npy"), _dtype),
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_feature_list_1_pt.npy"), _dtype),
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_feature_list_2_pt.npy"), _dtype),
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_feature_list_3_pt.npy"), _dtype),
            # ]
            # vision_tower_aux_attention_masks_list = [
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_attention_masks_list_0_pt.npy"), ms.bool_),
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_attention_masks_list_1_pt.npy"), ms.bool_),
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_attention_masks_list_2_pt.npy"), ms.bool_),
            #     Tensor(np.load("./pt_tensors/_vision_tower_aux_attention_masks_list_3_pt.npy"), ms.bool_),
            # ]
            # final_vision_feature_size = np.load("./pt_tensors/_final_vision_feature_size_pt.npy").tolist()
            # global_context_feature = Tensor(np.load("./pt_tensors/_global_context_feature_pt.npy"), _dtype)
            # inputs_embeds = Tensor(np.load("./pt_tensors/_inputs_embeds_pt_fixed.npy"), _dtype)
            # position_ids = None
            # attention_mask = Tensor(np.load("./pt_tensors/mask.npy"), ms.bool_)
            # breakpoint()

            if hasattr(self, "vision_tower_aux_feature_list") and isinstance(self.vision_tower_aux_feature_list, (Parameter, ParameterTuple)):
                vision_feature_num = len(vision_tower_aux_feature_list)

                assert len(self.vision_tower_aux_feature_list) == len(vision_tower_aux_feature_list)
                assert len(self.vision_tower_aux_attention_masks_list) == len(vision_tower_aux_attention_masks_list)

                for i in range(vision_feature_num):
                    ops.assign(self.vision_tower_aux_feature_list[i], vision_tower_aux_feature_list[i])
                    ops.assign(self.vision_tower_aux_attention_masks_list[i], vision_tower_aux_attention_masks_list[i])
                ops.assign(self.global_context_feature, global_context_feature)

            else:
                vision_feature_num = len(vision_tower_aux_feature_list)

                self.vision_tower_aux_feature_list = ParameterTuple([Parameter(vision_tower_aux_feature_list[_i],
                                                                               name=f"infer_vision_tower_aux_feature_list_{_i}")
                                                                     for _i in range(vision_feature_num)])
                self.vision_tower_aux_attention_masks_list = ParameterTuple([Parameter(vision_tower_aux_attention_masks_list[_i],
                                                                               name=f"infer_vision_tower_aux_attention_masks_list_{_i}")
                                                                             for _i in range(vision_feature_num)])
                self.global_context_feature = Parameter(global_context_feature, name="infer_global_context_feature")

        else:
            inputs_embeds = self.model.embed_tokens(inputs)

        # zhy_test infer
        # breakpoint()

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            final_vision_feature_size=Tensor(final_vision_feature_size, ms.int32),
            **kwargs
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            images=None,
            image_sizes=None,
            **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        if kwargs.get("final_vision_feature_size", None) is not None:
            inputs['final_vision_feature_size'] = kwargs['final_vision_feature_size']

        return inputs


class TrainWrapperForCambrianLlamaForCausalLM(nn.Cell):
    def __init__(self, network):
        super().__init__(auto_prefix=False)

        assert isinstance(network, CambrianLlamaForCausalLM)

        self.cambrian_llama_causal = network
        self.input_image_len = len(network.model.vision_tower_aux_list)

        self.input_keys = [
            "input_ids", "attention_mask", "position_ids", "labels",
            "images", "image_aux_attention_masks_list",
        ]

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            *images_and_masks
    ):
        # assert len(images_and_masks) == self.input_image_len * 2
        # assert self.training

        images = images_and_masks[:self.input_image_len]
        image_aux_attention_masks_list = images_and_masks[self.input_image_len:]

        return self.cambrian_llama_causal(
            input_ids,
            attention_mask,
            position_ids,
            None,
            None,
            labels,
            None,
            None,
            None,
            images,
            image_aux_attention_masks_list,
            None,
            None,
            None
        )[0]
