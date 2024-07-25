from typing import List, Optional, Tuple, Union

import mindspore as ms
import numpy as np
from mindspore import nn, ops, Tensor, Parameter

from transformers import GenerationConfig
from transformers.utils import logging

from cambrian.transformers.models.llama import LlamaConfig, LlamaModel, LlamaForCausalLM
from cambrian.transformers.cache_utils import Cache, DynamicCache, StaticCache
from cambrian.transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from cambrian.model.cambrian_arch import CambrianMetaModel, CambrianMetaForCausalLM
from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


logger = logging.get_logger(__name__)


EXPAND_FOR_BATCH = True


class CambrianConfig(LlamaConfig):
    model_type = "cambrian_llama"

    debug = "debug"


class CambrianLlamaModel(CambrianMetaModel, LlamaModel):
    config_class = CambrianConfig

    def __init__(self, config):
        self.config = config
        super(CambrianLlamaModel, self).__init__(config)

        _name_list = [
            'output_attentions', 'output_hidden_states', 'use_cache',
            "connector_only", "start_of_vision_sampler_layers", "stride_of_vision_sampler_layers",
            "image_position", "image_token_len", "query_num_list", "mm_projector_type"
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        config, _ = CambrianConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=None,
            return_unused_kwargs=True,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            token=None,
            revision="main",
            subfolder="",
            _from_auto=False,
            _from_pipeline=None,
        )
        return cls(config)

    @ms.jit
    def construct(
        self,
        input_ids: Tensor = None,
        input_ids_mask: Tensor = None,
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

        assert not output_attentions
        assert not output_hidden_states

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
        assert not use_cache, "NotImplementedError"  # TODO
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
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states, _, next_cache = layer_outputs

            if not self.connector_only:

                cross_layers_start_idx = self.start_of_vision_sampler_layers
                cross_index_step = self.stride_of_vision_sampler_layers
                cross_layers_start_idx_list = [
                    cross_layers_start_idx + cross_index * cross_index_step
                    for cross_index in range(len(self.vision_sampler_layers))
                ]

                if vision_tower_aux_feature_list is not None and i in cross_layers_start_idx_list:
                    latent_query_start_idx = self.image_position

                    if EXPAND_FOR_BATCH:
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
                        hidden_states[:, latent_query_start_idx:latent_query_start_idx+latent_query_newline_num] = latent_query_with_newline[:, :, :]
                    else:
                        bs = len(final_vision_feature_size)
                        latent_query_num_list = []
                        newline_embd_list = []
                        latent_query_list = []
                        for batch_i in range(bs):
                            cur_h, cur_w = final_vision_feature_size[batch_i]
                    
                            cur_latent_query_num = cur_h*cur_w
                            cur_latent_query_newline_num = cur_h * (cur_w+1)
                            cur_latent_query_with_newline = hidden_states[batch_i:batch_i+1, latent_query_start_idx:latent_query_start_idx+cur_latent_query_newline_num, :]

                            cur_latent_query_with_newline = cur_latent_query_with_newline.view(1, cur_h, cur_w+1, -1)
                            cur_latent_query = cur_latent_query_with_newline[:, :, :-1, :]
                            cur_newline_embd = cur_latent_query_with_newline[:, :, -1:, :]

                            latent_query_num_list.append(cur_latent_query_num)
                            latent_query_list.append(cur_latent_query.contiguous().view(cur_latent_query_num, 1, -1))
                            newline_embd_list.append(cur_newline_embd)

                        latent_query = ops.cat(latent_query_list, 0)

                        latent_query = self.vision_sampler_layers[(i-cross_layers_start_idx)//cross_index_step](
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list
                        )

                        latent_query = ops.split(latent_query, latent_query_num_list, 0)
                        for batch_i in range(bs):
                            cur_h, cur_w = final_vision_feature_size[batch_i]
                            cur_latent_query = latent_query[batch_i]
                            cur_newline_embd = newline_embd_list[batch_i]
                            cur_latent_query_newline_num = cur_h * (cur_w+1)
                            cur_latent_query = cur_latent_query.view(1, cur_h, cur_w, -1)
                            cur_latent_query_with_newline = ops.cat([cur_latent_query, cur_newline_embd], 2).flatten(start_dim=1, end_dim=2)
                            hidden_states[batch_i:batch_i+1, latent_query_start_idx:latent_query_start_idx+cur_latent_query_newline_num] = cur_latent_query_with_newline[:, :, :]

            if use_cache:
                past_key_values[i] = next_cache

        # if use_cache:
        #     next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        hidden_states = self.norm(hidden_states)

        # last_hidden_state, past_key_values, hidden_states, attentions
        return hidden_states, past_key_values


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

        self.embed_tokens = self.get_model().embed_tokens

        # TODO: Initialize weights and apply final processing
        # self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config, _ = CambrianConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=None,
            return_unused_kwargs=True,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            token=None,
            revision="main",
            subfolder="",
            _from_auto=False,
            _from_pipeline=None,
        )

        generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=None,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            token=None,
            revision="main",
            subfolder="",
            _from_auto=False,
            _from_pipeline=None,
        )

        # FIXME: support cache
        if generation_config.use_cache:
            generation_config.use_cache = False
            print(f"not support use_cache, `generation_config.use_cache` force to `False`")

        model = cls(config)
        setattr(model, "generation_config", generation_config)
        setattr(model, "config", config)

        dtype = kwargs.pop("mindspore_dtype", ms.float32)
        if dtype == ms.float16:
            model.to_float(ms.float16)
        elif dtype == "bfloat16":
            model.to_float(ms.bfloat16)

        return model

    def set_use_cache(self, use_cache: bool):
        self.use_cache = use_cache
        self.model.use_cache = use_cache

    def get_model(self):
        return self.model

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
            images: Optional[Tensor] = None,
            image_aux_attention_masks_list: Optional[List[Tensor]] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            cache_position=None
    ) -> Union[Tuple,]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
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
                past_key_values,
                labels,
                images,
                image_aux_attention_masks_list,
                image_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # training
        if self.training:
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
                vision_tower_aux_feature_list=vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list,
                final_vision_feature_size=final_vision_feature_size,
                global_context_feature=global_context_feature,
            )

        # inference
        else:
            if hasattr(self, "vision_tower_aux_feature_list"):
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
                    vision_tower_aux_feature_list=vision_tower_aux_feature_list if inputs_embeds is None else self.vision_tower_aux_feature_list,
                    vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list if inputs_embeds is None else self.vision_tower_aux_attention_masks_list,
                    final_vision_feature_size=final_vision_feature_size if inputs_embeds is None else self.final_vision_feature_size,
                    global_context_feature=global_context_feature if inputs_embeds is None else self.global_context_feature,
                )
            else:
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
                )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, axis=0)
            logits = [ops.dense(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = ops.cat(logits, axis=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.to(ms.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        _, _past_key_values, _hidden_states, _attentions = outputs
        _loss, _logits, _past_key_values, _hidden_states, _attentions = \
            loss, logits, _past_key_values, _hidden_states, _attentions

        return _loss, _logits, _past_key_values, _hidden_states, _attentions

    def generate(
            self,
            inputs: Optional[Tensor] = None,
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

        inputs, _, position_ids, attention_mask, input_ids_mask = \
            self.preprocess_input_before_generate(inputs, None, position_ids, attention_mask)

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                input_ids_mask=input_ids_mask
            )
            self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
            self.vision_tower_aux_attention_masks_list = vision_tower_aux_attention_masks_list
            self.final_vision_feature_size = final_vision_feature_size
            self.global_context_feature = global_context_feature
        else:
            inputs_embeds = self.embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
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
        return inputs
