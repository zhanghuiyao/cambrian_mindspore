#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import time
import warnings
import mindspore as ms
from transformers import AutoTokenizer, AutoConfig

from cambrian.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.model.language_model.cambrian_llama import CambrianLlamaForCausalLM

from ezcolorlog import root_logger as logger


def load_pretrained_model(model_path, model_base, model_name, use_flash_attn=False, **kwargs):

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention'

    checkpoint_path = kwargs.pop("checkpoint_path", None)
    load_8bit = kwargs.pop("load_8bit", False)
    load_4bit = kwargs.pop("load_8bit", False)
    if load_8bit:
        raise NotImplementedError
    elif load_4bit:
        raise NotImplementedError
    else:
        # FIXME: mix-precision
        kwargs['mindspore_dtype'] = ms.float32  # ms.float16

    if 'cambrian' in model_name.lower():
        # Load Cambrian model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
            raise NotImplementedError
        if 'lora' in model_name.lower() and model_base is not None:
            raise NotImplementedError
        elif model_base is not None:
            # this may be mm projector only
            logger.info(f'Loading Cambrian-1 from base model... {model_base}')
            raise NotImplementedError
        else:
            if 'mistral' in model_name.lower():
                # tokenizer = AutoTokenizer.from_pretrained(model_path)
                # model = CambrianMistralForCausalLM.from_pretrained(
                #     model_path,
                #     low_cpu_mem_usage=True,
                #     use_flash_attention_2=False,
                #     **kwargs
                # )
                raise NotImplementedError
            elif 'phi3' in model_name.lower():
                # from cambrian.model.language_model.cambrian_phi3 import CambrianPhi3ForCausalLM
                # tokenizer = AutoTokenizer.from_pretrained(model_path)
                # model = CambrianPhi3ForCausalLM.from_pretrained(
                #     model_path,
                #     low_cpu_mem_usage=True,
                #     use_flash_attention_2=False,
                #     **kwargs
                # )
                raise NotImplementedError
            else:
                logger.info(f'Loading Cambrian from {model_path}')
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = CambrianLlamaForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        raise NotImplementedError

    image_processor = None

    if 'cambrian' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        model.resize_token_embeddings(len(tokenizer))

        vision_tower_aux_list = model.get_vision_tower_aux_list()

        for vision_tower_aux in vision_tower_aux_list:
            if not vision_tower_aux.is_loaded:
                vision_tower_aux.load_model()

            # FIXME: mix-precision
            # vision_tower_aux.to_float(ms.float16)

        image_processor = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if checkpoint_path is not None:
        s_time = time.time()
        print(f"checkpoint loading...")
        state_dict = ms.load_checkpoint(checkpoint_path)
        m, u = ms.load_param_into_net(model, state_dict)

        m = [n for n in m if ("_buffer" not in n) and (".inv_freq" not in n)]
        if len(m) > 0:
            print(f"WARNING: missing keys num: {len(m)}, top 10 name is: {m[:10]}")
        if len(u) > 0:
            print(f"WARNING: unexpected keys num: {len(u)}, top 10 name is: {u[:10]}")

        print(f"load checkpoint from `{checkpoint_path}` success, time cost: {time.time()-s_time:.2f}s")
    else:
        print(f"WARNING: No available pre-trained weights")

    return tokenizer, model, image_processor, context_len
