import argparse
import os
import json
import math
import random
import re
import shortuuid
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from ezcolorlog import root_logger as logger

import mindspore as ms
from mindspore import context, Tensor

from transformers import AutoTokenizer

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from cambrian.model.language_model.cambrian_llama import CambrianLlamaForCausalLM


def load_model_and_process(model_path, model_base, model_name, use_flash_attn=False, **kwargs):

    kwargs['torch_dtype'] = ms.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention'

    if 'cambrian' in model_name.lower():
        # Load Cambrian model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
            raise NotImplementedError
        if 'lora' in model_name.lower() and model_base is not None:
            raise NotImplementedError
        elif model_base is not None:
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
                model = CambrianLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
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
            vision_tower_aux.to(ms.float16)

        image_processor = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


# cambrian-8b
conv_mode = "llama_3"


def process(image, question, tokenizer, image_processor, model_config):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensor = process_images([image], image_processor, model_config)
    image_tensor = tuple([Tensor(i) for i in image_tensor])
    # FIXME: unpad image input
    # image_size = [image.size]
    image_size = [i.shape for i in image_tensor]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='ms').unsqueeze(0)

    return input_ids, image_tensor, image_size, prompt


def test_cambrian_8b_inference(args):
    print(f"=====> test_cambrian_8b_inference:")
    print(f"=====> Building model...")

    model_path = args.model_path
    image_path = args.image_path
    question = args.question

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = \
        load_pretrained_model(model_path, None, model_name, checkpoint_path=args.checkpoint_path)

    print(f"=====> Building model done.")

    temperature = 0
    num_step = 1

    for _step in range(num_step):
        image = Image.open(image_path).convert('RGB')

        input_ids, image_tensor, image_sizes, prompt = process(image, question, tokenizer, image_processor,
                                                               model.config)

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            num_beams=1,
            max_new_tokens=512, # 512,
            use_cache=False,  #True,
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> step: {_step}/{num_step}, input prompt: {prompt}")
        print(f"=====> step: {_step}/{num_step}, output result: {outputs}")
        print(f"=====> step: {_step}/{num_step}, Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=1, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--image_path", type=str, default="./demo/math.png")
    parser.add_argument("--question", type=str, default="Please solve this question step by step.")
    parser.add_argument("--checkpoint_path", type=str, default="./cambrian-8b.ckpt")
    args, _ = parser.parse_known_args()

    if args.ms_mode == 0:
        # FIXME: OOM on 9-th step
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})
    elif args.ms_mode == 1:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", pynative_synchronize=True)
    else:
        raise ValueError

    test_cambrian_8b_inference(args)
