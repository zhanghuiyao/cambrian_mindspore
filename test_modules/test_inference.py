import ast
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

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


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
    image_size = [image.size]  # zhy_test infer, breakpoint()
    # image_size = [image_tensor[0].shape[-2:]]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='np')[None, ...]

    return input_ids, image_tensor, image_size, prompt


def test_cambrian_8b_inference(args):
    print(f"=====> test_cambrian_8b_inference:")
    print(f"=====> Building model...")

    model_path = args.model_path
    image_path = args.image_path
    question = args.question

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = \
        load_pretrained_model(
            model_path, None, model_name, use_flash_attn=args.use_fa, checkpoint_path=args.checkpoint_path)

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
    parser.add_argument("--jit_level", type=str, default="O0")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--image_path", type=str, default="./demo/math.png")
    parser.add_argument("--question", type=str, default="Please solve this question step by step.")
    parser.add_argument("--checkpoint_path", type=str, default=None)  #"./cambrian-8b.ckpt")
    parser.add_argument("--use_fa", type=ast.literal_eval, default=True)
    args, _ = parser.parse_known_args()

    if args.ms_mode == 0:
        if os.environ.get("MS_DEV_RUNTIME_CONF") is None:
            os.environ["MS_DEV_RUNTIME_CONF"] = "synchronize:True"
            print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")
        else:
            if "synchronize:True" not in os.environ.get("MS_DEV_RUNTIME_CONF"):
                _old = os.environ.get("MS_DEV_RUNTIME_CONF")
                _old.replace("synchronize:False,", "")
                _old.replace(",synchronize:False", "")
                _old.replace("synchronize:False", "")
                _new = "synchronize:True," + _old if len(_old) > 0 else "synchronize:True"
                os.environ["MS_DEV_RUNTIME_CONF"] = _new
                print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")

        ms.set_context(
            mode=ms.GRAPH_MODE,
            device_target="Ascend",
            jit_config={"jit_level": args.jit_level},
            max_device_memory="59GB",
            deterministic="ON"
        )

    elif args.ms_mode == 1:
        ms.set_context(
            mode=ms.PYNATIVE_MODE,
            device_target="Ascend",
            pynative_synchronize=True,
            max_device_memory="59GB",
            deterministic="ON"
        )
    else:
        raise ValueError

    test_cambrian_8b_inference(args)
