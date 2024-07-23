import argparse
import os
import json
import math
import random
import re
import shortuuid
import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import context

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
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

    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='ms').unsqueeze(0)

    return input_ids, image_tensor, image_size, prompt


def inference():
    seed = 42
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_path = os.path.expanduser("nyu-visionx/cambrian-8b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    temperature = 0

    while True:
        image_path = input("image path: ")
        image = Image.open(image_path).convert('RGB')
        question = input("question: ")

        input_ids, image_tensor, image_sizes, prompt = process(image, question, tokenizer, image_processor,
                                                               model.config)
        input_ids = input_ids.to(non_blocking=True)

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            num_beams=1,
            max_new_tokens=512,
            use_cache=False  #True
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(outputs)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    inference()
