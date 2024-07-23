from PIL import Image
from io import BytesIO
import numpy as np
import base64
import math
import ast

import mindspore as ms
from mindspore import Tensor

from cambrian.constants import IMAGE_TOKEN_INDEX


# multiple vision towers
def process_images(images, image_processor, model_cfg):
    processor_aux_list = image_processor
    new_images_aux_list = []
    for image in images:
        image_aux_list = []
        for processor_aux in processor_aux_list:
            image_aux = image
            if hasattr(processor_aux, 'image_mean'):
                target_resolution = processor_aux.crop_size['height']
                image_aux = expand2square(image_aux, tuple(int(x*255) for x in processor_aux.image_mean)).resize((target_resolution, target_resolution))
            image_aux = processor_aux.preprocess(image_aux, return_tensors='pt')['pixel_values'][0]
            image_aux_list.append(image_aux)
        new_images_aux_list.append(image_aux_list)
    new_images_aux_list = [list(batch_image_aux) for batch_image_aux in zip(*new_images_aux_list)]
    new_images_aux_list = [np.stack(image_aux) for image_aux in new_images_aux_list]

    return new_images_aux_list


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # offset = 1
    # prompt_chunks = [token1, token2]
    # sep = [-200, -200]
    # [ele for sublist in zip((token1, token2], [[-200, -200], [-200, -200]])) for ele in sublist][:-1]
    # -> [token1, [-200, -200], token2, [-200, -200]][:-1]
    # -> [token1, [-200, -200], token2]
    # -> [bos_token_id, *token1[1:], -200, *token2[1:]]

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'ms':
            return Tensor(input_ids, dtype=ms.int32)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
