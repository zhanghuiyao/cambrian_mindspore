import argparse
import os
import glob
import sys
import time

import mindspore as ms
from mindspore import Tensor, nn

sys.path.append("../weight_conversion/")

from load_pt_weight import load_from_folder


def name_replace_cambrian_8b(weight_name: str):
    """replace weight name"""

    # FIXME: modify when resize embedding
    # embedding token
    weight_name = weight_name.replace("model.embed_tokens.weight", "embed_tokens.embedding_table")

    # mm projector
    weight_name = weight_name.replace('model.mm_projector_aux_', 'model.mm_projector_auxes.')
    weight_name = weight_name.replace('model.mm_projector_aux_', 'model.mm_projector_auxes.')
    if weight_name.startswith("model.mm_projector_auxes."):
        weight_name = weight_name.replace(".3.weight", ".3.gamma")
        weight_name = weight_name.replace(".3.bias", ".3.beta")

    # vision sampler
    if weight_name.startswith("model.vision_sampler_layers."):
        pass
    else:
        weight_name = weight_name.replace('model.vision_sampler_', 'model.vision_samplers.')

    weight_name = weight_name.replace('cross_attn.k_proj_', 'cross_attn.k_projs.')
    weight_name = weight_name.replace('cross_attn.v_proj_', 'cross_attn.v_projs.')
    if weight_name.startswith("model.vision_samplers.") or weight_name.startswith("model.vision_sampler_layers."):
        if "cross_attn.k_projs." in weight_name or "cross_attn.v_projs." in weight_name or "cross_attn.q_proj.":
            weight_name = weight_name.replace(".0.weight", ".0.gamma")
            weight_name = weight_name.replace(".0.bias", ".0.beta")

        if "pos_embed_" in weight_name:
            weight_name = weight_name.replace('pos_embed_', 'pos_embeds.') + ".parameter_attr"

    # skip rmsnorm
    if weight_name.endswith("layernorm.weight"):
        return weight_name
    elif weight_name.endswith("model.norm.weight"):
        return weight_name

    # other norm layers
    weight_name = weight_name.replace('norm.weight', 'norm.gamma')
    weight_name = weight_name.replace('norm.bias', 'norm.beta')

    return weight_name


def name_replace_siglip(weight_name: str):
    """replace weight name"""

    # only load vision model
    if not weight_name.startswith("visual.trunk."):
        return None

    # prefix name
    weight_name = weight_name.replace("visual.trunk.", "model.vision_tower_aux_list.0.vision_tower.")

    # norm layers
    if "norm" in weight_name:
        weight_name = weight_name.replace(".weight", ".gamma")
        weight_name = weight_name.replace(".bias", ".beta")

    return weight_name


def name_replace_openai_clip(weight_name: str):
    """replace weight name"""

    # only load vision model
    if not weight_name.startswith("vision_model."):
        return None

    # prefix name
    weight_name = weight_name.replace("vision_model.", "model.vision_tower_aux_list.1.vision_tower.vision_model.")

    # embedding
    weight_name = weight_name.replace("position_embedding.weight", "position_embedding.embedding_table")

    # norm layers
    if "norm" in weight_name:
        weight_name = weight_name.replace(".weight", ".gamma")
        weight_name = weight_name.replace(".bias", ".beta")

    return weight_name


def name_replace_dinov2(weight_name: str):
    """replace weight name"""

    # prefix name
    weight_name = weight_name.replace("embeddings.", "model.vision_tower_aux_list.2.vision_tower.embeddings.", 1)  # just replace once
    weight_name = weight_name.replace("encoder.", "model.vision_tower_aux_list.2.vision_tower.encoder.")
    weight_name = weight_name.replace("layernorm.", "model.vision_tower_aux_list.2.vision_tower.layernorm.")

    # norm layers
    if "norm" in weight_name:
        weight_name = weight_name.replace(".weight", ".gamma")
        weight_name = weight_name.replace(".bias", ".beta")

    return weight_name


def name_replace_openclip_convnext(weight_name: str):
    """replace weight name"""

    # only load vision model
    if not weight_name.startswith("visual.trunk."):
        return None

    # prefix name
    weight_name = weight_name.replace("visual.trunk.", "model.vision_tower_aux_list.3.vision_tower.")

    # special norm layers
    weight_name = weight_name.replace("stages.1.downsample.0.weight", "stages.1.downsample.0.gamma")
    weight_name = weight_name.replace("stages.1.downsample.0.bias", "stages.1.downsample.0.beta")
    weight_name = weight_name.replace("stages.2.downsample.0.weight", "stages.2.downsample.0.gamma")
    weight_name = weight_name.replace("stages.2.downsample.0.bias", "stages.2.downsample.0.beta")
    weight_name = weight_name.replace("stages.3.downsample.0.weight", "stages.3.downsample.0.gamma")
    weight_name = weight_name.replace("stages.3.downsample.0.bias", "stages.3.downsample.0.beta")
    weight_name = weight_name.replace("stem.1.weight", "stem.1.gamma")
    weight_name = weight_name.replace("stem.1.bias", "stem.1.beta")

    # other norm layers
    weight_name = weight_name.replace("norm.weight", "norm.gamma")
    weight_name = weight_name.replace("norm.bias", "norm.beta")

    return weight_name


replace_func_map = {
    "cambrian-8b": name_replace_cambrian_8b,
    "siglip": name_replace_siglip,
    "openai": name_replace_openai_clip,
    "dinov2": name_replace_dinov2,
    "convnext": name_replace_openclip_convnext,
}


def path_parse(args):

    if args.full_folder is not None:
        assert os.path.isdir(args.full_folder)
        subfolders = glob.glob(args.full_folder + "/*/")

        for name in replace_func_map:

            if "cambrian" in name:
                folder_name = f"cambrian_folder"
                if name != args.model_name:
                    continue
            else:
                folder_name = f"{name}_folder"

            if getattr(args, folder_name) is None or not os.path.isdir(getattr(args, folder_name)):

                for sub_folder in subfolders:
                    if name in sub_folder.lower():
                        setattr(args, folder_name, sub_folder)
                        break

    assert os.path.isdir(args.cambrian_folder)
    assert os.path.isdir(args.siglip_folder)
    assert os.path.isdir(args.openai_folder)
    assert os.path.isdir(args.dinov2_folder)
    assert os.path.isdir(args.convnext_folder)

    return args


def pt_to_ms(args):
    ms_param_list = []
    num = 0
    for name in replace_func_map:

        if "cambrian" in name:
            folder_name = f"cambrian_folder"
            if name != args.model_name:
                continue
        else:
            folder_name = f"{name}_folder"

        replace_func = replace_func_map[name]
        folder = getattr(args, folder_name)
        state_dict = load_from_folder(folder)

        for k in state_dict:
            new_k = replace_func(k)
            if new_k is not None:
                ms_param_list.append({'name': new_k, 'data': Tensor(state_dict[k].numpy())})

        num += 1

        print(f"=======> {num}/5, convert {name} form {folder} done.")

    print(f"=======> saving mindspore checkpoint...")
    s_time = time.time()
    ckpt_path = args.mindspore_checkpoint_path \
        if args.mindspore_checkpoint_path is not None else f"{args.model_name}.ckpt"
    ms.save_checkpoint(ms_param_list, ckpt_path)
    print(f"=======> save mindspore checkpoint to {ckpt_path} success, time cost: {time.time()-s_time:.2f}s")


def convert_weight(args):

    args = path_parse(args)

    if args.task == "pt2ms":
        pt_to_ms(args)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert weight")

    # model name
    parser.add_argument("--model_name", type=str, default="cambrian-8b")
    parser.add_argument("--task", type=str, default="pt2ms")

    # pt checkpoint path
    parser.add_argument("--full_folder", type=str, default=None)
    parser.add_argument("--cambrian_folder", type=str, default=None, help="download from `https://huggingface.co/nyu-visionx/cambrian-8b`")
    parser.add_argument("--siglip_folder", type=str, default=None, help="download from `https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384`")
    parser.add_argument("--openai_folder", type=str, default=None, help="download from `https://huggingface.co/openai/clip-vit-large-patch14-336`")
    parser.add_argument("--dinov2_folder", type=str, default=None, help="download from `https://huggingface.co/facebook/dinov2-giant`")
    parser.add_argument("--convnext_folder", type=str, default=None, help="download from `https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup`")

    # ms checkpoint path
    parser.add_argument("--mindspore_checkpoint_path", type=str, default=None)

    args, _ = parser.parse_known_args()



    convert_weight(args)
