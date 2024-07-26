import argparse
import os
import glob

import mindspore as ms
from mindspore import Tensor, nn

from .load_pt_weight import load_from_folder


def name_replace_cambrian_8b(weight_name: str):
    """replace weight name"""

    # prefix

    # vision sampler
    weight_name = weight_name.replace('model.vision_sampler_', 'model.vision_sampler_layers.')

    # norm
    weight_name = weight_name.replace('norm.weight', 'norm.gamma')
    weight_name = weight_name.replace('norm.bias', 'norm.beta')
    # k_proj

    return weight_name


replace_func_map = {
    "cambrian-8b": name_replace_cambrian_8b,
    "siglip": None,
    "openai_clip": None,
    "dinov2": None,
    "openclip_convnext": None,
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

            if not os.path.isdir(getattr(args, folder_name)):

                for sub_folder in subfolders:
                    if name in sub_folder:
                        setattr(args, folder_name, sub_folder)
                        break

    assert os.path.isdir(args.cambrian_folder)
    assert os.path.isdir(args.siglip_folder)
    assert os.path.isdir(args.openai_clip_folder)
    assert os.path.isdir(args.dinov2_folder)
    assert os.path.isdir(args.openclip_convnext_folder)

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
            ms_param_list.append({'name': new_k, 'data': Tensor(state_dict[k].numpy())})

        num += 1

        print(f"{num}/5, convert {name} form {folder} done.")

    print(f"saving mindspore checkpoint...")
    ms.save_checkpoint(ms_param_list, args.mindspore_ckpt_path)
    print(f"save mindspore checkpoint...")


def convert_weight(args):

    args = path_parse(args)

    if args.task == "pt2ms":
        pt_to_ms(args)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert weight")

    parser.add_argument("--model_name", type=str, default="cambrian-8b")
    parser.add_argument("--task", type=str, default="pt2ms")

    # pt checkpoint path
    parser.add_argument("--full_folder", type=str, default=None)
    parser.add_argument("--cambrian_folder", type=str, default=None)
    parser.add_argument("--siglip_folder", type=str, default=None)
    parser.add_argument("--openai_clip_folder", type=str, default=None)
    parser.add_argument("--dinov2_folder", type=str, default=None)
    parser.add_argument("--openclip_convnext_folder", type=str, default=None)

    # ms checkpoint path
    parser.add_argument("--ms_checkpoint_path", type=str, default="cambrian-8b.ckpt")

    args, _ = parser.parse_known_args()

    convert_weight(args)
