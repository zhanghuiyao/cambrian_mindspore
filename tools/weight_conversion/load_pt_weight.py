import os
import json
from functools import partial
from packaging import version

import torch
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file


parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")


WEIGHTS_NAME_TRANSFORMERS = "pytorch_model.bin"
SAFE_WEIGHTS_NAME_TRANSFORMERS = "model.safetensors"
WEIGHTS_INDEX_NAME_TRANSFORMERS = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_INDEX_NAME_TRANSFORMERS = "model.safetensors.index.json"
WEIGHTS_NAME_OPENCLIP = "open_clip_pytorch_model.bin"
SAFE_WEIGHTS_NAME_OPENCLIP = "open_clip_model.safetensors"


NAME_DICT_TRANSFORMERS = dict(
    WEIGHTS_NAME="pytorch_model.bin",
    SAFE_WEIGHTS_NAME="model.safetensors",
    WEIGHTS_INDEX_NAME="pytorch_model.bin.index.json",
    SAFE_WEIGHTS_INDEX_NAME="model.safetensors.index.json"
)

NAME_DICT_OPENCLIP = dict(
    WEIGHTS_NAME="open_clip_pytorch_model.bin",
    SAFE_WEIGHTS_NAME="open_clip_model.safetensors",
)


def load_from_folder(folder, prefer_safe=True):
    """
    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """

    weights_file_transformers = os.path.join(folder, NAME_DICT_TRANSFORMERS["WEIGHTS_NAME"])
    safe_weights_file_transformers = os.path.join(folder, NAME_DICT_TRANSFORMERS["SAFE_WEIGHTS_NAME"])
    index_file_transformers = os.path.join(folder, NAME_DICT_TRANSFORMERS["WEIGHTS_INDEX_NAME"])
    safe_index_file_transformers = os.path.join(folder, NAME_DICT_TRANSFORMERS["SAFE_WEIGHTS_INDEX_NAME"])
    weights_file_openclip = os.path.join(folder, NAME_DICT_OPENCLIP["WEIGHTS_NAME"])
    safe_weights_file_openclip = os.path.join(folder, NAME_DICT_OPENCLIP["SAFE_WEIGHTS_NAME"])

    is_transformers_folder = False
    for file in (weights_file_transformers, safe_weights_file_transformers, index_file_transformers, safe_index_file_transformers):
        if os.path.isfile(file):
            is_transformers_folder = True
            break

    is_openclip_folder = False
    for file in (weights_file_openclip, safe_weights_file_openclip):
        if os.path.isfile(file):
            is_openclip_folder = True
            break

    assert is_transformers_folder ^ is_openclip_folder

    if is_transformers_folder and not is_openclip_folder:
        name_dict = NAME_DICT_TRANSFORMERS
    elif is_openclip_folder and not is_transformers_folder:
        name_dict = NAME_DICT_OPENCLIP
    else:
        raise ValueError

    WEIGHTS_NAME = name_dict.pop("WEIGHTS_NAME")
    SAFE_WEIGHTS_NAME = name_dict.pop("SAFE_WEIGHTS_NAME")
    WEIGHTS_INDEX_NAME = name_dict.pop("WEIGHTS_INDEX_NAME", None)
    SAFE_WEIGHTS_INDEX_NAME = name_dict.pop("SAFE_WEIGHTS_INDEX_NAME", None)

    weights_file = os.path.join(folder, WEIGHTS_NAME)
    safe_weights_file = os.path.join(folder, SAFE_WEIGHTS_NAME)

    if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
        safe_present = os.path.isfile(safe_weights_file)
        bin_present = os.path.isfile(weights_file)

        load_safe = False
        if safe_present:
            if prefer_safe:
                load_safe = True  # load safe due to preference
            elif not bin_present:
                load_safe = True  # load safe since we have no other choice

        weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
        loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu", **weights_only_kwarg)

        weight_file = safe_weights_file if load_safe else weights_file

        state_dict = loader(os.path.join(folder, weight_file))

        return state_dict

    else:

        # Load the index
        index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
        safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

        index_present = os.path.isfile(index_file)
        safe_index_present = os.path.isfile(safe_index_file)

        if not index_present and not safe_index_present:
            filenames = (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME)
            raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

        load_safe = False
        if safe_index_present:
            if prefer_safe:
                load_safe = True  # load safe due to preference
            elif not index_present:
                load_safe = True  # load safe since we have no other choice

        load_index = safe_index_file if load_safe else index_file

        with open(load_index, "r", encoding="utf-8") as f:
            index = json.load(f)

        shard_files = list(set(index["weight_map"].values()))

        weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
        loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu", **weights_only_kwarg)

        full_state_dict = {}
        for shard_file in shard_files:
            state_dict = loader(os.path.join(folder, shard_file))

            full_state_dict.update(state_dict)

            # Make sure memory is freed before we load the next state dict.
            del state_dict

        return full_state_dict
