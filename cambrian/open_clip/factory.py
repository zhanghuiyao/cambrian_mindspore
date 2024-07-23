# Reference to https://github.com/mlfoundations/open_clip

import os
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mindspore as ms

from cambrian.open_clip.model import CLIP, CustomTextCLIP
from cambrian.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from cambrian.open_clip.transform import PreprocessCfg, merge_preprocess_kwargs, image_transform_v2, merge_preprocess_dict

from .model import set_model_preprocess_cfg


HF_HUB_PREFIX = "hf-hub:"
_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            # if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
            if all(a in model_cfg for a in ("embed_dim", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def load_checkpoint(network, weight):
    if weight.endswith(".ckpt"):
        param_dict = ms.load_checkpoint(weight)
        ms.load_param_into_net(network, param_dict)
        logging.info(f'Checkpoint load from "{weight}" success.')
    else:
        raise ValueError("Not support weight format.")


def create_model(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    force_preprocess_cfg: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    pretrained_cfg = {}

    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT names
        raise NotImplementedError
    else:
        load_from_local = os.path.isfile(os.path.join(model_name, "open_clip_config.json"))
        if load_from_local:
            config_path = os.path.join(model_name, "open_clip_config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
            model_cfg = config['model_cfg']
        else:
            model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT names
            model_cfg = get_model_config(model_name)

    if model_cfg is not None:
        logging.info(f"Loaded {model_name} model config.")
    else:
        logging.error(f"Model config for {model_name} not found; available models {list_models()}.")
        raise RuntimeError(f"Model config for {model_name} not found.")

    if pretrained and pretrained.lower() == 'openai':
        raise NotImplementedError

    if force_patch_dropout is not None:
        # override the default patch dropout value
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

    if force_image_size is not None:
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = ms.float32 if precision == "fp32" else ms.float16
    custom_text = model_cfg.pop("custom_text", False) or force_custom_text

    if custom_text:
        if "multimodal_cfg" in model_cfg:
            raise NotImplementedError
        else:
            model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
    else:
        model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    if precision in ("fp16", "bf16", "pure_fp16", "pure_bf16"):
        # manual mixed precision that matches original OpenAI behaviour
        model.to_float(ms.float16)

    if pretrained:
        assert pretrained.endswith(".ckpt"), f"pretrained expect '*.ckpt', but got '{pretrained}'."
        load_checkpoint(model, pretrained)

    if model.visual is not None:
        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get("mean", None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get("std", None) or OPENAI_DATASET_STD

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess


if __name__ == "__main__":
    import argparse
    import ast

    parser_config = argparse.ArgumentParser(description="Config", add_help=False)
    parser_config.add_argument("--ms_jit", type=ast.literal_eval, default=False)
    args, _ = parser_config.parse_known_args()

    model = create_model(model_name="ViT-H-14-Text", pretrained="")  # "laion2b_s32b_b79k"

    @ms.jit
    def jit_warpper(token):
        return model.token_embedding(token)

    token = tokenize(["a photo of a cat", "a photo of a dog"])
    if not args.ms_jit:
        out = model.token_embedding(token)
    else:
        out = jit_warpper(token)

    print(f"token.shape: {token.shape}")
    print(f"out.shape: {out.shape}")
