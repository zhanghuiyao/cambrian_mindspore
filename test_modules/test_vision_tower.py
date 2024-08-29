import ast
import argparse
import sys
import time
import os
from PIL import Image

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.communication.management import init as init_ms

from cambrian.model.language_model.cambrian_llama import CambrianConfig
from cambrian.model.multimodal_encoder.builder import build_vision_tower_aux_list
from cambrian.mm_utils import process_images

from test_modules.build_train_network import build_train_net


def test_vision_tower(args):
    config, _ = CambrianConfig.from_pretrained(
        args.model_path,
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

    full_mm_vision_tower_aux_list = [
        "siglip/CLIP-ViT-SO400M-14-384",        # https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384
        "openai/clip-vit-large-patch14-336",    # https://huggingface.co/openai/clip-vit-large-patch14-336
        "facebook/dinov2-giant-res378",         # https://huggingface.co/facebook/dinov2-giant
        "clip-convnext-XXL-multi-stage"         # https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup
    ]
    full_mm_vision_tower_aux_token_len_list = [
        576,
        576,
        576,
        9216
    ]
    full_default_checkpoint_paths = [
        "./vision_0_siglip.ckpt",
        "./vision_1_openai_clip.ckpt",
        "./vision_2_dinov2.ckpt",
        "./vision_3_convnext.ckpt",
    ]

    vision_tower_index = [int(i) for i in args.vision_tower_index.split(",")]
    mm_vision_tower_aux_list = [full_mm_vision_tower_aux_list[i] for i in vision_tower_index]
    mm_vision_tower_aux_token_len_list = [full_mm_vision_tower_aux_token_len_list[i] for i in vision_tower_index]

    if args.checkpoint_path is None:
        checkpoint_paths = [full_default_checkpoint_paths[i] for i in vision_tower_index]
    else:
        checkpoint_paths = [_ckpt for _ckpt in args.checkpoint_path.split(",")]

    module_len = len(mm_vision_tower_aux_list)
    for i, (module_name, token_len, checkpoint_path, vision_index) in enumerate(
            zip(mm_vision_tower_aux_list, mm_vision_tower_aux_token_len_list, checkpoint_paths, vision_tower_index)):

        config.mm_vision_tower_aux_list = [module_name, ]
        config.mm_vision_tower_aux_token_len_list = [token_len, ]
        model = build_vision_tower_aux_list(config)[0]
        print(f"======> {i + 1}/{module_len}, build model `{module_name}` done")

        dtype = ms.float32
        if args.fp16:
            # convert model param dtype
            from cambrian.mindspore_adapter.amp import convert_module_dtype, auto_mixed_precision
            model = convert_module_dtype(model, dtype=ms.float16)
            model = auto_mixed_precision(model, amp_level="O2", dtype=ms.float16)

            dtype = ms.float16

        if checkpoint_path.lower() != "none":
            _state_dict = ms.load_checkpoint(checkpoint_path)
            state_dict = {k.replace(f"model.vision_tower_aux_list.{vision_index}.", ""): v for k, v in _state_dict.items()}

            param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict)
            print(f"======> {i + 1}/{module_len}, "
                  f"load checkpoint from {checkpoint_path}, "
                  f"param_not_load: {param_not_load}, "
                  f"ckpt_not_load: {ckpt_not_load}")
        else:
            print(f"======> {i + 1}/{module_len}, no available checkpoint.")

        train_net = build_train_net(model) if args.run_backward else None

        image = Image.open(args.image_path).convert('RGB')
        image_processor = [model.image_processor, ]
        image = process_images([image], image_processor, config)
        # (num, bs, 3, 384, 384) -> (1, 3, 384, 384)
        image_tensor = Tensor(image[0], dtype)

        if args.run_forward:
            s_time = time.time()
            print(f"======> {i + 1}/{module_len}, Run Forward...")

            breakpoint()

            out = model(image_tensor)

            print(f"======> {i + 1}/{module_len}, Image Process: image_tensor.shape: {image_tensor.shape}")
            print(f"======> {i + 1}/{module_len}, Result: out.shape: {out.shape}, time cost: {time.time() - s_time:.2f}s")

        if args.run_backward:
            s_time = time.time()
            print(f"======> {i + 1}/{module_len}, Run Backward...")

            model.set_train()
            train_net.set_train()

            out = train_net(image_tensor)
            loss, _, overflow = out

            print(f"======> {i + 1}/{module_len}, Image Process: image_tensor.shape: {image_tensor.shape}")
            print(f"======> {i + 1}/{module_len}, Result: loss: {loss.item():.4f}, overflow: {overflow}, time cost: {time.time() - s_time:.2f}s")

        print(f"======> {i + 1}/{module_len}, model name: {module_name}, token_len: {token_len}, run done.")
        print(f"=" * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=1, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--jit_level", type=str, default="O0")
    parser.add_argument("--image_path", type=str, default="./demo/math.png")
    parser.add_argument("--fp16", type=ast.literal_eval, default=True)

    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--vision_tower_index", type=str, default="0,1,2,3")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="None,None,None,None")

    parser.add_argument("--run_forward", type=ast.literal_eval, default=True)
    parser.add_argument("--run_backward", type=ast.literal_eval, default=False)
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

    else:
        ms.set_context(
            mode=ms.PYNATIVE_MODE,
            device_target="Ascend",
            pynative_synchronize=True,
            max_device_memory="59GB",
            deterministic="ON"
        )

    test_vision_tower(args)
