import ast
import argparse
import time
import numpy as np

import mindspore as ms
from mindspore import Tensor
from cambrian.transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaDecoderLayer
from transformers import AutoTokenizer


from test_modules.build_train_network import build_train_net


def _get_llama_config(pretrained_model_name_or_path):
    config, _ = LlamaConfig.from_pretrained(
        pretrained_model_name_or_path,
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
    return config


def test_llama3_attention(args):
    model_path = args.model_path
    layer_idx = args.layer_idx

    print(f"=====> test_llama3:")
    print(f"=====> Building model...")

    s_time = time.time()

    temp_data = dict(
        hidden_states=Tensor(np.random.randn(1, 2048, 4096), dtype=ms.float32),
        attention_mask=Tensor(np.full((1, 1, 2048, 2048), -65504), dtype=ms.float32),
        position_ids=Tensor(np.arange(0, 2048, 1)[None], ms.int32),
    )

    config = _get_llama_config(model_path)
    model = LlamaAttention(config, layer_idx=layer_idx)

    print(f"=====> Building model done.")

    result = model(**temp_data)

    print(f"=====> output result: {result}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")


def test_llama3_decoder_layer(args):
    model_path = args.model_path
    layer_idx = args.layer_idx

    print(f"=====> test_llama3:")
    print(f"=====> Building model...")

    s_time = time.time()

    # temp_data = dict(
    #     hidden_states=Tensor(np.random.randn(1, 2048, 4096), dtype=ms.float32),
    #     attention_mask=Tensor(np.full((1, 1, 2048, 2048), -65504), dtype=ms.float32),
    #     position_ids=Tensor(np.arange(0, 2048, 1)[None], ms.int32),
    # )
    temp_data = dict(
        # _hidden_states_in_1_pt, hidden_states_in_1, hidden_states_out_0
        hidden_states=Tensor(np.load("./hidden_states_in_2.npy"), dtype=ms.float32),
        attention_mask=Tensor(np.load("./attention_mask.npy"), dtype=ms.float32),
        position_ids=Tensor(np.load("./position_ids.npy"), ms.int32),
    )

    config = _get_llama_config(model_path)
    model = LlamaDecoderLayer(config, layer_idx=layer_idx)

    if args.fp16:
        # convert model param dtype
        from cambrian.mindspore_adapter.amp import convert_module_dtype, auto_mixed_precision
        model = convert_module_dtype(model, dtype=ms.float16)
        model = auto_mixed_precision(model, amp_level="O2", dtype=ms.float16)

    if args.checkpoint_path is not None:
        _state_dict = ms.load_checkpoint(args.checkpoint_path)
        state_dict = {k.replace(f"model.layers.{layer_idx}.", ""): v for k, v in _state_dict.items()}
        param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict)
        print(f"param_not_load: {param_not_load}")
        print(f"ckpt_not_load: {ckpt_not_load}")
        print(f"load checkpoint from `{args.checkpoint_path}` success.")

    print(f"=====> Building model done.")

    breakpoint()
    result = model(**temp_data)

    print(f"=====> output result: {result}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--checkpoint_path", type=str, default="./llama3_8b_layers_2.ckpt")
    parser.add_argument("--layer_idx", type=int, default=2)
    parser.add_argument("--ms_mode", type=int, default=1)
    parser.add_argument("--jit_level", type=str, default="O0")
    parser.add_argument("--module_name", type=str, default="decoder_layer", choices=["attention", "decoder_layer"])
    parser.add_argument("--fp16", type=ast.literal_eval, default=True)
    args, _ = parser.parse_known_args()

    if args.ms_mode == 0:
        ms.set_context(mode=ms.GRAPH_MODE,
                       device_target="Ascend",
                       jit_config={"jit_level": args.jit_level},
                       max_device_memory="59GB",
                       deterministic="ON")
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE,
                       device_target="Ascend",
                       pynative_synchronize=True,
                       max_device_memory="59GB",
                       deterministic="ON")

    if args.module_name == "attention":
        test_llama3_attention(args)
    elif args.module_name == "decoder_layer":
        test_llama3_decoder_layer(args)
    else:
        raise ValueError
