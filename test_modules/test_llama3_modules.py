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


def test_llama3_attention(model_path: str):

    print(f"=====> test_llama3:")
    print(f"=====> Building model...")

    s_time = time.time()

    temp_data = dict(
        hidden_states=Tensor(np.random.randn(1, 2048, 4096), dtype=ms.float32),
        attention_mask=Tensor(np.full((1, 1, 2048, 2048), -65504), dtype=ms.float32),
        position_ids=Tensor(np.arange(0, 2048, 1)[None], ms.int32),
    )

    config = _get_llama_config(model_path)
    model = LlamaAttention(config, layer_idx=0)

    print(f"=====> Building model done.")

    result = model(**temp_data)

    print(f"=====> output result: {result}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")


def test_llama3_decoder_layer(model_path: str):

    print(f"=====> test_llama3:")
    print(f"=====> Building model...")

    s_time = time.time()

    temp_data = dict(
        hidden_states=Tensor(np.random.randn(1, 2048, 4096), dtype=ms.float32),
        attention_mask=Tensor(np.full((1, 1, 2048, 2048), -65504), dtype=ms.float32),
        position_ids=Tensor(np.arange(0, 2048, 1)[None], ms.int32),
    )

    config = _get_llama_config(model_path)
    model = LlamaDecoderLayer(config, layer_idx=0)

    print(f"=====> Building model done.")

    result = model(**temp_data)

    print(f"=====> output result: {result}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--module_name", type=str, default="decoder_layer", choices=["attention", "decoder_layer"])
    args, _ = parser.parse_known_args()

    # ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU", pynative_synchronize=True)

    if args.module_name == "attention":
        test_llama3_attention(args.model_path)
    elif args.module_name == "decoder_layer":
        test_llama3_decoder_layer(args.model_path)
    else:
        raise ValueError
