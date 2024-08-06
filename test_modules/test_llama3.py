import argparse
import time

import mindspore as ms
from mindspore import Tensor
from cambrian.transformers.models.llama import LlamaModel, LlamaForCausalLM
from transformers import AutoTokenizer


from test_modules.build_train_network import build_train_net


def test_llama3(model_path: str):

    print(f"=====> test_llama3:")
    print(f"=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaModel.from_pretrained(model_path)

    print(f"=====> Building model done.")

    prompt = ["hello world.",]
    input_ids = Tensor(tokenizer(prompt).input_ids, ms.int32)

    result = model(input_ids)

    print(f"=====> input prompt: {prompt}")
    print(f"=====> output result: {result}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")


def test_llama3_causal(model_path: str):

    print(f"=====> test_llama3_causal:")
    print(f"=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    print(f"=====> Building model done.")

    prompt = ["hello world.",]
    input_ids = Tensor(tokenizer(prompt).input_ids, ms.int32)

    result = model(input_ids)

    print(f"=====> input prompt: {prompt}")
    print(f"=====> output result: {result}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")


def test_llama3_generate(model_path: str):

    print(f"=====> test_llama3_generate:")
    print(f"=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    print(f"=====> Building model done.")

    prompt = ["hello world.",]
    input_ids = Tensor(tokenizer(prompt).input_ids, ms.int32)

    model_input = model.prepare_inputs_for_generation(input_ids)
    result = model.generate(
        **model_input,
        max_new_tokens=20
    )

    print(f"=====> input prompt: {prompt}")
    print(f"=====> output result: {result}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")


def test_llama3_causal_bp(model_path: str):

    print(f"=====> test_llama3_causal:")
    print(f"=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    print(f"=====> Building model done.")

    prompt = ["hello world.",]
    input_ids = Tensor(tokenizer(prompt).input_ids, ms.int32)

    train_model = build_train_net(model, out_feature_index=1)
    result = train_model(input_ids)
    loss, _, overflow = result

    print(f"=====> input prompt, {prompt}")
    print(f"=====> output result, loss: {loss}, overflow: {overflow}, time cost: {time.time() - s_time:.2f}s")
    print(f"=====> Done.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    args, _ = parser.parse_known_args()

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})
    # ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU", pynative_synchronize=True)

    # test_llama3(args.model_path)
    test_llama3_causal(args.model_path)
    # test_llama3_generate(args.model_path)
    # test_llama3_causal_bp(args.model_path)
