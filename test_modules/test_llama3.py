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


def test_llama3_causal(model_path: str, args):

    print(f"=====> test_llama3_causal:")
    print(f"=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    kwargs = {}
    if args.enable_fa:
        kwargs.update({"attn_implementation": "flash_attention_2"})
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        **kwargs
    )
    model.set_train()

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
    import ast

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--device_target", type=str, default="CPU")
    parser.add_argument("--max_device_memory", type=str, default="59GB")
    parser.add_argument("--is_distribute", type=ast.literal_eval, default=False)

    parser.add_argument("--amp_level", type=str, default="O2")
    parser.add_argument("--enable_fa", type=ast.literal_eval, default=True)
    parser.add_argument("--gradient_checkpointing", type=ast.literal_eval, default=True)
    parser.add_argument("--force_param_fp16", type=ast.literal_eval, default=True)

    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--shard_size", type=int, default=8)

    parser.add_argument("--run_forward", type=ast.literal_eval, default=False)
    parser.add_argument("--run_backward", type=ast.literal_eval, default=True)
    args, _ = parser.parse_known_args()


    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", pynative_synchronize=True)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, jit_config={"jit_level": "O0"})
    ms.set_context(max_device_memory=args.max_device_memory)


    if args.is_distribute:
        from mindspore.communication.management import init, get_rank, get_group_size
        init()
        rank_id, world_size = get_rank(), get_group_size()
        print(f"init_environment, rank_id: {rank_id}, world_size: {world_size}")

        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=world_size,
        )


    # test_llama3(args.model_path)
    test_llama3_causal(args.model_path, args)
    # test_llama3_generate(args.model_path)
    # test_llama3_causal_bp(args.model_path)
