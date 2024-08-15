import argparse
import time
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor
from cambrian.mindspore_adapter import TrainOneStepWrapper
from cambrian.transformers.models.llama import LlamaModel, LlamaForCausalLM

from transformers import AutoTokenizer

from test_modules.build_train_network import build_train_net, NetWithLoss


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

    kwargs = {}
    if args.enable_fa:
        kwargs.update({"attn_implementation": "flash_attention_2"})
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        **kwargs
    )

    if args.run_forward:
        print("Test llama3-8b casual model, build forward model done.")
        print("Strat inference...")

        model.set_train(False)

        prompt = ["hello world.", ]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = Tensor(tokenizer(prompt).input_ids, ms.int32)

        s_time = time.time()
        for step in range(args.run_steps):
            result = model(input_ids)

            print(f"step: {step}, forward output: {result}, time cost: {time.time() - s_time:.2f}s")
            s_time = time.time()


    if args.run_backward:
        input_ids = Tensor(np.random.randint(0, 12000, size=(1, 2048)), dtype=ms.int32)

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # FIXME: zhy_test
        # 1. force param fp16
        if args.force_param_fp16:
            from cambrian.mindspore_adapter.amp import convert_module_param_to_fp16
            model = convert_module_param_to_fp16(model, keep_norm_fp32=True)

        # create optimizer
        if args.optim.lower() == "zero1":
            from cambrian.mindspore_adapter.adamw_zero import AdamWeightDecayZeRO1
            optimizer = AdamWeightDecayZeRO1(model.trainable_params(), 1e-5, shard_size=args.shard_size)
        elif args.optim.lower() == "zero2":
            from cambrian.mindspore_adapter.adamw_zero import AdamWeightDecayZeRO2
            optimizer = AdamWeightDecayZeRO2(model.trainable_params(), 1e-5, shard_size=args.shard_size)
        elif args.optim.lower() == "adamw":
            from cambrian.mindspore_adapter.adamw import AdamWeightDecay
            # optimizer = nn.AdamWeightDecay(model.trainable_params(), 1e-5)
            optimizer = AdamWeightDecay(model.trainable_params(), 1e-5)
        elif args.optim.lower() == "sgd":
            optimizer = nn.SGD(model.trainable_params(), 1e-5)
        else:
            raise NotImplementedError

        model = NetWithLoss(model, out_feature_index=1)
        train_model = TrainOneStepWrapper(
            model,
            optimizer,
            clip_grad="global_norm",
            clip_value=1.0
        )

        if args.amp_level == "O2":
            from cambrian.mindspore_adapter.amp import auto_mixed_precision
            train_model = auto_mixed_precision(train_model, amp_level=args.amp_level, dtype=ms.float16)

        model.set_train()
        train_model.set_train()

        print("Test llama3-8b casual model, build train model done.")
        print("Strat training...")

        s_time = time.time()
        for step in range(args.run_steps):
            loss, _, overflow = train_model(input_ids)
            print(f"step: {step}, loss: {loss}, overflow: {overflow}, time cost: {time.time() - s_time:.2f}s")
            s_time = time.time()


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
    parser.add_argument("--device_target", type=str, default="Ascend")
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
    parser.add_argument("--run_steps", type=int, default=1)

    parser.add_argument("--enable_tracker", type=ast.literal_eval, default=False)
    args, _ = parser.parse_known_args()


    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", pynative_synchronize=True)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, jit_config={"jit_level": "O0"})
    ms.set_context(max_device_memory=args.max_device_memory)
    if args.enable_tracker:
        ms.set_context(memory_optimize_level="O0", pynative_synchronize=True)

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
    # test_llama3_generate(args.model_path)
    # test_llama3_causal_bp(args.model_path)
    test_llama3_causal(args.model_path, args)
