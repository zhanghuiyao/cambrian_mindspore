import argparse
import time
import numpy as np
from typing import Optional

import mindspore as ms
from mindspore import nn, ops, Tensor

from cambrian.constants import IMAGE_TOKEN_INDEX
from cambrian.mindspore_adapter.train_onestep_wrapper import TrainOneStepWrapper
from cambrian.model.language_model.cambrian_llama import CambrianLlamaModel, CambrianLlamaForCausalLM, TrainWrapperForCambrianLlamaForCausalLM



def test_cambrian_llama(model_path: str):
    pass


def test_cambrian_llama_causal(model_path: str, run_forward: bool = True, run_backward: bool = True,
                               optim: Optional[str] = None, shard_size: Optional[int] = None):

    activate_len = 120
    temp_data = dict(
        input_ids=Tensor(np.random.randint(0, 12000, size=(1, 2048)), dtype=ms.int32),
        labels=Tensor(np.ones((1, 2048)), dtype=ms.int32),
        attention_mask=Tensor(np.ones((1, 2048)), dtype=ms.bool_),
        position_ids=Tensor(np.arange(0, 2048, 1)[None], ms.int32),
        image_aux_attention_masks_list=(Tensor(np.ones((1, 576, 1)), dtype=ms.bool_),),     # ((1, 576, 1),)
        images=(Tensor(np.random.randn(1, 3, 384, 384), dtype=ms.float32),)                 # ((1, 3, 384, 384), (1, 3, 1024, 1024))
    )
    temp_data["attention_mask"][0, activate_len:] = False
    temp_data["input_ids"][0, activate_len-1] = IMAGE_TOKEN_INDEX

    model = CambrianLlamaForCausalLM.from_pretrained(
        model_path,
        mindspore_dtype=ms.float16,
    )
    model.set_train()

    if run_forward:
        print("Test cambrian-8b casual model, build forward model done.")
        print("Strat inference...")

        s_time = time.time()
        for step in range(1):
            out = model(**temp_data)
            print(f"step: {step}, forward output: {out[0]}, time cost: {time.time() - s_time:.2f}s")
            s_time = time.time()


    if run_backward:

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # create optimizer
        if optim.lower() == "zero1":
            from cambrian.mindspore_adapter.adamw_zero import AdamWeightDecayZeRO1
            optimizer = AdamWeightDecayZeRO1(model.trainable_params(), 1e-5, shard_size=shard_size)
        elif optim.lower() == "zero2":
            from cambrian.mindspore_adapter.adamw_zero import AdamWeightDecayZeRO2
            optimizer = AdamWeightDecayZeRO2(model.trainable_params(), 1e-5, shard_size=shard_size)
        else:
            optimizer = nn.AdamWeightDecay(model.trainable_params(), 1e-5)

        model = TrainWrapperForCambrianLlamaForCausalLM(model)
        train_model = TrainOneStepWrapper(model, optimizer)

        if args.amp_level == "O2":
            from cambrian.mindspore_adapter.amp import auto_mixed_precision
            train_model = auto_mixed_precision(train_model, amp_level=args.amp_level, dtype=ms.float16)

        if args.force_param_fp16:
            # FIXME: zhy_test
            from cambrian.mindspore_adapter.amp import convert_module_param_to_fp16
            train_model = convert_module_param_to_fp16(train_model, keep_norm_fp32=True)
            if hasattr(train_model, "scaler") and train_model.scaler is not None:
                train_model.scaler.scale_value.set_dtype(ms.float16)

            # FIXME: zhy_test
            print(f"zhy_test: scaler.dtype is {train_model.scaler.scale_value.dtype}")

        model.set_train()
        train_model.set_train()

        print("Test cambrian-8b casual model, build train model done.")
        print("Strat training...")

        s_time = time.time()
        for step in range(10):

            temp_data_list = ()
            for k in model.input_keys:
                v = temp_data[k]
                if isinstance(v, (list, tuple)):
                    temp_data_list += (*v,)
                else:
                    temp_data_list += (v,)

            loss, _, overflow = train_model(*temp_data_list)
            print(f"step: {step}, loss: {loss}, overflow: {overflow}, time cost: {time.time() - s_time:.2f}s")
            s_time = time.time()


def test_generate_wo_image(model_path: str):
    from transformers import AutoTokenizer

    input_ids = np.random.randint(0, 10000, size=(1, 50))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    cambrian_llama_causal = CambrianLlamaForCausalLM.from_pretrained(model_path)

    output_ids = cambrian_llama_causal.generate(input_ids)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(outputs)




if __name__ == '__main__':

    import ast

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--device_target", type=str, default="CPU")
    parser.add_argument("--max_device_memory", type=str, default="59GB")
    parser.add_argument("--is_distribute", type=ast.literal_eval, default=False)

    parser.add_argument("--amp_level", type=str, default="O2")
    parser.add_argument("--gradient_checkpointing", type=ast.literal_eval, default=True)
    parser.add_argument("--force_param_fp16", type=ast.literal_eval, default=True)

    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--shard_size", type=int, default=8)
    args, _ = parser.parse_known_args()

    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", pynative_synchronize=True)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, jit_config = {"jit_level": "O0"})
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

    # test_generate_wo_image(args.model_path)
    test_cambrian_llama_causal(
        args.model_path, optim=args.optim, shard_size=args.shard_size,
        run_forward=False, run_backward=True)
