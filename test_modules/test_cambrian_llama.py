import argparse
import time
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor

from cambrian.constants import IMAGE_TOKEN_INDEX
from cambrian.mindspore_adapter.train_onestep_wrapper import TrainOneStepWrapper
from cambrian.model.language_model.cambrian_llama import CambrianLlamaModel, CambrianLlamaForCausalLM



def test_cambrian_llama(model_path: str):
    pass


def test_cambrian_llama_causal(model_path: str, training=True):

    activate_len = 120
    temp_data = dict(
        input_ids=Tensor(np.random.randint(0, 12000, size=(1, 2048)), dtype=ms.int32),
        labels=Tensor(np.ones((1, 2048)), dtype=ms.int32),
        attention_mask=Tensor(np.ones((1, 2048)), dtype=ms.bool_),
        position_ids=Tensor(np.arange(0, 2048, 1)[None], ms.int32),
        image_aux_attention_masks_list=Tensor(np.ones((1, 1, 576, 1)), dtype=ms.bool_),
        images=Tensor(np.random.randn(1, 1, 3, 384, 384), dtype=ms.float32)
    )
    temp_data["attention_mask"][0, activate_len:] = False
    temp_data["input_ids"][0, activate_len-1] = IMAGE_TOKEN_INDEX

    model = CambrianLlamaForCausalLM.from_pretrained(
        model_path,
        mindspore_dtype=None,
        # cache_dir=training_args.cache_dir,
        # do_sample=True,
    )
    optimizer = nn.AdamWeightDecay(model.trainable_params())
    train_model = TrainOneStepWrapper(model, optimizer)

    if training:
        model.set_train(True)
        train_model.set_train(True)

    s_time = time.time()
    print("Test cambrian-8b pretrain, build train model done.")
    print("Strat training...")

    for step in range(10):
        out = model(**temp_data)
        print(f"step: {step}, forward output: {out[0]}, time cost: {time.time() - s_time:.2f}s")
        s_time = time.time()

        # if training:
        #     loss, _, overflow = train_model(**temp_data)
        #     print(f"step: {step}, loss: {loss}, overflow: {overflow}, time cost: {time.time() - s_time:.2f}s")
        #     s_time = time.time()


def test_generate_wo_image(model_path: str):
    from transformers import AutoTokenizer

    input_ids = np.random.randint(0, 10000, size=(1, 50))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    cambrian_llama_causal = CambrianLlamaForCausalLM.from_pretrained(model_path)

    output_ids = cambrian_llama_causal.generate(input_ids)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(outputs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    args, _ = parser.parse_known_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", pynative_synchronize=True)
    # ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config = {"jit_level": "O0"})

    # test_generate_wo_image(args.model_path)
    test_cambrian_llama_causal(args.model_path, training=True)
