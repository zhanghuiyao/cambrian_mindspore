import os
import ast
import argparse
import time
import mindspore as ms
import numpy as np

from transformers import AutoTokenizer

from cambrian.transformers.models.llama import LlamaForCausalLM
from cambrian.mindspore_adapter import auto_mixed_precision


def preprocess_input_before_generate_numpy_2(
        max_len: int,
        input_ids: np.ndarray,
        labels: np.ndarray = None,
        position_ids: np.ndarray = None,
        attention_mask: np.ndarray = None,
):
    # init empty array
    bs = len(input_ids)
    padded_input_ids = np.zeros((bs, max_len), np.int32)
    padded_labels = np.full((bs, max_len), -1, np.int32)
    padded_position_ids = np.zeros((bs, max_len), np.int32)
    padded_attention_mask = np.zeros((bs, max_len), np.bool_)

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = np.ones_like(input_ids, dtype=np.bool_)
    else:
        attention_mask = attention_mask.astype(np.bool_)
    if position_ids is None:
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int32)
    if labels is None:
        labels = np.full_like(input_ids, -1)

    masked_input_ids = []
    masked_labels = []
    masked_attention_mask = []
    for i in range(len(input_ids)):
        cur_input_ids, cur_labels, cur_attention_mask = input_ids[i], labels[i], attention_mask[i]
        active_len = int(cur_attention_mask.sum())
        # assert cur_attention_mask[:active_len].sum() == cur_attention_mask.sum()
        masked_input_ids.append(cur_input_ids[:active_len])
        masked_labels.append(cur_labels[:active_len])
        masked_attention_mask.append(cur_attention_mask[:active_len])
    input_ids = masked_input_ids
    labels = masked_labels
    attention_mask = masked_attention_mask

    for batch_idx, cur_input_ids in enumerate(input_ids):

        cur_len = cur_input_ids.shape[0]

        padded_input_ids[batch_idx, :cur_len] = cur_input_ids[:]

        padded_labels[batch_idx, :cur_len] = labels[batch_idx][:]
        padded_attention_mask[batch_idx, :cur_len] = attention_mask[batch_idx][:]
        padded_position_ids[batch_idx, :cur_len] = np.arange(0, cur_len, dtype=position_ids.dtype)

    new_input_ids = ms.Tensor(padded_input_ids)
    new_attention_mask = ms.Tensor(padded_attention_mask)
    new_labels = None if _labels is None else ms.Tensor(padded_labels)
    new_position_ids = None if _position_ids is None else ms.Tensor(padded_position_ids)

    return new_input_ids, new_labels, new_position_ids, new_attention_mask


def run_llama3_generate(args):

    print(f"=====> test_llama3_generate:")
    print(f"=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(args.model_path, use_flash_attention_2=args.use_fa)
    model.embed_tokens = model.model.embed_tokens  # zhy_test

    model = auto_mixed_precision(model, amp_level="O2", dtype=ms.float16)
    print(f"=====> Building model done.")

    # loading checkpoint
    checkpoint_path = args.checkpoint_path if args.checkpoint_path.lower() not in ("", "none") else None
    if checkpoint_path is not None:
        s_time = time.time()
        print(f"checkpoint loading...")
        state_dict = ms.load_checkpoint(checkpoint_path)
        m, u = ms.load_param_into_net(model, state_dict)

        m = [n for n in m if ".inv_freq" not in n]
        if len(m) > 0:
            print(f"WARNING: missing keys num: {len(m)}, top 10 name is: {m[:10]}")
        if len(u) > 0:
            print(f"WARNING: unexpected keys num: {len(u)}, top 10 name is: {u[:10]}")

        print(f"load checkpoint from `{checkpoint_path}` success, time cost: {time.time() - s_time:.2f}s")
    else:
        print(f"WARNING: No available pre-trained weights")

    while True:
        # prompt = input("Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: ")
        prompt = "What's your name?"

        if prompt == "q":
            print("Generate task done, see you next time!")
            break

        prompt = [prompt,]
        input_ids = np.array(tokenizer(prompt).input_ids)

        # input_ids = ms.Tensor(input_ids, ms.int32)
        # input_kwargs = {}
        # if args.use_embed_input:
        #     input_kwargs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
        # else:
        #     input_kwargs["input_ids"] = input_ids

        input_ids, _, position_ids, attention_mask = \
            preprocess_input_before_generate_numpy_2(
                30, input_ids, None, position_ids=None, attention_mask=None
            )
        input_embeds = model.model.embed_tokens(input_ids)

        output_ids = model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            use_cache=False,
            max_new_tokens=30,
            do_sample=False,
        )
        # output_ids = output_ids.asnumpy()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
        print("=" * 46 + " Result " + "=" * 46)
        print(outputs)
        print("=" * 100)

        break


def run_llama3_generate_2(args):

    print(f"=====> test_llama3_generate:")
    print(f"=====> Building model...")

    s_time = time.time()

    from cambrian.model.language_model.cambrian_llama import CambrianLlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = CambrianLlamaForCausalLM.from_pretrained(args.model_path, use_flash_attention_2=args.use_fa)
    model.model.connector_only = True
    model.embed_tokens = model.model.embed_tokens  # zhy_test

    model = auto_mixed_precision(model, amp_level="O2", dtype=ms.float16)
    print(f"=====> Building model done.")

    # loading checkpoint
    checkpoint_path = args.checkpoint_path if args.checkpoint_path.lower() not in ("", "none") else None
    if checkpoint_path is not None:
        s_time = time.time()
        print(f"checkpoint loading...")
        state_dict = ms.load_checkpoint(checkpoint_path)
        m, u = ms.load_param_into_net(model, state_dict)

        m = [n for n in m if ".inv_freq" not in n]
        if len(m) > 0:
            print(f"WARNING: missing keys num: {len(m)}, top 10 name is: {m[:10]}")
        if len(u) > 0:
            print(f"WARNING: unexpected keys num: {len(u)}, top 10 name is: {u[:10]}")

        print(f"load checkpoint from `{checkpoint_path}` success, time cost: {time.time() - s_time:.2f}s")
    else:
        print(f"WARNING: No available pre-trained weights")

    while True:
        # prompt = input("Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: ")
        prompt = "What's your name?"

        if prompt == "q":
            print("Generate task done, see you next time!")
            break

        prompt = [prompt,]
        input_ids = np.array(tokenizer(prompt).input_ids)

        # input_ids = ms.Tensor(input_ids, ms.int32)
        # input_kwargs = {}
        # if args.use_embed_input:
        #     input_kwargs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
        # else:
        #     input_kwargs["input_ids"] = input_ids

        output_ids = model.generate(
            input_ids,
            images=None,
            use_cache=False,
            max_new_tokens=30,
            do_sample=False,
        )
        # output_ids = output_ids.asnumpy()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
        print("=" * 46 + " Result " + "=" * 46)
        print(outputs)
        print("=" * 100)

        break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=0, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--pynative_synchronize", type=ast.literal_eval, default=True)
    parser.add_argument("--jit_level", type=str, default="O0")
    parser.add_argument("--model_path", type=str, default="../hf_configs/meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--checkpoint_path", type=str, default="../llama3-8b.ckpt")
    parser.add_argument("--use_fa", type=ast.literal_eval, default=True)
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)
    parser.add_argument("--use_embed_input", type=ast.literal_eval, default=True)
    args, _ = parser.parse_known_args()

    if args.ms_mode == ms.GRAPH_MODE:

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

    elif args.ms_mode == ms.PYNATIVE_MODE:
        ms.set_context(
            mode=ms.PYNATIVE_MODE,
            device_target="Ascend",
            pynative_synchronize=True,
            max_device_memory="59GB",
            deterministic="ON"
        )

    else:
        raise ValueError

    # zhy_test
    # run_llama3_generate(args)
    run_llama3_generate_2(args)
