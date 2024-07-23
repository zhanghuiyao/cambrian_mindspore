import argparse
import mindspore as ms
from mindspore import Tensor
from cambrian.transformers.models.llama import LlamaModel
from transformers import AutoTokenizer


def test_llama3(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaModel.from_pretrained(model_path)

    prompt = "hello world."
    input_ids = Tensor(tokenizer(prompt), ms.int32)

    result = model(input_ids)

    print(result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="")
    args, _ = parser.parse_known_args()

    test_llama3(args.model_path)
