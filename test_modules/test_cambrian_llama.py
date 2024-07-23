import numpy as np

from cambrian.model.language_model.cambrian_llama import CambrianLlamaModel, CambrianLlamaForCausalLM


def test_CambrianLlamaModel(model_path: str):
    cambrian_llama = CambrianLlamaModel.from_pretrained(model_path)


def test_CambrianLlamaForCausalLM(model_path: str):
    cambrian_llama_causal = CambrianLlamaForCausalLM.from_pretrained(model_path)


def test_generate_wo_image(model_path: str):
    from transformers import AutoTokenizer

    input_ids = np.random.randint(0, 10000, size=(1, 50))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    cambrian_llama_causal = CambrianLlamaForCausalLM.from_pretrained(model_path)

    output_ids = cambrian_llama_causal.generate(input_ids)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(outputs)



if __name__ == '__main__':
    model_path = ""
    test_CambrianLlamaModel(model_path)
    # test_CambrianLlamaForCausalLM(model_path)
