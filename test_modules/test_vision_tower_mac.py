import argparse
import mindspore as ms
from mindspore import Tensor
from PIL import Image
from cambrian.model.language_model.cambrian_llama import CambrianConfig
from cambrian.model.multimodal_encoder.builder import build_vision_tower_aux_list
from cambrian.mm_utils import process_images


def test_vision_tower(args, config, replace_name: str = None, replace_len: int = None):

    config.mm_vision_tower_aux_list = [replace_name,]
    config.mm_vision_tower_aux_token_len_list = [replace_len,]

    model = build_vision_tower_aux_list(config)[0]

    image = Image.open(args.image_path).convert('RGB')
    image_processor = [model.image_processor,]
    image = process_images([image], image_processor, config)

    # (num, bs, 3, 384, 384) -> (1, 3, 384, 384)
    image_tensor = Tensor(image[0], ms.float32)

    print(f"Image Process: image_tensor.shape: {image_tensor.shape}")

    out = model(image_tensor)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="./cambrian/hf-configs/nyu-visionx-cambrian-8b")
    parser.add_argument("--image_path", type=str, default="./images/cambrian.png")
    args, _ = parser.parse_known_args()

    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    config, _ = CambrianConfig.from_pretrained(
        args.model_path,
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

    replace_dict = {
        "mm_vision_tower_aux_list": [
            "siglip/CLIP-ViT-SO400M-14-384",        # https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384
            "openai/clip-vit-large-patch14-336",    # https://huggingface.co/openai/clip-vit-large-patch14-336
            "facebook/dinov2-giant-res378",         # https://huggingface.co/facebook/dinov2-giant
            "clip-convnext-XXL-multi-stage"         # https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup
        ],
        "mm_vision_tower_aux_token_len_list": [
            576,
            576,
            576,
            9216
        ],
    }

    assert len(replace_dict['mm_vision_tower_aux_list']) == len(replace_dict['mm_vision_tower_aux_token_len_list'])

    module_len = len(replace_dict['mm_vision_tower_aux_list'])
    for i, (module_name, token_len) in enumerate(zip(replace_dict["mm_vision_tower_aux_list"], replace_dict["mm_vision_tower_aux_token_len_list"])):
        print(f"======> {i}/{module_len}, Before, model name: {module_name}")

        out = test_vision_tower(args, config, module_name, token_len)

        print(f"======> {i}/{module_len}, After, model name: {module_name}, token_len: {token_len}, out.shape: {out.shape}")