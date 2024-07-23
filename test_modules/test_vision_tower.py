import argparse
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
    image_tensor = process_images([image], image_processor, config)

    out = model(image_tensor)

    print(f"module name: {replace_name}, token_len: {replace_len}, out.shape: {out.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    args, _ = parser.parse_known_args()

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
            "siglip/CLIP-ViT-SO400M-14-384",
            "openai/clip-vit-large-patch14-336",
            "facebook/dinov2-giant-res378",
            "clip-convnext-XXL-multi-stage"
        ],
        "mm_vision_tower_aux_token_len_list": [
            576,
            576,
            576,
            9216
        ],
    }

    for module_name, token_len in zip(
            replace_dict["mm_vision_tower_aux_list"],
            replace_dict["mm_vision_tower_aux_token_len_list"]
    ):
        test_vision_tower(args, config, module_name, token_len)
