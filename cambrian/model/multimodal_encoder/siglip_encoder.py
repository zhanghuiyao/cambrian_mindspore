import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from cambrian.open_clip.factory import create_model_from_pretrained

from cambrian.model.multimodal_encoder.base_encoder import ProcessorWrapper
from cambrian.model.multimodal_encoder.clip_encoder import ClipVisionTower


def extract_res_interp(model_name):
    valid_model_prefixes = {
        "siglip/CLIP-ViT-SO400M-14-384":"timm/ViT-SO400M-14-SigLIP-384",
        "timm/ViT-SO400M-14-SigLIP-384":"timm/ViT-SO400M-14-SigLIP-384",
        "siglip/CLIP-ViT-SO400M-14":"timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP":"timm/ViT-SO400M-14-SigLIP"
    }

    res = 384 if '384' in model_name else 224
    interp = None

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class SiglipVisionTower(ClipVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, res, interp = extract_res_interp(vision_tower_name)
        self.vision_tower_name = base_model_name

        # replace model name to local path
        if self.vision_tower_name == "timm/ViT-SO400M-14-SigLIP-384":
            replace_local_path = "./cambrian/hf-configs/timm-ViT-SO400M-14-SigLIP-384"
            print(f"Warning: SiglipVisionTower, replace vision_tower_name to local path")
            self.vision_tower_name = replace_local_path

        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        clip_model, processor = create_model_from_pretrained(self.vision_tower_name)

        self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size[0]
        self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad = self.unfreeze_mm_vision_tower
        self.is_loaded = True

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = ops.interpolate(
                image_features.to(ms.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(start_dim=1, end_dim=2)

        return image_features

    def _forward(self, images, interpolate_token = 576):
        image_features = self.vision_tower.forward_features(images.to(dtype=self.dtype))
        interp_features = self.interpolate(image_features)
        if self.unfreeze_mm_vision_tower:
            interp_features = ops.stop_gradient(interp_features)
        return interp_features
