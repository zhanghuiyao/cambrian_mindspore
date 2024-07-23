import os
import copy

from ezcolorlog import root_logger as logger

from .clip_encoder import ClipVisionTower
from .siglip_encoder import SiglipVisionTower
from .clip_convnext_encoder import CLIPConvNextTower
from .dino_encoder import DinoVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower is None or not isinstance(vision_tower, str):
        raise ValueError(f'Vision Tower is not specified in the config: {vision_tower_cfg}')

    # CLIP-based Vision Towers
    if "openai/clip" in vision_tower.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower}")
        return ClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "siglip" in vision_tower.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "clip-convnext" in vision_tower.lower():
        logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower}")
        return CLIPConvNextTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # SSL-based Vision Towers
    if "dinov2" in vision_tower.lower():
        logger.info(f"Loading **DINO Vision Tower: {vision_tower}")
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Supervised Vision Towers
    # if "supervised-vit" in vision_tower.lower():
    #     logger.info(f"Loading **Supervised** Vision Tower: {vision_tower}")
    #     return SupervisedViT_VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Other Vision Towers
    # if "diffusion" in vision_tower.lower():
    #     logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower}")
    #     return DiffusionVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # if "sam" in vision_tower.lower():
    #     logger.info(f"Loading **SAM Vision Tower: {vision_tower}")
    #     return SAMVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):
    vision_tower_aux_name_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_list', getattr(vision_tower_cfg, 'vision_tower_aux_list', None))
    vision_tower_aux_token_len_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_token_len_list', getattr(vision_tower_cfg, 'vision_tower_aux_token_len_list', None))
    vision_tower_aux_list = []
    for vision_tower_aux_name, vision_tower_aux_token_len in zip(vision_tower_aux_name_list, vision_tower_aux_token_len_list):
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower_aux_name += "-interp{}".format(vision_tower_aux_token_len)

        # CLIP-based Vision Towers
        if "openai/clip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(ClipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "siglip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "clip-convnext" in vision_tower_aux_name.lower():
            logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(CLIPConvNextTower(vision_tower_aux_name, args=config, **kwargs))

        # SSL-based Vision Towers
        elif "dinov2" in vision_tower_aux_name.lower():
            logger.info(f"Loading **DINO Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(DinoVisionTower(vision_tower_aux_name, args=config, **kwargs))

        # Supervised Vision Towers
        # elif "supervised-vit" in vision_tower_aux_name.lower():
        #     logger.info(f"Loading **Supervised** Vision Tower: {vision_tower_aux_name}")
        #     vision_tower_aux_list.append(SupervisedViT_VisionTower(vision_tower_aux_name, args=config, **kwargs))

        # Other Vision Towers
        # elif "diffusion" in vision_tower_aux_name.lower():
        #     logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower_aux_name}")
        #     vision_tower_aux_list.append(DiffusionVisionTower(vision_tower_aux_name, args=config, **kwargs))
        # elif "sam" in vision_tower_aux_name.lower():
        #     logger.info(f"Loading **SAM Vision Tower: {vision_tower_aux_name}")
        #     vision_tower_aux_list.append(SAMVisionTower(vision_tower_aux_name, args=config, **kwargs))
        else:
            raise ValueError(f'Unknown vision tower: {vision_tower_aux_name}')

    return vision_tower_aux_list
