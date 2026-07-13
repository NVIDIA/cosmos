import os

from .oryx_vit import SigLIPViTAnysizeWrapper


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'vision_tower', getattr(vision_tower_cfg, 'mm_vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    print(f"Buiding OryxViTWrapper from {vision_tower}...")
    # path = vision_tower.split(":")[1]
    return SigLIPViTAnysizeWrapper(vision_tower, path=vision_tower, args=vision_tower_cfg, **kwargs)
