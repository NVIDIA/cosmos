import logging
import os.path as osp

import torch
from PIL import Image

from vlmeval.smp import splitlen
from .base import BaseModel


class InstructBLIP(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, name):
        self.config_map = {
            'instructblip_7b': 'misc/blip2_instruct_vicuna7b.yaml',
            'instructblip_13b': 'misc/blip2_instruct_vicuna13b.yaml',
        }

        self.file_path = __file__
        config_root = osp.dirname(self.file_path)

        try:
            from lavis.common.registry import registry
            from lavis.models import load_preprocess
            from omegaconf import OmegaConf
        except Exception as e:
            logging.critical('Please install lavis before using InstructBLIP. ')
            raise e

        assert name in self.config_map
        cfg_path = osp.join(config_root, self.config_map[name])
        cfg = OmegaConf.load(cfg_path)

        model_cfg = cfg.model
        assert osp.exists(model_cfg.llm_model) or splitlen(model_cfg.llm_model) == 2
        model_cls = registry.get_model_class(name='blip2_vicuna_instruct')
        model = model_cls.from_config(model_cfg)
        model.eval()

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        device = self.device
        model.to(device)
        self.model = model
        self.kwargs = {'max_length': 512}

        preprocess_cfg = cfg.preprocess
        vis_processors, _ = load_preprocess(preprocess_cfg)
        self.vis_processors = vis_processors

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        vis_processors = self.vis_processors
        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = vis_processors['eval'](raw_image).unsqueeze(0).to(self.device)
        outputs = self.model.generate(dict(image=image_tensor, prompt=prompt))
        return outputs[0]
