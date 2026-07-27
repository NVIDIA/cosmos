import logging
import os.path as osp
import sys
import warnings

import torch

from .base import BaseModel


class PandaGPT(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, name, root=None, **kwargs):
        if root is None:
            raise ValueError('Please set `root` to PandaGPT code directory, which is cloned from here: ')

        assert name == 'PandaGPT_13B'
        self.name = name
        sys.path.append(osp.join(root, 'code'))
        try:
            from model.openllama import OpenLLAMAPEFTModel
        except Exception as e:
            logging.critical(
                'Please first install PandaGPT and set the root path to use PandaGPT, '
                'which is cloned from here: https://github.com/yxuansu/PandaGPT. '
            )
            raise e

        self.args = {
            'model': 'openllama_peft',
            'imagebind_ckpt_path': osp.join(root, 'pretrained_ckpt/imagebind_ckpt'),
            'vicuna_ckpt_path': osp.join(root, 'pretrained_ckpt/vicuna_ckpt/13b_v0'),
            'delta_ckpt_path': osp.join(root, 'pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt'),
            'stage': 2,
            'max_tgt_len': 512,
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }
        model = OpenLLAMAPEFTModel(**self.args)
        delta_ckpt = torch.load(self.args['delta_ckpt_path'], map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
        torch.cuda.empty_cache()
        self.model = model.eval().half().cuda()
        kwargs_default = {'top_p': 0.9, 'do_sample': False, 'max_tgt_len': 128, 'temperature': 0.001}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        struct = {
            'prompt': prompt,
            'image_paths': [image_path],
            'audio_paths': [],
            'video_paths': [],
            'thermal_paths': [],
            'modality_embeds': []
        }
        struct.update(self.kwargs)
        resp = self.model.generate(struct)
        return resp
