import os.path as osp
import sys
import warnings

import torch
from transformers import StoppingCriteriaList

from .base import BaseModel


class MiniGPT4(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self,
                 mode='v2',
                 root='/mnt/petrelfs/share_data/duanhaodong/MiniGPT-4/',
                 temperature=1,
                 max_out_len=512):

        if root is None:
            warnings.warn(
                'Please set root to the directory of MiniGPT-4, which is cloned from here: '
                'https://github.com/Vision-CAIR/MiniGPT-4. '
            )

        if mode == 'v2':
            cfg = 'minigptv2_eval.yaml'
        elif mode == 'v1_7b':
            cfg = 'minigpt4_7b_eval.yaml'
        elif mode == 'v1_13b':
            cfg = 'minigpt4_13b_eval.yaml'
        else:
            raise NotImplementedError

        self.mode = mode
        self.temperature = temperature
        self.max_out_len = max_out_len
        self.root = root
        this_dir = osp.dirname(__file__)

        self.cfg = osp.join(this_dir, 'misc', cfg)
        sys.path.append(self.root)

        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import (CONV_VISION_minigptv2, CONV_VISION_Vicuna0,
                                                        StoppingCriteriaSub)
        from omegaconf import OmegaConf

        device = torch.cuda.current_device()
        self.device = device

        cfg_path = self.cfg
        cfg = OmegaConf.load(cfg_path)

        model_cfg = cfg.model
        model_cfg.device_8bit = device
        model_cls = registry.get_model_class(model_cfg.arch)
        model = model_cls.from_config(model_cfg)
        model = model.to(device)
        model.eval()
        vis_processor_cfg = cfg.datasets.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model = model
        self.vis_processor = vis_processor

        self.CONV_VISION = CONV_VISION_minigptv2 if self.mode == 'v2' else CONV_VISION_Vicuna0
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device) for ids in stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def generate_inner(self, message, dataset=None):
        from minigpt4.conversation.conversation import Chat
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        if self.mode == 'v2':
            chat = Chat(self.model, self.vis_processor, device=self.device)
        else:
            chat = Chat(self.model, self.vis_processor, device=self.device, stopping_criteria=self.stopping_criteria)

        chat_state = self.CONV_VISION.copy()
        img_list = []
        _ = chat.upload_img(image_path, chat_state, img_list)
        chat.encode_img(img_list)
        chat.ask(prompt, chat_state)
        with torch.inference_mode():
            msg = chat.answer(conv=chat_state, img_list=img_list)[0]
        return msg
