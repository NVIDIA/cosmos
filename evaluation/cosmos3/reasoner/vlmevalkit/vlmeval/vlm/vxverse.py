import os.path as osp
import sys
import warnings

import torch
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import get_cache_path
from .base import BaseModel

model_cfgs = {
    'XVERSE-V-13B': {
        'arch': 'vxverse',
        'model_type': 'pretrain_xverse13b-chat',
        'max_txt_len': 512,
        'end_sym': '<|endoftext|>',
        'low_resource': False,
        'prompt_template': 'Human: {}\nAssistant: ',
        'ckpt': 'xverse/XVERSE-V-13B',
        'lora_r': 128,
        'lora_alpha': 256,
        'lora_dropout': 0.05,
        'lora_target_modules': 'all_linear',
        'has_qformer': False,
        'n_proj_layers': 2,
        'vit_model': 'openai/clip-vit-large-patch14',
        'vit_path': 'openai/clip-vit-large-patch14',
        'image_size': 224,
        'drop_path_rate': 0,
        'vit_precision': 'fp16',
        'llama_model': 'xverse/XVERSE-13B-Chat',
    }
}


class VXVERSE(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_name='XVERSE-V-13B', root=None, **kwargs):
        from omegaconf import OmegaConf
        if root is None:
            warnings.warn('Please set root to the directory of vxverse.')

        if model_name == 'XVERSE-V-13B':
            cfg = model_cfgs['XVERSE-V-13B']
        else:
            raise NotImplementedError

        ckpt_dir = cfg['ckpt']
        if not osp.isdir(ckpt_dir):
            cache_path = get_cache_path(ckpt_dir)
            if cache_path is not None:
                ckpt_dir = cache_path
            else:
                ckpt_dir = snapshot_download(repo_id=ckpt_dir)
        assert osp.exists(ckpt_dir) and osp.isdir(ckpt_dir)
        ckpt = osp.join(ckpt_dir, 'adapter_and_lora.bin')
        cfg['ckpt'] = ckpt
        model_cfg = OmegaConf.create(cfg)

        self.model_name = model_name

        self.root = root
        sys.path.append(self.root)

        from vxverse.common.registry import registry
        from vxverse.conversation.conversation import CONV_VISION_XVERSE

        device = torch.cuda.current_device()
        self.device = device

        model_cls = registry.get_model_class(model_cfg.arch)
        model = model_cls.from_config(model_cfg)
        model = model.to(device)
        model.eval()
        vis_processor_cfg = OmegaConf.create(dict(name='hd_image_train', image_size=224))
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)

        self.model = model
        self.vis_processor = vis_processor
        self.vis_processor_cfg = vis_processor_cfg

        self.CONV_VISION = CONV_VISION_XVERSE
        self.CONV_VISION.system = ''
        stop_words_ids = [[835], [2277, 29937]]
        self.stop_words_ids = stop_words_ids
        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)

        if self.vis_processor_cfg.name == 'hd_image_train':
            patches_per_image = [[image.shape[0]]]
            image = [image]
        else:
            patches_per_image = None
            image = image.unsqueeze(0)

        chat_state = self.CONV_VISION.copy()
        texts = self.prepare_texts([prompt], chat_state)
        texts = [text.lstrip() for text in texts]
        answers = self.model.generate(
            image,
            texts,
            patches_per_images=patches_per_image,
            do_sample=False,
            stop_words_ids=self.stop_words_ids,
            **self.kwargs
        )
        return answers[0]

    def prepare_texts(self, texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [
            conv.append_message(conv.roles[0], '<ImageHere>\n{}'.format(text))
            for conv, text in zip(convs, texts)
        ]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts
