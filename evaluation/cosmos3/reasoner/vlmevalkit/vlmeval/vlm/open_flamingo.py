import logging
import os.path as osp
import warnings

import torch
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import get_cache_path, splitlen
from .base import BaseModel


class OpenFlamingo(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 name,
                 mpt_pth=None,
                 ckpt_pth=None,
                 **kwargs):

        if mpt_pth is None:
            raise ValueError(
                'Please set `mpt_pth` to the directory of MPT-7B, which is cloned from here: '
                'https://huggingface.co/mosaicml/mpt-7b. '
            )
            raise ValueError
        if ckpt_pth is None:
            raise ValueError(
                'Please set `ckpt_pth` to the openflamingo ckpt, which is the `checkpoint.pt` file downloaded '
                'from: https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b/tree/main. '
            )
        else:
            if osp.exists(ckpt_pth):
                if ckpt_pth.endswith('checkpoint.pt'):
                    pass
                elif osp.isdir(ckpt_pth):
                    ckpt_pth = osp.join(ckpt_pth, 'checkpoint.pt')
                    if not osp.exists(ckpt_pth):
                        raise ValueError(f'File {ckpt_pth} does not exist. ')
            elif splitlen(ckpt_pth, '/') == 2:
                cache_path = get_cache_path(ckpt_pth)
                if cache_path is None:
                    snapshot_download(ckpt_pth)
                cache_path = get_cache_path(ckpt_pth)
                if cache_path is None:
                    raise ValueError(f'Directory {cache_path} does not exist. ')
                else:
                    ckpt_pth = osp.join(cache_path, 'checkpoint.pt')

        self.name = name
        assert name in ['v2']
        self.mpt_pth = mpt_pth
        try:
            from open_flamingo import create_model_and_transforms
        except Exception as e:
            logging.critical('Please first install open_flamingo to use OpenFlamingo')
            raise e

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path='ViT-L-14',
            clip_vision_encoder_pretrained='openai',
            lang_encoder_path=mpt_pth,
            tokenizer_path=mpt_pth,
            cross_attn_every_n_layers=4)
        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt, strict=False)
        torch.cuda.empty_cache()
        self.model = model.eval().cuda()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.image_proc = image_processor

        kwargs_default = dict(max_new_tokens=512, num_beams=3)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        vision_x = []
        prompt = ''
        for msg in message:
            if msg['type'] == 'image':
                img = Image.open(msg['value'])
                vision_x.append(self.image_proc(img).unsqueeze(0))
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value']
        prompt += 'Answer: '
        vision_x = torch.cat(vision_x, dim=0) if len(vision_x) > 1 else vision_x[0]
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = self.tokenizer([prompt], return_tensors='pt')
        generated_text = self.model.generate(
            vision_x=vision_x.cuda(),
            lang_x=lang_x['input_ids'].cuda(),
            attention_mask=lang_x['attention_mask'].cuda(),
            **self.kwargs)
        generated_text = self.tokenizer.decode(generated_text[0])
        text = generated_text[len(prompt):].split('<|endofchunk|>')[0]
        return text
