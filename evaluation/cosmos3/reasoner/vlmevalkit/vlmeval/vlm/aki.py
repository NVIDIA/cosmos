import warnings

import torch
from PIL import Image
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor
from transformers import AutoConfig, AutoTokenizer

from .base import BaseModel

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AKI(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self,
                 name,
                 ckpt_pth=None,
                 **kwargs):

        self.name = name
        try:
            from open_flamingo.src.modeling_aki import AKI
        except Exception:
            raise ImportError('Please first install AKIVLM from https://github.com/sony/aki')

        # replace GenerationMixin to modify attention mask handling
        from open_flamingo import _aki_update_model_kwargs_for_generation
        from transformers.generation.utils import GenerationMixin
        GenerationMixin._update_model_kwargs_for_generation = _aki_update_model_kwargs_for_generation

        config = AutoConfig.from_pretrained(ckpt_pth)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_pth)
        model = AKI.from_pretrained(ckpt_pth, tokenizer=tokenizer)

        n_px = getattr(config, "n_px", 384)
        norm_mean = getattr(config, "norm_mean", 0.5)
        norm_std = getattr(config, "norm_std", 0.5)

        image_processor = Compose([
            Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True),
            Lambda(lambda x: x.convert('RGB')),
            ToTensor(),
            Normalize(mean=(norm_mean, norm_mean, norm_mean), std=(norm_std, norm_std, norm_std))
        ])
        self.model = model.eval().cuda()

        tokenizer.padding_side = 'left'
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
        self.image_proc = image_processor

        kwargs_default = {
            'max_new_tokens': 512,
            'temperature': 0.0,
            'do_sample': False,
            'eos_token_id': tokenizer.eos_token_id,
        }
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def apply_prompt_template(self, query):
        SYSTEM_BASE = "A chat between a curious user and an artificial intelligence assistant."
        SYSTEM_DETAIL = "The assistant gives helpful, detailed, and polite answers to the user's questions."
        SYSTEM_MESSAGE = SYSTEM_BASE + " " + SYSTEM_DETAIL
        SYSTEM_MESSAGE_ROLE = '<|system|>' + '\n' + SYSTEM_MESSAGE + '<|end|>\n'

        s = (
            f'{SYSTEM_MESSAGE_ROLE}'
            f'<|user|>\n{query}<|end|>\n<|assistant|>\n'
        )
        return s

    def generate_inner(self, message, dataset=None):
        vision_x, prompt = [], ''
        for msg in message:
            if msg['type'] == 'image':
                img = Image.open(msg['value']).convert('RGB')

                # [NOTE]: only use the first image in this work if including multiple images in a sample
                if len(vision_x) == 0:
                    vision_x.append(self.image_proc(img).unsqueeze(0))
                    prompt += '<image>'
                else:
                    warnings.warn('======Only the first image is used in the input.')
            elif msg['type'] == 'text':
                prompt += msg['value']
                # prompt += f"\nAnswer the question using a single word or phrase. {msg['value']}"      # for YorN

        vision_x = torch.cat(vision_x, dim=0) if len(vision_x) > 1 else vision_x[0]
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        prompt = self.apply_prompt_template(prompt)
        lang_x = self.tokenizer([prompt], return_tensors='pt')

        generated_text = self.model.generate(
            vision_x=vision_x.cuda(),
            lang_x=lang_x['input_ids'].cuda(),
            attention_mask=lang_x['attention_mask'].cuda(),
            **self.kwargs)
        generated_text = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)
        return generated_text
