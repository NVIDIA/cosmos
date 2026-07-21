import logging

import torch
from PIL import Image

from .base import BaseModel


class Chameleon(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='facebook/chameleon-7b', **kwargs):
        try:
            from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
        except Exception as e:
            logging.critical('Please install the latest transformers.')
            raise e

        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)

        self.model = model.cuda().eval()
        self.processor = processor

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for x in message:
            if x['type'] == 'text':
                content += x['value']
            elif x['type'] == 'image':
                content += '<image>\n'
                images.append(Image.open(x['value']))

        inputs = self.processor(
            text=[content],
            images=images,
            padding=True,
            return_tensors='pt'
        ).to(device='cuda', dtype=torch.bfloat16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=2048)
        input_token_len = inputs.input_ids.shape[1]
        text = self.processor.batch_decode(
            generate_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return text
