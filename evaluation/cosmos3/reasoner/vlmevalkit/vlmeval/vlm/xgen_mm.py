import logging

from PIL import Image

from .base import BaseModel


class XGenMM(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5', **kwargs):
        try:
            from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer
        except Exception as err:
            logging.critical('Please install the latest version transformers.')
            raise err

        model = AutoModelForVision2Seq.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True, torch_dtype='auto'
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False, legacy=False
        )
        tokenizer = model.update_special_tokens(tokenizer)
        tokenizer.eos_token = '<|end|>'
        tokenizer.padding_side = 'left'
        image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def apply_prompt_template(self, query):
        s = (
            '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
            f'<|user|>\n{query}<|end|>\n<|assistant|>\n'
        )
        return s

    def generate_inner(self, message, dataset=None):

        content, images, image_sizes = '', [], []

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                image = Image.open(msg['value']).convert('RGB')
                images.append(self.image_processor([image], image_aspect_ratio='anyres')['pixel_values'].to('cuda'))
                image_sizes.append(image.size)
                content += '<image> '

        inputs = {'pixel_values': [images]}
        prompt = self.apply_prompt_template(content)
        language_inputs = self.tokenizer([prompt], return_tensors='pt').to('cuda')
        inputs.update(language_inputs)

        generation_args = {
            'max_new_tokens': 1024,
            'temperature': 0.0,
            'do_sample': False,
            'top_p': None,
            'num_beams': 1
        }
        generation_args.update(self.kwargs)

        generate_ids = self.model.generate(
            **inputs, image_size=[image_sizes],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generation_args
        )

        # remove input tokens
        response = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True).split('<|end|>')[0]

        return response
