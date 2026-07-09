import base64
import logging
from io import BytesIO
from mimetypes import guess_type

import torch
from PIL import Image

from .base import BaseModel


class PaliGemma(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='google/paligemma-3b-mix-448', **kwargs):
        try:
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        ).eval()
        self.model = model.cuda()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')

        model_inputs = self.processor(
            text=prompt, images=image, return_tensors='pt'
        ).to('cuda')
        input_len = model_inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=512, do_sample=False
            )
            generation = generation[0][input_len:]
            res = self.processor.decode(generation, skip_special_tokens=True)
        return res


class Gemma3(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='google/gemma-3-4b-it', **kwargs):
        logging.info(
            "Please install transformers via \n"
            "pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"
        )
        try:
            import torch
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except Exception as e:
            logging.critical('Please install torch and transformers')
            raise e

        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = 24
        if self.use_vllm:
            from vllm import LLM

            # Set tensor_parallel_size [8, 4, 2, 1] based on the number of available GPUs
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for Llama4 inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            self.llm = LLM(
                model=model_path,
                max_num_seqs=4,
                max_model_len=16384,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )
            # export VLLM_WORKER_MULTIPROC_METHOD=spawn
        else:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path, device_map="cuda", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            ).eval()
            self.device = self.model.device

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.system_prompt = kwargs.pop('system_prompt', 'You are a helpful assistant. ')
        default_kwargs = {
            'do_sample': False,
            'max_new_tokens': 4096
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def message2pipeline(self, message):
        ret = []
        if hasattr(self, 'system_prompt') and self.system_prompt is not None:
            ret = [
                dict(role='system', content=[dict(type='text', text=self.system_prompt)])
            ]
        content = []
        for m in message:
            if m['type'] == 'text':
                content.append(dict(type='text', text=m['value']))
            elif m['type'] == 'image':
                content.append(dict(type='image', url=m['value']))
        ret.append(dict(role='user', content=content))
        return ret

    def encode_image(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"
        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)

        encoded_image = self._encode_image(image, image_format)

        return encoded_image

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def message_to_promptimg_vllm(self, message, dataset=None):
        processed_message = []
        images = []
        num_images = 0
        for item in message:
            if item['type'] == 'text':
                processed_message.append({
                    "type": "text",
                    "text": item['value']
                })
            elif item['type'] == 'image':
                if num_images < self.limit_mm_per_prompt:
                    image_path = item['value']
                    encoded_image = self.encode_image(image_path)
                    image = Image.open(BytesIO(base64.b64decode(encoded_image)))
                    image.load()
                    processed_message.append({
                        "type": "image",
                        "image": "",
                    })
                    images.append(image)
                    num_images += 1
        if num_images >= self.limit_mm_per_prompt:
            logging.warning(
                f"Number of images exceeds the limit of {self.limit_mm_per_prompt}."
                f"Only the first {self.limit_mm_per_prompt} images will be used."
            )
        return processed_message, images

    def generate_inner_transformers(self, message, dataset=None):
        messages = self.message2pipeline(message)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        prompt, images = self.message_to_promptimg_vllm(message, dataset=dataset)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=self.kwargs['max_new_tokens'])
        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": images
                },
            },
            sampling_params=sampling_params
        )
        for o in outputs:
            generated_text = o.outputs[0].text
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
