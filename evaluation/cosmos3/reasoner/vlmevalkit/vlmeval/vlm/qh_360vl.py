import warnings

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel


class QH_360VL(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='qihoo360/360VL-70B', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          torch_dtype=torch.float16,
                                                          low_cpu_mem_usage=True,
                                                          device_map="auto",
                                                          trust_remote_code=True).eval()
        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device='cuda', dtype=torch.float16)
        self.image_processor = vision_tower.image_processor
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate(self, message, dataset=None):

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        print(prompt)
        image = Image.open(image_path).convert('RGB')
        terminators = [
            self.tokenizer.convert_tokens_to_ids('<|eot_id|>',)
        ]
        inputs = self.model.build_conversation_input_ids(self.tokenizer,
                                                         query=prompt,
                                                         image=image,
                                                         image_processor=self.image_processor)
        input_ids = inputs['input_ids'].to(device='cuda', non_blocking=True)
        images = inputs['image'].to(dtype=torch.float16, device='cuda', non_blocking=True)

        output_ids = self.model.generate(input_ids=input_ids,
                                         images=images,
                                         do_sample=False,
                                         num_beams=1,
                                         max_new_tokens=512,
                                         eos_token_id=terminators,
                                         use_cache=True)

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        response = outputs.strip()

        return response
