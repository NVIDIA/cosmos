import warnings

import torch
import transformers
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel


class LLama3Mixsense(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='Zero-Vision/Llama-3-MixSenseV1_1', **kwargs):
        assert model_path is not None
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map="cuda"
        ).eval()
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        input_ids = self.model.text_process(prompt, self.tokenizer).to(device='cuda')
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.model.image_process([image]).to(dtype=self.model.dtype, device='cuda')
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=2048,
                use_cache=True,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids(['<|eot_id|>'])[0],
                ],
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
