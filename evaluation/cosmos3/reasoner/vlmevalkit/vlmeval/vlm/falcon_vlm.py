from PIL import Image

from .base import BaseModel


class Falcon2VLM(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='tiiuae/falcon-11B-vlm', **kwargs):
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self.model_path = model_path
        self.processor = LlavaNextProcessor.from_pretrained(model_path, tokenizer_class='PreTrainedTokenizerFast')
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='cuda').eval()
        default_kwargs = {'max_new_tokens': 512}
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')

        prompt = f'User:<image>\n{prompt} Falcon:'
        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to('cuda')

        output = self.model.generate(**inputs, **self.kwargs)
        prompt_length = inputs['input_ids'].shape[1]
        model_response = self.processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
        return model_response
