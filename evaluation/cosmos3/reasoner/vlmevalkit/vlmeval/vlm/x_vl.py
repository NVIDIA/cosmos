import torch
from PIL import Image

from ..dataset import DATASET_MODALITY
from .base import BaseModel


class X_VL_HF(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="YannQi/X-VL-4B", **kwargs):
        from transformers import AutoModel, AutoProcessor
        assert model_path is not None, "Model path must be provided."
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to('cuda', torch.float16)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.model_path = model_path

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=16384, use_cache=True)
        answer = self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.split('</think>')[-1].strip()
        return answer

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == "VIDEO" and 'megabench' not in dataset.lower():
            raise NotImplementedError("Video generation is not supported yet.")
        else:
            return self.generate_inner_image(message, dataset)
