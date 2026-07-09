import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import BaseModel

try:
    import av  # noqa: F401
    from decord import VideoReader, cpu  # noqa: F401
except ImportError:
    print("Please install pyav to use video processing functions.")


def move_to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


class Logics_Thinking(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path: str = "Logics-MLLM/Logics-Thinking-8B",
                 **kwargs):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.stop_str = "<|im_end|>"

    def generate_inner_image(self, message):
        text_prompt = ""
        image_paths = []

        for msg in message:
            if msg["type"] == "text":
                text_prompt += msg["value"]
            elif msg["type"] == "image":
                image_paths.append(msg["value"])

        inputs = self.processor(
            text=text_prompt,
            images=image_paths,
            return_tensors="pt"
        )

        DEVICE = self.model.device
        inputs = move_to_device(inputs, DEVICE)
        generated_ids = self.model.generate(**inputs)
        text_outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text_outputs = text_outputs.strip()

        if text_outputs.endswith(self.stop_str):
            text_outputs = text_outputs[:-len(self.stop_str)]
        text_outputs = text_outputs.strip()
        return text_outputs

    def generate_inner(self, message, dataset):
        return self.generate_inner_image(message)
