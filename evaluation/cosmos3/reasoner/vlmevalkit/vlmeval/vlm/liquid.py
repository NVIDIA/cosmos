import torch
from PIL import Image

from .base import BaseModel


class LFM2VL(BaseModel):
    def __init__(self, model_path, **kwargs):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.default_instruction_prompt = (
            "\nPlease answer directly with only the final answer, "
            "do not give any explanation."
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path,
                # attn_implementation="flash_attention_2",
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
            .cuda()
            .eval()
        )

        kwargs_default = {"max_new_tokens": 1024, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def custom_instruction_prompt_by_dataset(self, dataset):
        if dataset == "MathVista_MINI" or dataset == "MM-IFEval" or dataset == "MMVet":
            return ""
        else:
            return self.default_instruction_prompt

    def message_to_chat_messages(self, message, instruction_prompt, dataset):
        single_turn_messages = []
        image_paths = []

        for item in message:
            if item["type"] == "image":
                image_paths.append(item["value"])
                single_turn_messages.append({"type": "image", "url": item["value"]})
            elif item["type"] == "text":
                single_turn_messages.append({"type": "text", "text": item["value"]})
        if instruction_prompt:
            single_turn_messages.append({"type": "text", "text": instruction_prompt})

        if dataset == "MM-IFEval":
            # move images to the beginning of the conversation list, in the same order as they appear in the message
            index = 0
            image_index = 0
            while index < len(single_turn_messages):
                if single_turn_messages[index]["type"] == "image":
                    single_turn_messages.insert(
                        image_index, single_turn_messages.pop(index)
                    )
                    image_index += 1
                index += 1

        chat_messages = [{"role": "user", "content": single_turn_messages}]
        images_pil = [Image.open(p).convert("RGB") for p in image_paths]

        return chat_messages, images_pil

    def generate_inner(self, message, dataset=None):
        instruction_prompt = self.custom_instruction_prompt_by_dataset(dataset)

        chat_messages, images = self.message_to_chat_messages(message, instruction_prompt, dataset)

        chat_inputs = self.processor.apply_chat_template(chat_messages, add_generation_prompt=True, tokenize=False)

        generation_inputs = self.processor(
            images=images,
            text=[chat_inputs],
            return_tensors="pt",
        ).to(dtype=torch.bfloat16, device="cuda")

        history = self.model.generate(**generation_inputs, **self.kwargs)
        decoded = self.processor.decode(history[0], skip_special_tokens=False)
        assistant_response = decoded.split("<|im_start|>assistant\n")[-1].strip()
        if assistant_response.endswith("<|im_end|>"):
            assistant_response = assistant_response[:-10]
        return assistant_response

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset)
