# vlmeval/vlm/cosmos.py
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from .base import BaseModel


class Cosmos(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="nvidia/Cosmos-Reason1-7B", **kwargs):
        from vllm import LLM, SamplingParams
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10}
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=4096,
        )

    def message_for_promptimg(self, message):
        processed_messages = []
        for part in message:
            if part["type"] == "text":
                processed_messages.append({"type": "text", "text": part["value"].strip()})
            elif part["type"] == "image":
                processed_messages.append({"type": "image", "image": part["value"]})
            elif part["type"] == "video":
                processed_messages.append({
                    "type": "video",
                    "video": part["value"],
                    "fps": part.get("fps", 4)
                })
        return processed_messages

    def generate_inner(self, message, dataset=None):
        user_message = {
            "role": "user",
            "content": self.message_for_promptimg(message)
        }
        system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question in the following format:\n"
                "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
            )
        }
        messages = [system_prompt, user_message]

        # Generate prompt and multimodal inputs
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
