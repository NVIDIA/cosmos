import logging
import os
import warnings

import torch
from transformers import AutoModel, AutoProcessor

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin


def ensure_image_url(image: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:image;"]
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return "file://" + image
    raise ValueError(f"Invalid image: {image}")


def ensure_video_url(video: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:video;"]
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return "file://" + video
    raise ValueError(f"Invalid video: {video}")


class KeyeChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path='Kwai-Keye/Keye-VL-1_5-8B',
        max_new_tokens=8192,  # Use a larger value for "think" models to avoid output truncation (e.g., MMMU)
        top_p=0.001,
        top_k=1,
        temperature=0,
        repetition_penalty=1.0,
        do_sample=False,
        use_custom_prompt: bool = False,
        system_prompt: str | None = None,
        verbose: bool = True,
        no_think: bool = False,
        think: bool = False,
        use_vllm: bool = False,
        min_pixels: int = None,
        max_pixels: int = None,
        post_process: bool = True,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.use_vllm = use_vllm

        if not self.use_vllm:
            self.generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )
            self.model = (
                AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                    device_map="cuda",
                )
                .eval()
                .cuda()
            )
        else:
            from vllm import LLM, SamplingParams

            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1

            self.generate_kwargs = SamplingParams(
                max_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            self.model = LLM(
                model=model_path,
                max_num_batched_tokens=161072,
                max_model_len=161072,
                trust_remote_code=True,
                enforce_eager=False,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={"image": 100, "video": 10},
            )

        self.system_prompt = system_prompt
        self.verbose = verbose

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.no_think = no_think
        self.think = think

        self.post_process = post_process

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s["type"] == "image":
                item = {"type": "image", "image": ensure_image_url(s["value"])}
                if dataset == "OCRBench":
                    item["min_pixels"] = 10 * 10 * 28 * 28
                    warnings.warn(
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}"
                    )
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def add_think_token(self, messages, mode):
        for i in range(len(messages[-1]["content"]) - 1, -1, -1):
            if messages[-1]["content"][i]["type"] == "text":
                messages[-1]["content"][i]["text"] += mode
                break
        return messages

    def post_process_func(self, prediction):
        import re

        def get_boxed(response, bb="\\boxed{"):
            resp = response.split(bb)[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]
            return response

        if "</analysis>" in prediction:
            prediction = prediction.split("</analysis>")[-1]
        if "</think>" in prediction:
            prediction = prediction.split("</think>")[-1].lstrip("\n").strip()
        if "<answer>" not in prediction:
            boxed_matches = get_boxed(prediction, bb=r"\boxed{")
            if len(boxed_matches) != len(prediction):
                return boxed_matches
            else:
                boxed_matches = get_boxed(prediction, bb="\boxed{")
                return (
                    boxed_matches
                    if len(boxed_matches) != len(prediction)
                    else prediction
                )
        matches = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
        if matches:
            content_match = matches[-1]
            boxed_matches = get_boxed(content_match, bb=r"\boxed{")
            if len(boxed_matches) != len(content_match):
                return boxed_matches
            else:
                boxed_matches = get_boxed(content_match, bb="\boxed{")
                return (
                    boxed_matches
                    if len(boxed_matches) != len(content_match)
                    else content_match
                )
        else:
            return prediction

    def generate_inner(self, message, dataset=None):
        try:
            from keye_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "keye_vl_utils not found, please install it via 'pip install keye-vl-utils'"
            )
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )

        if self.no_think:
            messages = self.add_think_token(messages, "/no_think")
        elif self.think:
            messages = self.add_think_token(messages, "/think")

        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)

        if not self.use_vllm:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_processor_kwargs
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            mmdata = {}
            if image_inputs is not None:
                mmdata["image"] = image_inputs
            if video_inputs is not None:
                mmdata["video"] = video_inputs
            inputs = [{"prompt": text, "multi_modal_data": mmdata}]
            generated = self.model.generate(inputs, self.generate_kwargs)
            response = generated[0].outputs[0].text

        if self.post_process:
            response = self.post_process_func(response)

        return response
