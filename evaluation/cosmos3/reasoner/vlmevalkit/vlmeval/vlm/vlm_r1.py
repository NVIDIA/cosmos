from __future__ import annotations
import logging
import os
import re
import warnings

import torch

from ..smp import get_gpu_memory
from .base import BaseModel
from .qwen2_vl.model import ensure_image_url, ensure_video_url
from .qwen2_vl.prompt import Qwen2VLPromptMixin


class VLMR1Chat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=4096,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        save_raw_output: bool = False,
        output_dir: str = "./outputs",
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            use_cache=True
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.save_raw_output = save_raw_output
        self.output_dir = output_dir
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if "2.5" in model_path:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        print(f"now testing.....{self.model_path}")
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.model = MODEL_CLS.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        torch.cuda.empty_cache()

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []

        post_prompt = '  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'

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
            elif s["type"] == "video":
                item = {"type": "video", "video": ensure_video_url(s["value"])}
                if self.fps is not None:
                    item["fps"] = self.fps
                elif self.nframe is not None:
                    import cv2

                    video = cv2.VideoCapture(s["value"])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = (
                            frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        )
                        print(f"use {new_frame_count} for {s['value']}")
                        item["nframes"] = new_frame_count
                    else:
                        item["nframes"] = self.nframe
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"] + post_prompt}

            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)

        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        def extract_answer_content(output_str):
            # Try to find the content within <answer> tags, if can not find, return None
            answer_pattern = r"<answer>\s*(.*?)\s*<\/answer>"
            match = re.search(answer_pattern, output_str, re.DOTALL)

            if match:
                return match.group(1).strip()
            return output_str

        def replace_last_dot(input_string):
            if input_string.endswith("."):
                return input_string[:-1]
            else:
                return input_string

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        from termcolor import colored

        print(colored(f"messages: === {messages}", "red"))
        print(colored(f"generate_kwargs: === {self.generate_kwargs}", "blue"))
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )

        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=text, images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        raw_output = response  # save raw output
        if self.post_process:
            resp = response.split("\\boxed{")[-1]
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

        response = extract_answer_content(response)
        response = replace_last_dot(response)

        if self.save_raw_output:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(
                self.output_dir, f"{self.model_path.split('/')[-1]}_{dataset}.jsonl"
            )
            if message[0]['type'] == 'image':
                id = message[0]['value'].rsplit('/')[-1].split('.')[0]
            else:
                id = None
            import jsonlines
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write({"id": id, "response": raw_output})

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response
