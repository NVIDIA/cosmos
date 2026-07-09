from __future__ import annotations
import logging
import re
import warnings

import torch

from ..dataset import DATASET_TYPE
from ..smp import get_gpu_memory
from .base import BaseModel
from .qwen2_vl.model import ensure_image_url, ensure_video_url
from .qwen2_vl.prompt import Qwen2VLPromptMixin

SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format."""   # noqa: E501


class TreeVGR(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=1024,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
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
            use_cache=True,
            do_sample=False,
        )
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self.verbose = verbose
        self.post_process = post_process
        self.output_dir = output_dir
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        MODEL_CLS = Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path)

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

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet'}:  # MMVet VQA has it's own prompt
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if 'treebench' in dataset.lower():
            return self._build_treebench_prompt(line, dataset)
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_treebench_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        tgt_path = self.dump_image(line, dataset)

        prompt = line['question']
        if not line["category"] == "Perception/OCR":
            prompt += "\nOptions:\n" + line["multi-choice options"]
        prompt += "\nAnswer with the option's letter directly."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for YORN dataset:"""
        YORN_PROMPT = ' Please answer yes or no.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []

        post_prompt = ''
        if DATASET_TYPE(dataset) == "MCQ":
            post_prompt = '\nSelect the best answer to the above multiple-choice question based on the image. After the reasoning process, respond with ONLY the letter of the correct option between <answer> and </answer>.'  # noqa: E501

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
                qs = s["value"].replace("\nPlease select the correct answer from the options above.", "").replace("\nAnswer with the option's letter directly", "")     # noqa: E501
                item = {"type": "text", "text": qs + post_prompt}

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
            answer_pattern = r"<answer>(.*?)</answer>"
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

        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        text = [txt + "<think>" for txt in text]
        self.processor.tokenizer.padding_side = 'left'
        print(text[0])

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

        ans = extract_answer_content(response)
        ans = replace_last_dot(ans)
        print(response)

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
            print(f"\033[32m{ans}\033[0m")

        if 'treebench' in dataset.lower():
            return response

        return ans
