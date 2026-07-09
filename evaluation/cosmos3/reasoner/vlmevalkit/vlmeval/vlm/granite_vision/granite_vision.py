import string
import warnings

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string
from ..base import BaseModel

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    from transformers import AutoProcessor

flash_attn_flag = False
try:
    import flash_attn  # noqa: F401
    flash_attn_flag = True
except ImportError:
    pass


class GraniteVision3(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self, model_path="ibm-granite/granite-vision-3.3-2b", use_vllm=False, **kwargs
    ):
        # assert not use_vllm "vLLM is not yet supported for evaluations in VLMEvalKit"
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        attn_impl = "flash_attention_2" if flash_attn_flag else "eager"
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl
        )

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(do_sample=False, max_new_tokens=2048)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def output_process(self, answer, dataset):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "<|assistant|>" in answer:
            answer = answer.split("<|assistant|>")[1].strip("\n .")
        elif "<|start_of_role|>assistant<|end_of_role|>" in answer:
            answer = answer.split("<|start_of_role|>assistant<|end_of_role|>")[1].strip(
                "\n ."
            )

        if "<|end_of_text|>" in answer:
            answer = answer.split("<|end_of_text|>")[0].strip("\n ")
        if "answer" in answer.lower():
            answer = answer.lower().split("answer")[-1].strip(" :.-\n")
        if dataset in [
            "ChartQA_TEST",
            "DocVQA_VAL",
            "DocVQA_TEST",
            "InfoVQA_VAL",
            "InfoVQA_TEST",
            "OCRVQA_TEST",
            "OCRVQA_TESTCORE",
            "TextVQA_VAL"
        ]:
            answer = answer.strip(".")
        if "ChartMuseum" in dataset:
            answer = f"<answer>{answer}</answer>"
        return answer.strip("\n")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        if dataset in ["OCRBench", "COCO_VAL", "ChartQA_TEST", "CharXiv_descriptive_val", "ChartMimic_v1_direct",
                       "ChartMimic_v2_direct", "ChartMimic_v2_customized",]:
            return True
        return False

    def get_pre_post_prompt(self, dataset, chineese=False):
        pre_post_prompt = {
            "OCRBench": (
                "",
                "\nReply with only one word or a short phrase or a full address.",
            ),
            "COCO_VAL": ("", "\nReply with one short sentence."),
            "ChartQA_TEST": ("", "\nAnswer the question with a single word."),
            "CharXiv_descriptive_val": ("", "\nAnswer the question with a single word or short phrase."),
            "ChartMimic_v1_direct": ("", "\nAnswer using code only. strating with ```python and ending with ```"),
            "ChartMimic_v2_direct": ("", "\nAnswer using code only. strating with ```python and ending with ```"),
            "ChartMimic_v2_customized": ("", "\nAnswer using code only. strating with ```python and ending with ```"),
        }
        pre_post_prompt_cn = {}

        return (
            pre_post_prompt.get(dataset, ("", ""))
            if not chineese
            else pre_post_prompt_cn.get(dataset, ("", ""))
        )

    def build_promt_mcq(self, line):
        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)
        if DATASET_TYPE(dataset) == "MCQ":
            prompt = self.build_promt_mcq(line)
        else:
            prompt = line["question"]
        pre_promt, post_prompt = self.get_pre_post_prompt(
            dataset, chineese=cn_string(prompt)
        )
        prompt = pre_promt + prompt + post_prompt
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))

        return message

    def resize_to_max_dim(self, image: Image.Image, max_dim: int = 768) -> Image.Image:
        """Resize image so the longer side is exactly `max_dim` pixels."""
        w, h = image.size
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        # print(f"Resized image from {(w,h)} to {new_size}")
        return image.resize(new_size, Image.LANCZOS)

    def generate_inner(self, message, dataset=None):
        content, images = [], []
        img_count = 0
        for msg in message:
            if not msg["type"] == "text":
                img_count += 1
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            else:
                content.append({"type": "image"})
                img = Image.open(msg["value"]).convert("RGB")
                if img_count > 2:
                    img = self.resize_to_max_dim(img)
                images.append(img)
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device, self.model.dtype)
        with torch.no_grad():
            output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        answer = self.output_process(answer, dataset)
        return answer
