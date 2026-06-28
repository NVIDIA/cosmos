import copy
import logging
import os.path as osp
import re
import string
import warnings

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr, splitlen
from .base import BaseModel


class _MoondreamPromptMixin:
    CUSTOM_DATASETS = {
        "ChartQA_TEST",
        "TextVQA_VAL",
        "DocVQA_VAL",
        "POPE",
        "RealWorldQA",
        "TallyQA",
        "CountbenchQA",
        "MMVet",
    }

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False

        if listinstr(["MMMU"], dataset):
            return False
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        if dataset in self.CUSTOM_DATASETS:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]

        if dataset == "ChartQA_TEST":
            prompt = (
                "Analyze the chart carefully, consider both visual features and data values,"
                " and provide a precise answer without any additional explanation or formatting. "
                + question
            )
        elif dataset == "TextVQA_VAL":
            prompt = (
                "Read the text in the image and provide a brief lowercase answer. "
                "Respond 'unanswerable' only if there is no plausible answer. "
                + question
            )
        elif dataset == "DocVQA_VAL":
            prompt = question + " The answer should be a short text span taken verbatim from the document."
        elif dataset == "POPE":
            prompt = f"{question}\nAnswer yes or no."
        elif dataset == "RealWorldQA":
            prompt = question
        elif dataset in ["TallyQA", "CountbenchQA"]:
            prompt = (
                "Look at the image carefully and count the objects. "
                "Answer with just a number, without any additional text. "
                + question
            )
        elif dataset == "MMVet":
            prompt = question + "\nAnswer the question directly. "
        elif DATASET_TYPE(dataset) == "MCQ":
            options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
            options_prompt = ""
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"

            hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
            prompt = f"Hint: {hint}\n" if hint is not None else ""
            prompt += f"{question}\n"
            prompt += (
                f"{options_prompt}\nAnswer with the option’s letter from the given choices directly. "
                if len(options)
                else "Answer the question directly. "
            )
        else:
            raise NotImplementedError

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])
        return message


class Moondream1(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="vikhyatk/moondream1", **kwargs):
        try:
            from transformers import AutoModelForCausalLM
            from transformers import CodeGenTokenizerFast as Tokenizer
        except Exception as e:
            logging.critical(
                "Please install Transformers version 4.36.2 by running: 'pip install transformers==4.36.2', "
                "please intall torchvision>=0.16."
            )
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.tokenizer = Tokenizer.from_pretrained(model_path)

        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, img = self.message_to_promptimg(message)
        enc_image = self.model.encode_image(Image.open(img))

        prompt_wtmpl = f"<image>\n\nQuestion: {prompt}\n\nAnswer:"
        answer = self.model.generate(
            enc_image,
            prompt_wtmpl,
            eos_text="<END>",
            tokenizer=self.tokenizer,
            **self.kwargs,
        )[0]
        cleaned_answer = re.sub("<$", "", re.sub("END$", "", answer)).strip()
        return cleaned_answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(["MMMU"], dataset):
            return False
        if DATASET_TYPE(dataset) == "MCQ" or dataset in [
            "MMVet",
        ]:
            return True

        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        if dataset == "MMVet":
            prompt = question + "\nAnswer the question directly. "
        elif DATASET_TYPE(dataset) == "MCQ":
            options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
            options_prompt = ""
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"

            hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
            prompt = f"Hint: {hint}\n" if hint is not None else ""
            prompt += f"{question}\n"
            prompt += (
                f"{options_prompt}\nAnswer with the option’s letter from the given choices directly. "
                if len(options)
                else "Answer the question directly. "
            )
        else:
            raise NotImplementedError

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])
        return message


class Moondream2(_MoondreamPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="vikhyatk/moondream2", revision="2025-01-09", **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            logging.critical(
                """Please install Transformers version 4.44 by running: "pip install transformers==4.44.0",
            please intall torchvision>=0.16."""
            )
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
            revision=revision,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, img = self.message_to_promptimg(message)
        enc_image = self.model.encode_image(Image.open(img))
        print(f"prompt for {dataset} -> ", prompt)

        answer = self.model.query(enc_image, prompt)["answer"]
        cleaned_answer = answer.strip()

        return cleaned_answer


class Moondream3(_MoondreamPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(
        self,
        model_path="moondream/moondream3-preview",
        torch_dtype="auto",
        device_map=None,
        compile_model=True,
        reasoning=True,
        stream=False,
        settings=None,
        **kwargs,
    ):
        try:
            from transformers import AutoModelForCausalLM
        except Exception as e:
            logging.critical(
                'Please install Transformers version >= 4.48.0 by running: "pip install transformers>=4.48.0"; '
                "please install torchvision>=0.16."
            )
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        resolved_dtype = self._resolve_dtype(torch_dtype)
        if device_map is None:
            device_map = {"": "cuda"} if torch.cuda.is_available() else {"": "cpu"}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=resolved_dtype,
            device_map=device_map,
        )

        if compile_model and hasattr(self.model, "compile"):
            try:
                self.model.compile()
            except Exception as err:
                warnings.warn(f"Moondream3 compilation failed due to {err}. Continuing without compilation.")

        default_settings = dict(max_tokens=512)
        if settings is not None:
            default_settings.update(settings)
        if kwargs:
            warnings.warn(
                "Following kwargs will be merged into Moondream3 query settings "
                f"(pass a `settings` dict to avoid this implicit behavior): {kwargs}"
            )
            default_settings.update(kwargs)
        if "max_new_tokens" in default_settings and "max_tokens" not in default_settings:
            default_settings["max_tokens"] = default_settings.pop("max_new_tokens")

        self.query_settings = default_settings
        self.reasoning = reasoning
        self.stream = stream

        torch.cuda.empty_cache()

    def _resolve_dtype(self, torch_dtype):
        if isinstance(torch_dtype, str):
            if torch_dtype == "auto":
                return self._resolve_dtype_auto()
            if hasattr(torch, torch_dtype):
                return getattr(torch, torch_dtype)
            raise ValueError(f"Unsupported torch dtype string: {torch_dtype}")
        if torch_dtype is None:
            return self._resolve_dtype_auto()
        return torch_dtype

    def _resolve_dtype_auto(self):
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability(0)
                if major >= 8:
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
        return torch.float32

    def generate_inner(self, message, dataset=None):
        prompt, img = self.message_to_promptimg(message)

        if img is not None:
            with Image.open(img) as pil_img:
                result = self._run_query(pil_img, prompt)
        else:
            result = self._run_query(None, prompt)

        answer = self._extract_answer(result)
        return answer.strip()

    def _run_query(self, image, prompt):
        settings = copy.deepcopy(self.query_settings) if self.query_settings else None
        return self.model.query(
            image=image,
            question=prompt,
            reasoning=self.reasoning,
            stream=self.stream,
            settings=settings,
        )

    def _extract_answer(self, payload):
        if isinstance(payload, dict):
            answer_field = payload.get("answer", payload.get("response", ""))
            if self.stream and answer_field is not None and not isinstance(answer_field, str):
                chunks = []
                for chunk in answer_field:
                    if chunk is None:
                        continue
                    chunks.append(str(chunk))
                return "".join(chunks)
            return answer_field if answer_field is not None else ""
        return str(payload)
