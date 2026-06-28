import logging
import re
import string

import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor, set_seed

from vlmeval.dataset import DATASET_TYPE
from ..base import BaseModel

GTHINKER_SYS_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. "
    "In the reasoning process enclosed within <think> </think>, each specific visual cue is enclosed within "
    "<vcues_*>...</vcues_*>, where * indicates the index of the specific cue. Before concluding the final answer, "
    "pause for a quick consistency check: verify whether the visual cues support the reasoning "
    "and whether each step logically follows from what is seen. If correct, conclude the answer; "
    "otherwise, revise the visual cues and reasoning, then conclude."
)


class Valley3Chat(BaseModel):
    """
    A multimodal chat model interface for the Valley2 architecture. (https://arxiv.org/abs/2501.05901)

    Attributes:
        - seed (int): Random seed used for reproducibility. Default is 42.
        - torch_dtype (torch.dtype): Data type for model weights (e.g., torch.bfloat16). Default is bfloat16.
        - model_path (str): Hugging Face model path or local checkpoint directory for loading
            the Valley2-DPO model. Default is 'bytedance-research/Valley2-DPO'.
        - max_new_tokens (int): Maximum number of tokens to be generated during inference. Default is 2048.
        - min_pixels (int): Minimum number of image pixels allowed after preprocessing. Default is 1280*28*28.
        - max_pixels (int): Maximum number of image pixels allowed after preprocessing. Default is 16384*28*28.
        - use_custom_prompt (bool): Whether to enable custom prompts tailored to specific benchmarks. These prompts
            have been manually optimized for individual datasets.
        - use_custom_pixels (bool): Whether to enable benchmark-specific custom image pixel limits during
            preprocessing. Some datasets benefit from lower or higher resolution constraints. Default is True.
    """

    def __init__(
        self,
        model_path='bytedance-research/Valley3',
        max_new_tokens: int = 2048,
        seed=42,
        torch_dtype=torch.bfloat16,
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=True,
        use_custom_pixels_limit=True,
        use_gthinker_thinking=False,
        **kwargs
    ):
        set_seed(seed)
        self.torch_dtype = torch_dtype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f"Start loading valley model from {model_path}")
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=self.torch_dtype, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            anyres=self.model.config.anyres,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            trust_remote_code=True
        )

        self._use_custom_prompt = use_custom_prompt
        self._use_custom_pixels_limit = use_custom_pixels_limit
        self._use_gthinker_thinking = use_gthinker_thinking
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.screen_parse = True

        kwargs_default = dict(do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.0) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def use_custom_prompt(self, dataset):
        return self._use_custom_prompt

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_yorn_prompt_gthinker(self, line, dataset=None):
        prompt = line['question']
        prompt += (
            '\nAnswer this question and the answer enclosed by '
            '<answer> </answer> should be a single word or phrase.'
        )

        return prompt

    def build_multi_choice_prompt(self, line, dataset=None):
        prompt = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt = hint + '\n' + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f'\n{key}. {item}'

        if len(options):
            prompt += "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_multi_choice_prompt_gthinker(self, line, dataset=None):
        prompt = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt = hint + '\n' + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f'\n{key}. {item}'

        if len(options):
            prompt += (
                "\nAnswer the multiple-choice question. "
                "The answer enclosed by <answer> </answer> should be "
                "'$LETTER' (without quotes) where LETTER is one of the options. "
            )

        return prompt

    def build_mmvet_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += "\nProvide a step-by-step solution to the problem carefully.\nPlease think step by step."
        return prompt

    def build_mmvet_prompt_gthinker(self, line, dataset=None):
        prompt = line['question']
        prompt += "\nProvide a step-by-step solution to the problem carefully.\nPlease think step by step."
        return prompt

    def build_math_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += (
            "\nProvide a step-by-step solution to the problem, "
            "and conclude with 'the answer is' followed by the final solution."
        )
        return prompt

    def build_mathvista_prompt_gthinker(self, line, dataset=None):
        prompt = line["question"]
        type_ = line["question_type"]
        if type_ == "multi_choice":
            prompt = prompt + (
                "\nAnswer the multiple-choice question. The answer enclosed by <answer> </answer> "
                "should be '$LETTER' (without quotes) where LETTER is one of the options."
            )
        elif type_ == "free_form":
            prompt = prompt + (
                "\nAnswer this question and the answer enclosed by <answer> </answer> "
                "should be single word or phrase as indicated by the hint."
            )
        else:
            raise AssertionError
        # return msgs
        return prompt

    def build_mathvision_prompt_gthinker(self, line, dataset=None):

        prompt = line["question"]
        hint = prompt.split("\n")[0]
        if hint == (
            "Hint: Please answer the question and provide the correct option letter,"
            " e.g., A, B, C, D, at the end."
        ):
            prompt = prompt.replace(hint, "") + (
                "\nAnswer the multiple-choice question. The answer enclosed by <answer> </answer> "
                "should be '$LETTER' (without quotes) where LETTER is one of the options."
            )
            prompt = prompt.replace("(A)", "A.").replace("(B)", "B.").replace("(C)", "C.")
            prompt = prompt.replace("(D)", "D.").replace("(E)", "E.")
        elif hint == 'Hint: Please answer the question and provide the final answer at the end.':
            prompt = prompt.replace(hint, "") + (
                "\nAnswer this question and the answer enclosed by <answer> </answer> should be single word or phrase."
            )
        else:
            raise AssertionError
        return prompt

    def build_ocr_cmmmu_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += "\nPlease try to answer the question with short words or phrases if possible."
        return prompt

    def build_ocrbench_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += "\nPlease think step by step."
        return prompt

    def build_ocrbench_prompt_gthinker(self, line, dataset=None):
        prompt = line["question"]

        if line["category"] == "Handwritten Mathematical Expression Recognition":
            prompt += (
                "\nAnswer this question withou solving the math problem and the answer enclosed by <answer> </answer>"
                " should be the OCR result without missing any character."
            )
        else:
            prompt += (
                "\nAnswer this question and the answer enclosed by <answer> </answer> "
                "should be the OCR result without missing any character."
            )
        return prompt

    def build_m3cot_prompt(self, line, dataset=None, use_cot=False):
        prompt = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt = hint + '\n' + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f'\n{key}. {item}'
        prompt += (
            "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by "
            "the '$LETTER' (without quotes) where LETTER is one of the options."
        )
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        # ========== thinking ========== #
        if self._use_gthinker_thinking:
            if any(dataset.startswith(prefix) for prefix in (
                'MMStar', 'AI2D_TEST', 'AI2D_TEST_NO_MASK', 'MMMU', 'MMBench', 'M3CoT', 'MME-RealWorld'
            )):
                prompt = self.build_multi_choice_prompt_gthinker(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('HallusionBench',)):
                prompt = self.build_yorn_prompt_gthinker(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('MathVista_MINI',)):
                prompt = self.build_mathvista_prompt_gthinker(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('MathVision',)):
                prompt = self.build_mathvision_prompt_gthinker(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('MMVet',)):
                prompt = self.build_mmvet_prompt_gthinker(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('OCRBench',)):
                prompt = self.build_ocrbench_prompt_gthinker(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('ScreenSpot',)):
                prompt = self.build_screen_prompt_gthinker(line, dataset)
            elif DATASET_TYPE(dataset) == 'Y/N':
                prompt = self.build_yorn_prompt_gthinker(line, dataset)
            elif DATASET_TYPE(dataset) == 'MCQ':
                prompt = self.build_multi_choice_prompt_gthinker(line, dataset)
            else:
                prompt = line['question']

        # ========== no_thinking ==========
        else:
            if dataset == 'MMVet':
                prompt = self.build_mmvet_prompt(line, dataset)
            elif dataset == 'OCRBench':
                prompt = self.build_ocrbench_prompt(line, dataset)
            elif dataset == 'M3CoT_Test':
                prompt = self.build_m3cot_prompt(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('TextVQA', 'ChartQA', 'InfoVQA', 'DocVQA', 'CMMMU')):
                prompt = self.build_ocr_cmmmu_prompt(line, dataset)
            elif any(dataset.startswith(prefix) for prefix in ('MathVista', 'MathVerse', 'MathVision')):
                prompt = self.build_math_prompt(line, dataset)
            elif DATASET_TYPE(dataset) == 'Y/N':
                prompt = self.build_yorn_prompt(line, dataset)
            elif DATASET_TYPE(dataset) == 'MCQ':
                prompt = self.build_multi_choice_prompt(line, dataset)
            else:
                prompt = line['question']

        # add image to message
        message = []
        if isinstance(tgt_path, list):
            message.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            message.extend([dict(type='image', value=tgt_path)])

        # add text to message
        message.append(dict(type='text', value=prompt))
        return message

    def post_process(self, generation_text, dataset=None):
        # print(f">>> RAW RESPONSE:\n{generation_text}")
        if dataset.startswith("MMVet"):
            generation_text += "(Please only output the final score.)"
            return generation_text

        if dataset.startswith("OCRBench"):
            return generation_text

        # Extract answer content
        pattern = r"(?:<answer>)([\s\S]*?)(?:</answer>)"
        match = re.search(pattern, generation_text)
        if match:
            generation_text = match.group(1)

        if any(dataset.startswith(prefix) for prefix in ('TextVQA', 'ChartQA', 'InfoVQA', 'DocVQA', 'CMMMU')):
            generation_text = generation_text.strip()
            if generation_text.endswith('.'):
                generation_text = generation_text[:-1]

        return generation_text

    def get_pixels_limit(self, dataset=None):
        if self._use_custom_pixels_limit and \
                any(dataset.startswith(prefix) for prefix in ['OCRBench', 'HallusionBench', 'MMMU']):
            min_pixels = 100 * 28 * 28
            max_pixels = 1280 * 28 * 28
        else:
            min_pixels = self.min_pixels
            max_pixels = self.max_pixels
        return min_pixels, max_pixels

    def generate_inner(self, message, dataset=None):
        # Construct messages and images
        # print(f">>> ORIGIN_MSG: {message}")
        messages = []
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])

        if self._use_gthinker_thinking:
            gthinker_system_prompt = None

            if any(dataset.startswith(prefix) for prefix in ('MMBench', 'MMMU', 'AI2D')):
                gthinker_system_prompt = GTHINKER_SYS_PROMPT
            elif any(dataset.startswith(prefix) for prefix in ('MMStar', 'MathVista', 'HallusionBench')):
                gthinker_system_prompt = \
                    GTHINKER_SYS_PROMPT.replace("<think>", "<reason>").replace("</think>", "</reason>")
            else:
                gthinker_system_prompt = GTHINKER_SYS_PROMPT

            if gthinker_system_prompt is not None:
                messages.append({"role": 'system', "content": gthinker_system_prompt})

        messages.append({"role": 'user', "content": text})

        min_pixels, max_pixels = self.get_pixels_limit(dataset)
        # print(f">>> min_pixels: {min_pixels}, max_pixels: {max_pixels}")
        # print(f">>> MSG: {messages}")
        # print(f">>> IMAGES: {images}")
        # print(f">>> ENABLE_THINKING: {self._use_gthinker_thinking}")
        data_dict = self.processor(
            {
                "conversations": messages,
                "images": images
            },
            min_pixels=min_pixels // len(images),
            max_pixels=max_pixels // len(images),
            enable_thinking=self._use_gthinker_thinking
        )

        # Inference
        with torch.inference_mode():
            self.model.to(dtype=self.torch_dtype, device=self.device)

            if type(data_dict["images"][0]) is list:
                images = [
                    [item.to(dtype=self.torch_dtype, device=self.device) for item in img]
                    for img in data_dict["images"]
                ]
            else:
                images = [img.to(dtype=self.torch_dtype, device=self.device) for img in data_dict["images"]]

            # print(f'pixel_values.shape: {data_dict["pixel_values"].shape}')

            output_ids = self.model.generate(
                input_ids=data_dict["input_ids"].to(self.device),
                images=images,
                image_sizes=data_dict["image_sizes"],
                pixel_values=data_dict["pixel_values"].to(dtype=self.torch_dtype, device=self.device),
                image_grid_thw=data_dict["image_grid_thw"].to(self.device),
                do_sample=self.kwargs["do_sample"],
                max_new_tokens=self.kwargs["max_new_tokens"],
                repetition_penalty=self.kwargs["repetition_penalty"],
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_token_len = data_dict["input_ids"].shape[1]

        # sequences = output_ids.sequences.clone()
        # sequences[sequences < 0] = 0
        # full_seq = self.processor.batch_decode(sequences)[0]
        # print(f">>> FULL_SEQ: {full_seq}")

        generation_text = self.processor.batch_decode(output_ids.sequences[:, input_token_len:])[0]
        generation_text = generation_text.replace("<|im_end|>", "")

        # Postprocess
        generation_text = self.post_process(generation_text, dataset)

        # print(f">>> POST_PROCESS: {generation_text}")
        return generation_text
