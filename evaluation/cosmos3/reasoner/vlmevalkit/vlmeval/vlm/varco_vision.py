import string

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import listinstr
from .base import BaseModel


class VarcoVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="NCSOFT/VARCO-VISION-2.0-14B", **kwargs):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        assert model_path is not None, "Model path must be provided."
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(model_path)

        id_prompt = "You are VARCO-VISION, created by NC AI. "
        self.processor.chat_template = self.processor.chat_template.replace(id_prompt, "")
        self.processor.tokenizer.chat_template = self.processor.tokenizer.chat_template.replace(id_prompt, "")

        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)
        self.nframe = kwargs.get("nframe", 8)
        self.fps = 1
        self.model_path = model_path

    def set_ratio(self, n):
        config = self.model.config
        processor = self.processor
        processor.vision_aspect_ratio = config.vision_aspect_ratio = f"anyres_max_{n}"

    def set_grid(self, n, reduced=False):
        config = self.model.config
        image_processor = self.processor.image_processor
        size = min(image_processor.size.values())
        grid = []
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if reduced:
                    if i * j <= n and i != n and j != n:
                        grid.append([i * size, j * size])
                else:
                    grid.append([i * size, j * size])
        image_processor.image_grid_pinpoints = config.image_grid_pinpoints = grid

    def set_res(self, dataset):
        res_4_datasets = [
            'ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST',
            'MME-RealWorld', 'VCR_EN', 'VCR_ZH', 'OCRVQA',
            'BMMR', 'MMStar', 'HallusionBench', 'MMVet',
            'AI2D_MINI', 'AI2D_TEST', 'AI2D_TEST_NO_MASK']
        res_16_datasets = [
            'InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench',
            'HRBench4K', 'HRBench8K', 'MathVista', 'LLaVABench']
        self.set_ratio(9)
        self.set_grid(6)
        if listinstr(res_4_datasets, dataset):
            self.set_ratio(4)
            self.set_grid(4, reduced=True)
        elif listinstr(res_16_datasets, dataset):
            self.set_ratio(16)
            self.set_grid(8)

    def use_custom_prompt(self, dataset):
        if any(dataset.startswith(prefix) for prefix in
               ['MMVet', 'MathVista', 'MathVerse', 'MathVision', 'LLaVABench']):
            return True
        if DATASET_TYPE(dataset) == 'Y/N':
            return True
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset.startswith('MathVista_'):
            prompt = self.build_mathvista_prompt(line, dataset)
        elif dataset.startswith('MMMU_'):
            prompt = self.build_mmmu_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'VQA':
            prompt = self.build_vqa_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')
        message = []
        message.extend([dict(type='image', value=s) for s in tgt_path])
        message.append(dict(type='text', value=prompt))

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None):
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        hint = ''
        if 'hint' in line and not pd.isna(line['hint']):
            hint = f"{line['hint']}\n"
        elif options:
            hint = 'Make sure your answer is in the given choice list.\n'

        prompt = f"{hint}{line['question']}"
        if options:
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'\n{key}. {item}'
            prompt += f"{options_prompt}\nAnswer with the option's letter directly."
        else:
            prompt += '\nAnswer the question directly.'
        return prompt

    def build_mathvista_prompt(self, line, dataset=None):
        prompt = line['question']
        if 'Choices:' in prompt:
            for i in range(1, 7):
                prompt = prompt.replace(f'({chr(64 + i)})', f'{chr(64 + i)}.')
        else:
            prompt += '\nAnswer the question directly.'
        return prompt

    def build_mmmu_prompt(self, line, dataset=None):
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        hint = ''
        if 'hint' in line and not pd.isna(line['hint']):
            hint = f"Hint: {line['hint']}\n"

        prompt = f"{hint}Question: {line['question']}"
        if options:
            options_prompt = '\nOptions:'
            for key, item in options.items():
                options_prompt += f'\n{key}. {item}'
            prompt += f'{options_prompt}\nAnswer the preceding question.'
        else:
            prompt += ' Preserve details.'
        return prompt

    def build_vqa_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += ' Preserve details.'
        return prompt

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                with Image.open(msg["value"]) as img:
                    img = img.convert("RGB")
                images.append(img)
                image_sizes.append(img.size)
                content += f"{self.DEFAULT_IMAGE_TOKEN}\n"

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

        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            elif msg["type"] == "video":
                videos.append(msg["value"])
                visual_content += f"{self.DEFAULT_IMAGE_TOKEN}\n"

        if len(videos) > 1:
            raise ValueError("LLaVA-OneVision does not support multiple videos as input.")

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, fps=1, force_sample=self.force_sample
        )

        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, "
            f"and {len(video_frames)} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video.\n"
        )

        content = visual_content + time_instruction + text_content
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": content}, {"type": "video"}],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(videos=video_frames, text=prompt, return_tensors="pt").to('cuda', torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        import numpy as np
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()

        if avg_fps == 0:
            raise ValueError(f"Video '{video_path}' has an average FPS of 0, which is invalid.")
        if fps <= 0:
            raise ValueError("FPS argument must be greater than 0.")

        effective_fps = round(avg_fps / fps)
        frame_idx = list(range(0, total_frame_num, effective_fps))
        frame_time = [i / avg_fps for i in frame_idx]

        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / avg_fps for i in frame_idx]

        frame_time_str = ", ".join([f"{t:.2f}s" for t in frame_time])
        video_frames = vr.get_batch(frame_idx).asnumpy()
        video_time = total_frame_num / avg_fps

        return video_frames, frame_time_str, video_time

    def generate_inner(self, message, dataset=None):
        self.set_res(dataset)
        if DATASET_MODALITY(dataset) == "VIDEO" and "megabench" not in dataset.lower():
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)
