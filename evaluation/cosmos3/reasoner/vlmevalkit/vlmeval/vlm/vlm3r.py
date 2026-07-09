"""
This scirpt is modified from the official VLM-3R implementation (https://github.com/VITA-Group/VLM-3R),
providing a unified Vision-Language Model (VLM) framework integrating 3D reconstructive instruction tuning
for deep spatial understanding from monocular video.
"""
import logging
import math
from typing import Optional

import decord
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig

from .base import BaseModel


class VLM3R(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        pretrained: str = 'Journey9ni/vlm-3r-llava-qwen2-lora',
        model_base: str = 'lmms-lab/LLaVA-NeXT-Video-7B-Qwen2',
        model_name: Optional[str] = None,
        conv_template: str = 'qwen_1_5',
        device_map: str = 'auto',
        device: str = "cuda:0",
        use_cache: bool = True,
        delay_load: bool = False,
        tie_weights: bool = True,
        overwrite: bool = True,
        # hyper-parameters
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = False,
        max_new_tokens: int = 16,
        # video params
        mm_resampler_type: str = 'spatial_pool',
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_out_channels: int = 1024,
        mm_spatial_pool_mode: str = "bilinear",
        mm_newline_position: str = "grid",
        mm_pooling_position: str = "after",
        video_max_frames: int = 32,
        video_fps: int = 1,
        video_force_sample: bool = False,
        **kwargs
    ):
        super().__init__()
        try:
            from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                         DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_TOKEN,
                                         IMAGE_TOKEN_INDEX)
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                                        tokenizer_image_token)
            from llava.model.builder import load_pretrained_model

        except Exception as e:
            logging.critical(
                "Failed to import VLM3R modules. Please ensure you have:\n"
                "  git clone --recurse-submodules https://github.com/VITA-Group/VLM-3R.git\n"
                "  cd VLM-3R && pip install -e .\n"
                "  cd CUT3R && pip install -r requirements.txt\n"
                "  cd src/croco/models/curope/ && python setup.py build_ext --inplace\n"
                "  cd ../../../../..\n"
                f"Original error: {e}"
            )
            raise e

        assert pretrained is not None
        self.pretrained = pretrained
        self.overwrite = overwrite
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = get_model_name_from_path(pretrained)

        self.device_map = device_map
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        # Load constants and functions from official repo
        self.default_im_start_token = DEFAULT_IM_START_TOKEN
        self.default_im_end_token = DEFAULT_IM_END_TOKEN
        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_image_patch_token = DEFAULT_IMAGE_PATCH_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token
        self.separatorstyle = SeparatorStyle
        self.keywordsstoppingcriteria = KeywordsStoppingCriteria
        self.load_pretrained_model = load_pretrained_model

        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)

        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(video_max_frames)
        self.mm_resampler_location = mm_pooling_position
        self.mm_newline_position = mm_newline_position
        self.delay_load = delay_load

        # Currently only overwrite=True is supported.
        if self.overwrite:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = self.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_out_channels"] = self.mm_spatial_pool_out_channels
            overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
            overwrite_config["mm_pooling_position"] = self.mm_resampler_location
            overwrite_config["mm_newline_position"] = self.mm_newline_position
            overwrite_config["add_faster_video"] = False
            overwrite_config["delay_load"] = self.delay_load

            cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

            if cfg_pretrained.architectures[0] == "LlavaLlamaForCausalLM":
                # Ugly code, only used in  vicuna that needs ROPE
                if "224" in cfg_pretrained.mm_vision_tower:
                    least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            if "v1.5" in pretrained:
                # A hardcode solution here to load v1.5 model, otherwise it will use LlavaConfig from hf transformers
                from llavavid.model.language_model.llava_llama import (LlavaConfig,
                                                                       LlavaLlamaForCausalLM)
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
                cfg_pretrained = LlavaConfig.from_pretrained(pretrained)
                if overwrite_config is not None:
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                kwargs["torch_dtype"] = torch.float16
                self.model = LlavaLlamaForCausalLM.from_pretrained(
                    pretrained, low_cpu_mem_usage=True, config=cfg_pretrained,
                    device_map=self.device_map, **kwargs)
                vision_tower = self.model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model(device_map=self.device_map)
                if self.device_map != "auto":
                    vision_tower.to(device="cuda", dtype=torch.float16)
                self.image_processor = vision_tower.image_processor

                if hasattr(self.model.config, "max_sequence_length"):
                    self.max_length = self.model.config.max_sequence_length
                else:
                    self.max_length = 2048
            else:
                self.tokenizer, self.model, self.image_processor, self.max_length = \
                    self.load_pretrained_model(
                        pretrained, model_base, self.model_name,
                        device_map=self.device_map,
                        overwrite_config=overwrite_config)
        else:
            self.tokenizer, self.model, self.image_processor, self.max_length = \
                self.load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map)
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.conv_template = conv_template
        self.use_cache = use_cache

        self.model.config.video_max_frames = video_max_frames
        self.model.config.video_fps = video_fps
        self.model.config.video_force_sample = video_force_sample
        self.nframe = video_max_frames
        self.fps = video_fps
        self.model.to(device="cuda", dtype=torch.float16)

    def generate_inner(self, message, dataset=None):
        videos = []
        images = []
        question = ''
        image_sizes = []
        for s in message:
            if s['type'] == 'image':
                image_obj = Image.open(s['value']).convert("RGB")
                image = self.image_processor.preprocess(image_obj, return_tensors="pt")["pixel_values"].half().cuda()
                image_sizes.append(image_obj.size)  # (width, height)
                images.append(image[0])
            elif s['type'] == 'text':
                question += s['value']
            elif s['type'] == 'video':
                video_path = s['value']
                video_frames, _, _ = self.get_video_frames(video_path)
                video = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)

        if len(images) > 0:
            image_tensor = images
        elif len(videos) > 0:
            image_tensor = videos
        else:
            assert False, "No images or videos found in the message."

        if self.model.config.mm_use_im_start_end:
            image_placeholders = \
                (self.default_im_start_token + self.default_image_token
                 + self.default_im_end_token + "\n") * len(image_tensor)
        else:
            image_placeholders = self.default_image_token * len(image_tensor) + "\n"

        question = image_placeholders + question
        conv = self.conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.image_token_index,
            return_tensors="pt").unsqueeze(0).cuda()

        if self.tokenizer.pad_token_id is None and "qwen" in self.tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            self.tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
        stop_str = conv.sep if conv.sep_style != self.separatorstyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.keywordsstoppingcriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                attention_mask=attention_masks,
                modalities=["video" for _ in videos] if len(videos) > 0 else ["image" for _ in images],
                image_sizes=image_sizes if len(images) > 0 else None,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                num_beams=self.num_beams,
                use_cache=self.use_cache,
                stopping_criteria=[stopping_criteria]
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

    def get_video_frames(self, vid_path):

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        # Align with offical video frame sampling strategy
        indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()

        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]

        return images, indices, video_info
