import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from .base import BaseModel


class CambrianS(BaseModel):
    """
    Cambrian-S: Towards Spatial Supersensing in Video

    Requirements:
    1. Clone the official Cambrian-S repository:

       git clone https://github.com/cambrian-mllm/cambrian-s.git

    2. Install it in development mode (recommended to use the same Python env
       as VLMEvalKit):

       cd cambrian-s
       pip install -e .

       or, if you use uv:

       cd cambrian-s
       uv pip install -e .

    Note: This will automatically install all dependencies including torch, transformers, etc.
    """

    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str = "nyu-visionx/Cambrian-S-7B",
        conv_template: str = "qwen_2",
        use_cache: bool = False,
        # video-related params
        video_max_frames: int = 32,
        video_fps: int = 1,
        video_force_sample: bool = False,
        # anyres / multi-scale image processing
        miv_token_len: int = 64,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
    ):
        assert model_path is not None
        try:
            from cambrian.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                            DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
            from cambrian.conversation import conv_templates
            from cambrian.mm_utils import (expand2square, get_model_name_from_path, process_images,
                                           tokenizer_image_token)
            from cambrian.model.builder import load_pretrained_model
        except Exception as e:
            logging.critical(
                "Failed to import Cambrian-S modules. Please ensure you have:\n"
                "  git clone https://github.com/cambrian-mllm/cambrian-s.git\n"
                "  cd cambrian-s && pip install -e .\n"
                f"Original error: {e}"
            )
            raise e

        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.get_model_name_from_path = get_model_name_from_path
        self.expand2square = expand2square
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.conv_templates = conv_templates

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = self.get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path, None, model_name, device_map=self.device
        )

        cfg = self.model.config
        cfg.video_max_frames = video_max_frames
        cfg.video_fps = video_fps
        cfg.video_force_sample = video_force_sample
        cfg.miv_token_len = miv_token_len
        cfg.si_token_len = si_token_len
        cfg.image_aspect_ratio = image_aspect_ratio
        cfg.anyres_max_subimages = anyres_max_subimages

        cfg.mm_use_im_start_end = getattr(cfg, "mm_use_im_start_end", True)

        # for video dataset override nframe or fps
        self.nframe = video_max_frames
        self.fps = video_fps

        self.use_cache = use_cache
        self.model_config = cfg
        self.conv_template = conv_template

        self.model.eval()

    def _get_video_sampling_cfg(self):
        cfg = self.model_config

        # 1) self.fps > cfg.video_fps
        fps_override = getattr(self, "fps", None)
        if fps_override is not None and fps_override > 0:
            target_fps = float(fps_override)
        else:
            target_fps = float(getattr(cfg, "video_fps", 1.0))

        # 2) self.nframe > cfg.video_max_frames
        nframes_override = getattr(self, "nframe", None)

        if nframes_override is not None and nframes_override > 0:
            max_frames = int(nframes_override)
        else:
            max_frames = int(getattr(cfg, "video_max_frames", 0))

        return target_fps, max_frames

    def _process_video_with_decord(self, video_file: str, num_threads: int = 0):
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=num_threads)

        total_frame_num = len(vr)
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 30.0

        target_fps, max_frames = self._get_video_sampling_cfg()
        avg_fps = round(fps / max(1.0, target_fps))
        avg_fps = max(1, int(avg_fps))

        video_time = total_frame_num / fps
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
        frame_time = [i / fps for i in frame_idx]

        if max_frames > 0:
            if len(frame_idx) > max_frames or self.model_config.video_force_sample:
                uniform_sampled_frames = np.linspace(
                    0, total_frame_num - 1, max_frames, dtype=int
                )
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i / fps for i in frame_idx]

        video = vr.get_batch(frame_idx).asnumpy()
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])

        vr.seek(0)
        num_frames_to_sample = len(frame_idx)
        return video, video_time, frame_time_str, num_frames_to_sample

    def _process_videos(self, videos: List[str]):
        processor_aux_list = self.image_processor
        if not isinstance(processor_aux_list, (list, tuple)):
            processor_aux_list = [processor_aux_list]

        new_videos_aux_list = []
        video_sizes = []
        last_meta = None

        for video_path in videos:
            video_np, video_time, frame_time, num_frames_to_sample = self._process_video_with_decord(video_path)
            last_meta = (video_time, frame_time, num_frames_to_sample)

            T, H, W, _ = video_np.shape
            video_sizes.append((W, H, T))  # W, H, T

            frames_pil = [Image.fromarray(video_np[i], mode="RGB") for i in range(T)]

            video_aux_list = []
            for processor_aux in processor_aux_list:
                mean_tuple = tuple(int(x * 255) for x in processor_aux.image_mean)
                frames_square = [self.expand2square(im, mean_tuple) for im in frames_pil]
                video_aux = processor_aux.preprocess(frames_square, return_tensors="pt")["pixel_values"]
                # (T, C, H', W')
                video_aux_list.append(video_aux)

            new_videos_aux_list.append(video_aux_list)

        new_videos_aux_list = [list(batch_video_aux) for batch_video_aux in zip(*new_videos_aux_list)]
        new_videos_aux_list = [torch.stack(video_aux) for video_aux in new_videos_aux_list]  # (B, T, C, H', W')

        return new_videos_aux_list, video_sizes, last_meta

    def prepare_input(
        self, message: Union[str, List[Dict]]
    ) -> Tuple[str, Optional[List[torch.Tensor]], Optional[List]]:
        """
        Returns:
            qs: str, prompt text with image tokens inserted (contexts -> qs)
            visual_tensors: List[Tensor] or None
            visual_sizes: anyres/video size metadata passed to the model
        """
        if isinstance(message, str):
            return message, None, None

        if not isinstance(message, list):
            raise TypeError("message must be str or List[Dict]")

        video_items = []
        image_items = []
        text_items = []

        for m in message:
            if "type" in m and "value" in m:
                mtype, mval = m["type"], m["value"]
            else:
                if "text" in m:
                    mtype, mval = "text", m["text"]
                elif "image" in m:
                    mtype, mval = "image", m["image"]
                elif "video" in m:
                    mtype, mval = "video", m["video"]
                else:
                    raise TypeError("Each message element must contain type+value or text/image/video field")

            if mtype == "video":
                video_items.append(mval)
            elif mtype == "image":
                image_items.append(mval)
            elif mtype == "text":
                text_items.append(mval)
            else:
                raise TypeError("type must be one of {'text','image','video'}")

        num_videos = len(video_items)
        num_images = len(image_items)

        # 1: single video + text (video appears before text)
        if num_videos == 1 and num_images == 0:
            vpath = video_items[0]
            if not isinstance(vpath, str):
                raise TypeError("video currently only supports a file path string")

            # Concatenate all text segments into a single context string
            contexts: str = ""
            for m in message:
                ttype = m.get("type") or ("text" if "text" in m else None)
                if ttype == "text":
                    contexts += str(m.get("value") or m.get("text") or "")

            visuals = [vpath]

            visual_tensors, visual_sizes, _ = self._process_videos(visuals)

            qs = contexts
            if self.model_config.mm_use_im_start_end:
                qs = (
                    self.DEFAULT_IM_START_TOKEN
                    + self.DEFAULT_IMAGE_TOKEN
                    + self.DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                assert len(visual_tensors) == 1, "This should not happen."
                qs = self.DEFAULT_IMAGE_TOKEN * len(visual_tensors) + "\n" + qs

            return qs, visual_tensors, visual_sizes

        # 2: images only (single or multiple; image and text can interleave)
        if num_videos == 0 and num_images >= 1:
            visuals: List[Union[str, Image.Image]] = []
            contexts_list: List[str] = []

            for m in message:
                if "type" in m and "value" in m:
                    mtype, mval = m["type"], m["value"]
                else:
                    if "text" in m:
                        mtype, mval = "text", m["text"]
                    elif "image" in m:
                        mtype, mval = "image", m["image"]
                    else:
                        raise TypeError("Each message element must contain type+value or text/image field")

                if mtype == "text":
                    sval = str(mval)
                    if sval != "":
                        contexts_list.append(sval)
                elif mtype == "image":
                    visuals.append(mval)
                    contexts_list.append("")  # empty string placeholder; will be turned into DEFAULT_IMAGE_TOKEN later
                else:
                    raise TypeError("In this mode, 'video' is not supported; please ensure message does not mix videos")

            pil_visuals: List[Image.Image] = []
            for idx, v in enumerate(visuals):
                if isinstance(v, Image.Image):
                    pil_visuals.append(v)
                elif isinstance(v, str):
                    p = v.strip().strip('"').strip("'")
                    try:
                        im = Image.open(p).convert("RGB")
                    except Exception as e:
                        logging.error(f"[CambrianS] Failed to open image idx={idx}, path={p}, err={repr(e)}")
                        raise
                    pil_visuals.append(im)
                else:
                    raise TypeError(f"Unsupported image type: {type(v)}")

            if len(pil_visuals) == 1:
                visual_tensors, visual_sizes = self.process_images(
                    pil_visuals, self.image_processor, self.model_config
                )
            else:
                visual_tensors, visual_sizes = self.process_images(
                    pil_visuals, self.image_processor, self.model_config, use_pad=True
                )

            real_qs = ""
            for i, sub_qs in enumerate(contexts_list):
                if sub_qs:
                    real_qs += sub_qs
                else:
                    if i + 1 < len(contexts_list):
                        nxt = contexts_list[i + 1]
                        if nxt != "" and len(nxt) > 0 and nxt[0] == "\n":
                            real_qs += self.DEFAULT_IMAGE_TOKEN
                        else:
                            real_qs += self.DEFAULT_IMAGE_TOKEN + "\n"
                    else:
                        real_qs += self.DEFAULT_IMAGE_TOKEN + "\n"

            qs = real_qs
            return qs, visual_tensors, visual_sizes

        # 3. multiple videos or mixed image + video
        raise NotImplementedError(
            f"Currently only two input patterns are supported:\n"
            f"1) Single video + text (video first), detected num_videos={num_videos}\n"
            f"2) Images only (single/multi image + text), detected num_images={num_images}\n"
            f"Other combinations (multiple videos or mixed image+video) are not supported, "
            f"consistent with the official implementation."
        )

    def generate_inner(self, message, dataset: Optional[str] = None) -> str:
        qs, visual_tensors, visual_sizes = self.prepare_input(message)

        conv = self.conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # tokenize
        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            use_cache=self.use_cache,
            do_sample=False,
        )

        with torch.inference_mode():
            images_arg = None
            if visual_tensors is not None:
                images_arg = [vt.to(self.device) for vt in visual_tensors]

            output_ids = self.model.generate(
                inputs=input_ids,
                images=images_arg,
                image_sizes=visual_sizes,
                **gen_kwargs,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def chat_inner(self, message: Union[str, List[Dict]], dataset: Optional[str] = None) -> str:
        return self.generate_inner(message, dataset)
