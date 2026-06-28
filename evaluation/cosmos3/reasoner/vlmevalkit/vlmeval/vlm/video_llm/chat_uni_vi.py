import logging
import os

import numpy as np
import torch
from PIL import Image

from vlmeval.smp import listinstr
from ..base import BaseModel


def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=64,
    image_resolution=224,
    video_framerate=1,
    s=None,
    e=None,
):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    # T x 3 x H x W
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        from decord import VideoReader, cpu
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = torch.stack(
            [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images]
        )
        slice_len = patch_images.shape[0]

        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            video[:slice_len, ...] = patch_images

        return patch_images, slice_len
    else:
        print('video path: {} error.'.format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return torch.from_numpy(video), video_mask


class Chatunivi(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True
    # sample 1 fps (maximum 64 frames) from the video

    def __init__(self, model_path='Chat-UniVi/Chat-UniVi', **kwargs):
        assert model_path is not None
        try:
            from ChatUniVi.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical('Please install Chat-UniVi from https://github.com/PKU-YuanGroup/Chat-UniVi.git.')
            raise err

        model_name = 'ChatUniVi'
        tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
        self.tokenizer = tokenizer
        self.model = model
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        self.processor = image_processor
        self.context_len = context_len
        self.kwargs = kwargs
        self.fps = 1
        self.resolution = 224
        if 'v1.5' in model_path:
            self.resolution = 336

    def get_model_output(self, model, video_processor, tokenizer, video, qs):
        from ChatUniVi.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                         DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_TOKEN,
                                         IMAGE_TOKEN_INDEX, MAX_IMAGE_LENGTH)
        from ChatUniVi.conversation import SeparatorStyle, conv_templates
        from ChatUniVi.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

        mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
        mm_use_im_patch_token = getattr(model.config, 'mm_use_im_patch_token', True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if model.config.config['use_cluster']:
            for n, m in model.named_modules():
                m = m.to(dtype=torch.bfloat16)

        video_frames, slice_len = _get_rawvideo_dec(
            video, video_processor, max_frames=MAX_IMAGE_LENGTH,
            image_resolution=self.resolution, video_framerate=self.fps
        )

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
        if type(qs) is dict and 'user' in qs:
            qs['user'] = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs['user']
        else:
            qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

        conv = conv_templates['v1'].copy()
        if type(qs) is dict and 'system' in qs:
            conv.system = qs['system']
        if type(qs) is dict and 'user' in qs:
            conv.append_message(conv.roles[0], qs['user'])
        else:
            conv.append_message(conv.roles[0], qs)
        if type(qs) is dict and 'assistant' in qs:
            conv.append_message(conv.roles[1], qs['assistant'])
        else:
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt().strip('</s>')

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video_frames.half().cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs

    def generate_inner(self, message, dataset=None):
        if listinstr(['MLVU', 'MVBench'], dataset):
            question, video = self.message_to_promptvideo_withrole(message, dataset)
        else:
            question, video = self.message_to_promptvideo(message)
        response = self.get_model_output(self.model, self.processor, self.tokenizer, video, question)
        return response
