import logging
import sys

import numpy as np
import torchvision
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr
from ..base import BaseModel


class PLLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='ermu2001/pllava-13b', dir_root=None, **kwargs):
        sys.path.append(dir_root)
        try:
            from tasks.eval.model_utils import load_pllava
        except Exception as err:
            logging.critical(
                'Please first install requirements and set the root path to use PLLaVA. \
                Follow the instructions at https://github.com/magic-research/PLLaVA.'
            )
            raise err

        self.nframe = 16
        self.use_lora = True
        self.lora_alpha = 4
        self.pooling_shape = (16, 12, 12)
        self.RESOLUTION = 672
        self.model_path = model_path
        # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
        weight_dir = snapshot_download(model_path)
        self.model, self.processor = load_pllava(
            model_path, num_frames=self.nframe, use_lora=self.use_lora,
            weight_dir=weight_dir, lora_alpha=self.lora_alpha, pooling_shape=self.pooling_shape
        )

        #  position embedding
        self.model = self.model.to('cuda')
        self.model = self.model.eval()

    def load_video(self, video_path, num_segments=8, resolution=336):
        from decord import VideoReader, cpu
        transforms = torchvision.transforms.Resize(size=resolution)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(transforms(img))
        return images_group

    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def generate_inner(self, message, dataset=None):
        from tasks.eval.eval_utils import conv_templates
        from tasks.eval.model_utils import pllava_answer

        question, video = self.message_to_promptvideo(message)

        img_list = self.load_video(video, num_segments=self.nframe, resolution=self.RESOLUTION)

        if self.model_path == 'ermu2001/pllava-34b':  # using slightly different conversation mode for 34b model
            if listinstr(['Video-MCQ'], DATASET_TYPE(dataset)):  # MCQ dataset
                conv_mode = 'eval_mvbench_llavanext'
            else:  # VQA dataset
                conv_mode = 'eval_videoqa_llavanext'
        else:
            if listinstr(['Video-MCQ'], DATASET_TYPE(dataset)):  # MCQ dataset
                conv_mode = 'eval_mvbench'
            else:  # VQA dataset
                conv_mode = 'eval_videoqabench'

        conv = conv_templates[conv_mode].copy()
        if dataset in ['MVBench', 'MVBench_MP4']:
            conv.user_query(message[1]['value'], message[0]['value'], message[-2]['value'], is_mm=True)
            conv.assistant_response(message[-1]['value'])
        else:
            conv.user_query(question, is_mm=True)
        llm_response, conv = pllava_answer(
            conv=conv, model=self.model, processor=self.processor,
            do_sample=False, img_list=img_list, max_new_tokens=512, print_res=False
        )
        if dataset in ['MVBench', 'MVBench_MP4']:
            llm_response = '(' + ''.join(llm_response.split(message[-1]['value'])[1:])
        return llm_response
