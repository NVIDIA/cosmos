import math
import os
import os.path as osp
import random
import string

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import LMUDataRoot, get_logger, listinstr
from .base import BaseModel

logger = get_logger(__name__)


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'middle']:
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices

    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    elif 'interval' in sample:
        if num_frames == 1:
            frame_indices = [random.randint(0, vlen - 1)]
        else:
            # transform FPS
            interval = 8
            clip_length = num_frames * interval * input_fps / 30
            max_idx = max(vlen - clip_length, 0)
            start_idx = random.uniform(0, max_idx)
            end_idx = start_idx + clip_length - 1

            frame_indices = torch.linspace(start_idx, end_idx, num_frames)
            frame_indices = torch.clamp(frame_indices, 0, vlen - 1).long().tolist()
    else:
        raise ValueError
    return frame_indices


def get_frame_indices_start_end(num_frames, vlen, fps, start_time, end_time):
    start_idx = max(int(fps * start_time), 0) if start_time is not None and not math.isnan(start_time) else 0
    end_idx = min(int(fps * end_time), vlen) if end_time is not None and not math.isnan(end_time) else vlen
    clip_len = end_idx - start_idx

    acc_samples = min(num_frames, clip_len)
    # split the video into `acc_samples` intervals, and sample from each interval.
    intervals = np.linspace(start=start_idx, stop=end_idx, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    try:
        frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
    except Exception:
        frame_indices = np.random.permutation(list(range(start_idx, end_idx)))[:acc_samples]
        frame_indices.sort()
        frame_indices = list(frame_indices)

    if len(frame_indices) < num_frames:  # padded with last frame
        padded_frame_indices = [frame_indices[-1]] * num_frames
        padded_frame_indices[:len(frame_indices)] = frame_indices
        frame_indices = padded_frame_indices

    return frame_indices


def read_frames_decord(
    video_path, width=None, height=None,
    num_frames=8, sample='rand', fix_start=None,
    max_num_frames=-1, start_time=None, end_time=None
):
    import decord
    decord.bridge.set_bridge('torch')
    if video_path.lower().endswith('.webm'):
        # a workaround for webm, large/auto num_threads will cause error.
        num_threads = 2
    else:
        num_threads = 0

    if width is not None and height is not None:
        video_reader = decord.VideoReader(video_path, width=width, height=height, num_threads=num_threads)
    else:
        video_reader = decord.VideoReader(video_path, num_threads=num_threads)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    if start_time and end_time:
        frame_indices = get_frame_indices_start_end(
            num_frames, vlen, fps, start_time, end_time
        )
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps, max_num_frames=max_num_frames
        )
    frames = video_reader.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()  # (T, H, W, C), torch.uint8
    else:
        print(frames.shape)
        frames = frames.asnumpy()
    timestamp = {
        'num_frames': len(frame_indices),
        'timestamp': ', '.join([str(round(f / fps, 1)) for f in frame_indices])
    }
    return frames, timestamp


class mPLUG_Owl3(BaseModel):
    # No separate model module is required, but the dependencies must be met.
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl3/requirements.txt
    INSTALL_REQ = True
    INTERLEAVE = True
    INSTALL_REQ_TXT = 'https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl3/requirements.txt'

    def __init__(self, model_path=None, **kwargs):
        assert model_path is not None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation='sdpa',
            torch_dtype=torch.half,
            trust_remote_code=True
        )
        self.model.eval().cuda()
        self.processor = self.model.init_processor(self.tokenizer)
        if self.INSTALL_REQ:
            logger.info(
                f'Please remember to meet the requirements first\n'
                f'Here: {self.INSTALL_REQ_TXT}'
            )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if listinstr(['MVBench', 'MMVet'], dataset):
            return True
        return False

    def save_video_into_images(self, line, num_frames=16, dataset_class=None):
        video_url = {
            'video': osp.join(line['prefix'], line['video']),
            'num_frames': num_frames,
            'bound': line.get('bound', None)
        }
        if osp.isdir(video_url['video']):
            frame_paths = []
            max_frame = len(os.listdir(video_url['video']))
            fps = 3
            if video_url['bound']:
                start, end = line['start'], line['end']
            else:
                start, end = -100000, 100000
            start_idx = max(1, round(start * fps))
            end_idx = min(round(end * fps), max_frame)
            seg_size = float(end_idx - start_idx) / num_frames
            frame_indices = np.array([
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_frames)
            ])

            for frame_index in frame_indices:
                img = os.path.join(video_url['video'], f'{frame_index:05d}.jpg')
                frame_paths.append(img)

            return frame_paths

        if isinstance(video_url, dict):
            if video_url['bound']:
                start_time = line['start']
                end_time = line['end']
            else:
                start_time = None
                end_time = None
            num_frames = video_url.get('num_frames', num_frames)
            video_url = video_url['video']
        else:
            start_time = None
            end_time = None
            video_url = str(video_url)

        if not osp.exists(video_url):  # for MVBench_MP4
            video_url = osp.join(dataset_class.data_root, video_url)
        video, timestamp = read_frames_decord(
            video_url, num_frames=num_frames, sample='middle', start_time=start_time, end_time=end_time
        )

        to_pil = transforms.ToPILImage()
        frames = [to_pil(video[ti]) for ti in range(video.shape[0])]
        lmu_root = LMUDataRoot()
        frame_root = osp.join(lmu_root, 'images', dataset_class.dataset_name, 'mplug_owl3')
        frame_root = osp.join(frame_root, video_url.split('/')[-1].split('.')[0])
        os.makedirs(frame_root, exist_ok=True)
        frame_tmpl = 'frame-{}-of-{}.jpg'
        frame_paths = [osp.join(frame_root, frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]
        for im, pth in zip(frames, frame_paths):
            if not osp.exists(pth):
                im.save(pth)

        return frame_paths

    # Currently same to mPLUG_Owl2
    def build_prompt(self, line, dataset=None, num_frames=16, video_llm=False):
        if not isinstance(dataset, str):
            dataset_class = dataset
            dataset = dataset_class.dataset_name
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        if dataset_class.MODALITY == 'VIDEO':
            if listinstr(['MVBench'], dataset):
                tgt_path = self.save_video_into_images(line, num_frames, dataset_class)
            else:
                tgt_path = dataset_class.save_video_into_images(line, num_frames)
            if type(line['candidates']) is not list:
                line['candidates'] = eval(line['candidates'])
            for idx, c in enumerate(line['candidates']):
                line[chr(ord('A') + idx)] = c
        else:
            tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif listinstr(['MCQ', 'Video-MCQ'], DATASET_TYPE(dataset)):
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def preproc_image(self, fname, dataset=None):
        from PIL import Image

        image = Image.open(fname).convert('RGB')
        # resize to max_size
        max_size = 448 * 16
        if max(image.size) > max_size and not listinstr(['MVBench'], dataset):
            w, h = image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        return image

    def generate_inner(self, message, dataset=None):
        num_images = len([x for x in message if x['type'] == 'image'])
        assert num_images >= 0

        images = []
        prompt_full = ''

        for msg in message:
            if msg['type'] == 'image':
                images.append(msg['value'])
                prompt_full += '<|image|>'
            elif msg['type'] == 'text':
                prompt_full += msg['value']

        needed_messages = [
            {'role': 'user', 'content': prompt_full},
            {'role': 'assistant', 'content': ''}
        ]

        images = [self.preproc_image(fname, dataset) for fname in images]

        inputs = self.processor(needed_messages, images=images, videos=None, cut_enable=False)

        inputs.to('cuda')
        if listinstr(['MVBench'], dataset):
            inputs.update({
                'tokenizer': self.tokenizer,
                'max_new_tokens': 100,
                'decode_text': True,
                'do_sample': True,
                'top_k': 1,
            })
        else:
            inputs.update({
                'tokenizer': self.tokenizer,
                'max_new_tokens': 1024,
                'decode_text': True,
            })

        g = self.model.generate(**inputs)
        return g[0]
