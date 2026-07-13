import glob
import os
import os.path as osp
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         load, md5)
from vlmeval.smp.file import LMUDataRoot
from ..utils import DEBUG_MESSAGE, build_judge
from ..video_base import VideoBaseDataset
from .utils import Stack, ToTorchFormatTensor

FAIL_MSG = 'Failed to obtain answer via API.'


class EgoExoBench_MCQ(VideoBaseDataset):
    MD5 = '9c0aa8da235d766d02dd7e9a19182719'
    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def __init__(self, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=False):
        super().__init__(dataset=dataset, nframe=nframe)
        self.frame_fps = 2
        self.skip_EgoExo4D = skip_EgoExo4D

    @classmethod
    def supported_datasets(cls):
        return ['EgoExoBench_MCQ']

    def prepare_dataset(self, dataset_name='EgoExoBench_MCQ', repo_id='Heleun/EgoExoBench_MCQ', video_repo_id='onlyfaces/EgoExoBench'):  # noqa: E501
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            return True
        cache_path = get_cache_path(repo_id)
        self.video_root = os.path.join(LMUDataRoot(), 'videos', 'EgoExoBench')
        os.makedirs(self.video_root, exist_ok=True)
        if not osp.exists(osp.join(self.video_root, 'processed_videos')) or not osp.exists(osp.join(self.video_root, 'processed_frames')):  # noqa: E501
            snapshot_download(
                repo_id=video_repo_id,
                repo_type='dataset',
                allow_patterns=['*.tar.gz.part*'],
                local_dir=self.video_root
            )

            def combine_and_extract(root_dir, prefix, remove_parts=True):
                parts_pattern = osp.join(root_dir, f'{prefix}.tar.gz.part*')
                combined_archive = osp.join(root_dir, f'{prefix}.tar.gz')
                if not osp.exists(combined_archive):
                    parts = sorted(glob.glob(parts_pattern))
                    with open(combined_archive, 'wb') as outfile:
                        for part in parts:
                            with open(part, 'rb') as infile:
                                shutil.copyfileobj(infile, outfile)
                shutil.unpack_archive(combined_archive, root_dir)
                if remove_parts:
                    for part in parts:
                        os.remove(part)
                    os.remove(combined_archive)

            combine_and_extract(self.video_root, 'processed_videos')
            combine_and_extract(self.video_root, 'processed_frames')

        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        # transform
        self.transform = T.Compose([
            Stack(),
            ToTorchFormatTensor()
        ])

        return dict(root=dataset_path, data_file=data_file)

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=16):
        start, end = bound if bound else (-100000, 100000)
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = (end_idx - start_idx) / num_segments
        mid_seg_size = seg_size / 2
        indices = np.arange(num_segments)
        frame_indices = start_idx + mid_seg_size + np.round(seg_size * indices)
        return frame_indices.astype(int)

    def load_into_video_and_process(self, media, mcq_idx):
        try:
            from moviepy.editor import ImageSequenceClip, VideoFileClip
        except Exception:
            raise ImportError(
                'MoviePy is not installed, please install it by running "pip install moviepy==1.0.3"'
            )
        video_root = self.video_root
        if media['type'] in ['image']:
            original_image_path = osp.join(video_root, media['image_paths'][0])
            processed_video_path = osp.join(video_root, 'processed_videos', f'{mcq_idx}.jpg')
            if not os.path.exists(processed_video_path):
                shutil.copy(original_image_path, processed_video_path)
            return dict(type='image', value=processed_video_path)
        elif media['type'] in ['frames']:
            input_images = [osp.join(video_root, im) for im in media['image_paths']]
            processed_video_path = osp.join(video_root, 'processed_videos', f'{mcq_idx}.mp4')
            media['nframes'] = len(input_images) // 2 * 2
            if not os.path.exists(processed_video_path):
                # using MoviePy to transform images into mp4
                image_files = sorted(input_images)
                image_clip = ImageSequenceClip(image_files, fps=self.frame_fps)
                image_clip.write_videofile(processed_video_path, codec='libx264')
                image_clip.close()
        elif media['type'] in ['video']:
            original_video_path = osp.join(video_root, media['video_path'])
            processed_video_path = osp.join(video_root, 'processed_videos', f'{mcq_idx}.mp4')
            if 'video_start' in media and 'video_end' in media and media['video_start'] is not None and media['video_end'] is not None:  # noqa: E501
                video_start, video_end = media['video_start'], media['video_end']
                if not os.path.exists(processed_video_path):
                    video_clip = VideoFileClip(original_video_path)
                    clip = video_clip.subclip(video_start, min(video_end, video_clip.duration))
                    clip.write_videofile(processed_video_path)
                    clip.close()
            else:
                if not os.path.exists(processed_video_path):
                    shutil.copy(original_video_path, processed_video_path)
        else:
            raise ValueError(f"Unsupported media type: {media['type']}")

        return dict(type='video', value=processed_video_path, nframes=media.get('nframes', 8))

    def save_video_into_images(self, media, mcq_idx):
        bound = None
        video_root = self.video_root

        if media['type'] in ['frames', 'image']:
            media_paths = [osp.join(video_root, im) for im in media['image_paths']]
            save_dir = osp.join(video_root, 'processed_frames', str(mcq_idx))
            os.makedirs(save_dir, exist_ok=True)
            input_images = []
            for media_path in media_paths:
                img_path = media_path.split('/')[-1]
                save_image_path = osp.join(save_dir, img_path)
                shutil.copy(media_path, save_image_path)
                input_images.append(save_image_path)
            return input_images

        if 'video_start' in media and 'video_end' in media and media['video_start'] is not None and media['video_end'] is not None:  # noqa: E501
            bound = (
                media['video_start'], media['video_end']
            )
        video_path = os.path.join(video_root, media['video_path'])

        def read_video(video_path, bound=None, num_segments=16):
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())

            images_group = list()
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            save_dir = osp.join(video_root, 'processed_frames', str(mcq_idx))

            if osp.exists(save_dir) and len(os.listdir(save_dir)) > 0:
                return None, frame_indices

            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy())
                images_group.append(img)
            torch_imgs = self.transform(images_group)
            return torch_imgs, frame_indices

        def save_video_frames(imgs, video_root, frame_indices, mcq_idx):
            save_dir = osp.join(video_root, 'processed_frames', str(mcq_idx))
            os.makedirs(save_dir, exist_ok=True)
            frame_paths = [osp.join(save_dir, f'{fidx:07d}.jpg') for fidx in frame_indices]

            flag = np.all([osp.exists(pth) for pth in frame_paths])

            if not flag:
                block_size = imgs.size(0) // len(frame_indices)
                split_tensors = torch.split(imgs, block_size)
                to_pil = transforms.ToPILImage()
                images = [to_pil(arr) for arr in split_tensors]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)

            return frame_paths

        torch_imgs, frame_indices = read_video(video_path, bound, media['nframes'])
        img_frame_paths = save_video_frames(torch_imgs, video_root, frame_indices, mcq_idx)
        return img_frame_paths

    def process_text_and_media(self, text, media_list, video_llm, mcq_idx):

        message = []
        chunks = re.split(r'(<image>|<video>)', text)
        media_index = 0
        media_list = eval(media_list)

        placeholder_count = sum(1 for chunk in chunks if chunk in ['<image>', '<video>'])
        assert placeholder_count == len(media_list), \
            f"Placeholder count {placeholder_count} does not match media list length {len(media_list)}."

        for chunk in chunks:
            if chunk in ['<image>', '<video>']:
                if video_llm:
                    media_content = self.load_into_video_and_process(media_list[media_index], f'question{mcq_idx}_video{media_index}')  # noqa: E501
                    message.append(media_content)
                else:
                    # Save the video as individual image frames for processing
                    img_frame_paths = self.save_video_into_images(media_list[media_index], f'question{mcq_idx}_video{media_index}')  # noqa: E501
                    for im in img_frame_paths:
                        message.append(dict(type='image', value=im))

                media_index += 1
            elif chunk.strip():
                message.append(dict(type='text', value=chunk.strip()))

        return message

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            mcq_idx = line
            assert line < len(self)
            line = self.data.iloc[line]
            mcq_idx = int(line['index'])
        if self.skip_EgoExo4D and 'EgoExo4D' in line['medias']:
            return None
        text = line['question'] + '\nOptions:\n' + line['options'] + '\n' + line['response_format']
        message = self.process_text_and_media(text, line['medias'], video_llm, mcq_idx)
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils import extract_characters_regex, extract_option, get_dimension_rating

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be an supported format (xlsx/json/tsv) file'

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'EgoExoBench_MCQ',
                    )
                    data.loc[idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
