# flake8: noqa
import ast
import copy as cp
import os
import os.path as osp
import random
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import (dump, extract_json_objects, get_cache_path, get_file_extension,
                         get_intermediate_file_path, load, md5)
from vlmeval.utils import track_progress_rich
from .utils import build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

camera_caption_prompts = [
    "Summary of the view shot, camera movement and changes in shooting angles in the sequence of video frames.",
    "Describe the camera movements in these frames.",
    "What are the camera angles and movements throughout the video?",
    "Summarize the camera actions and perspectives.",
    "Describe any camera zooms, pans, or angle changes.",
    "What camera movements are present in these frames?",
    "Describe the camera's movements, including pans, zooms, and angle changes in these frames.",
    "Summarize the camera actions and changes in shooting angles during the video.",
    "Provide a detailed description of the camera's movements and perspectives.",
    "Describe the camera's actions and how it follows the main subject.",
    "What are the camera movements and angle shifts in these frames?",
    "Given these equally spaced frames, provide a comprehensive description of the camera's movements, including any pans, zooms, and changes in shooting angles.",
    "Describe the camera's movements and angles in detail, explaining how it follows the main subject and changes perspectives.",
    "Based on these frames, provide a detailed description of the camera's actions, including any pans, zooms, angle shifts, and how it captures the scene.",
    "Using these frames, describe the camera's movements, including its tracking of the main subject, changes in angles, and any zooms or pans.",
    "Provide an elaborate description of the camera movements, covering pans, zooms, and changes in shooting angles as shown in these frames."
]

detailed_caption_prompts = [
    "The images are given containing equally spaced video frames. Please imagine the video based on the sequence of frames, and provide a faithfully detailed description of this video in more than three sentences.",
    "You are given a sequence of equally spaced video frames. Based on these frames, imagine the full video and provide a detailed description of what is happening in more than three sentences.",
    "The following set contains equally spaced video frames. Imagine the video from which these frames were taken and describe it in detail in at least three sentences.",
    "Below are equally spaced frames from a video. Use these frames to visualize the entire video and provide a detailed description in more than three sentences.",
    "A sequence of equally spaced video frames is presented. Please imagine the full video and write a faithfully detailed description of the events in more than three sentences.",
    "The images provided include equally spaced frames from a video. Based on these frames, imagine the video and describe it comprehensively in at least three sentences.",
    "You are given equally spaced frames from a video. Use these frames to envision the entire video and provide a detailed description of the events in more than three sentences.",
    "The sequence includes equally spaced frames from a video. Imagine the full video based on these frames and provide a detailed description in more than three sentences.",
    "The provided images contain equally spaced frames from a video. Visualize the video from these frames and describe it in detail in more than three sentences.",
    "Here are equally spaced frames from a video. Based on these frames, imagine the video and provide a detailed, faithful description of it in more than three sentences.",
    "The set of images includes equally spaced video frames. Please imagine the video these frames come from and describe it comprehensively in at least three sentences.",
    "Describe the video based on these frames in a few sentences.",
    "What is happening in the video shown in these frames?",
    "Explain the video using these frames.",
    "Imagine the video from these frames and describe it in detail in a few sentences.",
    "Based on these frames, provide a narrative of the video in more than three sentences.",
    "Describe the events in the video shown by these frames in at least three sentences.",
    "Visualize the video from these frames and explain what is happening in more than three sentences.",
    "Describe the sequence of events in the video depicted by these frames in a detailed manner.",
    "Given these equally spaced frames, imagine the entire video and provide a detailed description of the events, including the setting, characters, and actions, in more than three sentences.",
    "Visualize the video based on these frames and write a comprehensive description of what happens, describing the beginning, middle, and end in at least three sentences.",
    "Using these frames as a reference, imagine the full video and provide a thorough description of the plot, including key details and actions, in more than three sentences.",
    "Based on the sequence of these frames, describe the entire video in detail, mentioning important aspects such as the context, movements, and transitions in more than three sentences.",
    "Imagine the video that corresponds to these frames and provide an elaborate description, covering the storyline, visual elements, and any notable features in at least three sentences."
]

background_caption_prompts = [
    "The images are given containing equally spaced video frames.Summary of the background. This should also include the objects, location, weather, and time.",
    "Describe the background, including objects, location, weather, and time.",
    "Summarize the background setting of the video based on these frames.",
    "What is the environment like in these frames?",
    "Describe the location and weather in these frames.",
    "What background objects and settings are visible in these frames?",
    "Summarize the background of the video, including details about the location, objects, weather, and time.",
    "Describe the environment shown in these frames, covering objects, location, weather, and time.",
    "Provide a detailed background description based on these frames, mentioning objects, location, weather, and time.",
    "Explain the setting of the video, focusing on the background elements like objects, location, weather, and time.",
    "Describe the overall environment in these frames, including details about objects, location, weather, and time.",
    "Given these equally spaced frames, provide a comprehensive background description, covering the objects, location, weather, and time of day.",
    "Imagine the environment from these frames and write a detailed description of the background, including objects, location, weather, and time.",
    "Based on these frames, describe the setting in detail, mentioning the objects present, the specific location, the weather conditions, and the time of day.",
    "Provide an elaborate background description based on these frames, covering all aspects of the environment such as objects, location, weather, and time.",
    "Using these frames as a reference, give a thorough description of the background, including details about the objects, location, weather, and time."
]

short_caption_prompts = [
    "Write a one-sentence summary of the video.",
    "Summarize the video in one concise sentence.",
    "Provide a brief description of the video in one sentence.",
    "Describe the main action in the video in one sentence.",
    "What is the video about? Summarize it in one sentence.",
    "In one sentence, summarize the key visual elements of the video.",
    "Provide a one-sentence summary that captures the main subject and action in the video.",
    "Write a concise one-sentence description that encapsulates the essence of the video.",
    "Describe the main theme or action of the video in a single sentence.",
    "What is happening in the video? Provide a one-sentence summary.",
    "Given these frames, write a brief one-sentence summary that captures the essence of the video's visual and artistic style.",
    "Summarize the key visual and thematic elements of the video in one concise sentence.",
    "Provide a one-sentence description that highlights the main subject and action depicted in the video.",
    "In one sentence, describe the primary visual and artistic elements of the video.",
    "Write a concise one-sentence summary that encapsulates the main action and visual style of the video.",
    "Briefly one-sentence Summary of the visual, Photographic and artistic style."
]

main_object_caption_prompts = [
    "Description of the main subject actions or status sequence. This suggests including the main subjects (person, object, animal, or none) and their attributes, their action, their position, and movements during the video frames.",
    "Describe the main subject's actions and movements.",
    "What is the main object doing in these frames?",
    "Summarize the primary subject's attributes and actions.",
    "Describe the main subject's position and movements.",
    "What actions does the main object take in these frames?",
    "Describe the main subject, including their attributes and movements throughout the video.",
    "Provide a detailed description of the main object's actions and positions in these frames.",
    "Summarize the main subject's actions, attributes, and movements during the video.",
    "Describe the primary subject's movements and actions in detail.",
    "What are the main object's attributes and how do they move throughout the video?",
    "Given these equally spaced frames, provide a comprehensive description of the main subject, including their attributes, actions, positions, and movements.",
    "Describe the primary object or subject in the video, detailing their attributes, actions, positions, and movements in these frames.",
    "Based on these frames, provide a detailed description of the main subject, including their attributes, actions, positions, and how they navigate through the video.",
    "Using these frames, describe the main subject's attributes, actions, and movements, detailing their positions and how they interact with the environment.",
    "Provide an elaborate description of the main object in the video, covering their attributes, actions, positions, and movements as shown in these frames."
]


class VDC(VideoBaseDataset):

    MD5 = ''

    TYPE = 'Video-VQA'

    def __init__(self, dataset='VDC', pack=False, nframe=0, fps=-1, subset='all', limit=1.0):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

        if subset == 'all':
            pass
        elif subset == 'breakpoint':
            self.data = self.data[self.data['caption_type'] == 'breakpoint']
        elif subset == 'short':
            self.data = self.data[self.data['caption_type'] == 'short']
        elif subset == 'detailed':
            self.data = self.data[self.data['caption_type'] == 'detailed']
        elif subset == 'background':
            self.data = self.data[self.data['caption_type'] == 'background']
        elif subset == 'main_object':
            self.data = self.data[self.data['caption_type'] == 'main_object']
        else:
            raise ValueError(f'Invalid subset: {subset}')

        if limit <= 1.0 and limit > 0:
            sample_num = int(limit * len(self.data))
            self.data = self.data.iloc[:sample_num]
        elif limit > 1.0 and limit < len(self.data):
            self.data = self.data.iloc[:limit]
        else:
            raise ValueError(f'Invalid limit: {limit}')

    @classmethod
    def supported_datasets(cls):
        return ['VDC']

    def prepare_dataset(self, dataset_name='VDC', repo_id='Enxin/VLMEval-VDC'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video']:
                if not osp.exists(osp.join(pth, 'videos', video_pth)):
                    return False
            return True

        if os.path.exists(repo_id):
            dataset_path = repo_id
        else:
            cache_path = get_cache_path(repo_id)
            if cache_path is not None and check_integrity(cache_path):
                dataset_path = cache_path
            else:
                cache_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
                if not glob(osp.join(cache_path, "video")):
                    tar_files = glob(osp.join(cache_path, "**/*.tar*"), recursive=True)

                    def untar_video_data(tar_file, cache_dir):
                        import tarfile
                        with tarfile.open(tar_file, "r") as tar_ref:
                            tar_ref.extractall(cache_dir)
                            print(f"Extracted all files from {tar_file} to {cache_dir}")

                    def concat_tar_parts(tar_parts, output_tar):
                        with open(output_tar, "wb") as out_tar:
                            from tqdm import tqdm
                            for part in tqdm(sorted(tar_parts)):
                                with open(part, "rb") as part_file:
                                    out_tar.write(part_file.read())
                        print(f"Concatenated parts {tar_parts} into {output_tar}")

                    tar_parts_dict = {}

                    # Group tar parts together
                    for tar_file in tar_files:
                        base_name = tar_file.split(".tar")[0]
                        if base_name not in tar_parts_dict:
                            tar_parts_dict[base_name] = []
                        tar_parts_dict[base_name].append(tar_file)

                    # Concatenate and untar split parts
                    for base_name, parts in tar_parts_dict.items():
                        print(f"Extracting following tar files: {parts}")
                        output_tar = base_name + ".tar"
                        if not osp.exists(output_tar):
                            print('Start concatenating tar files')

                            concat_tar_parts(parts, output_tar)
                            print('Finish concatenating tar files')

                        if not osp.exists(osp.join(cache_path, osp.basename(base_name))):
                            untar_video_data(output_tar, cache_path)
        dataset_path = cache_path
        self.video_path = osp.join(dataset_path, 'videos/')
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=osp.join(dataset_path, 'video'))

    def build_prompt_pack(self, line):
        if isinstance(line, int):
            assert line < len(self)
            video = self.videos[line]
        elif isinstance(line, pd.Series):
            video = line['video']
        elif isinstance(line, str):
            video = line

        frames = self.save_video_frames(video)
        message = []
        for im in frames:
            message.append(dict(type='image', value=im))

        if self.data['caption_type'] == 'short':
            prompt = random.choice(short_caption_prompts)
        elif self.data['caption_type'] == 'detailed':
            prompt = random.choice(detailed_caption_prompts)
        elif self.data['caption_type'] == 'background':
            prompt = random.choice(background_caption_prompts)
        elif self.data['caption_type'] == 'main_object':
            prompt = random.choice(main_object_caption_prompts)
        else:
            prompt = random.choice(camera_caption_prompts)
        message.append(dict(type='text', value=prompt, role='user'))
        return message

    def build_prompt_nopack(self, line, video_llm):
        """Build prompt for a single line without packing"""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        if line['caption_type'] == 'short':
            prompt = random.choice(short_caption_prompts)
        elif line['caption_type'] == 'detailed':
            prompt = random.choice(detailed_caption_prompts)
        elif line['caption_type'] == 'background':
            prompt = random.choice(background_caption_prompts)
        elif line['caption_type'] == 'main_object':
            prompt = random.choice(main_object_caption_prompts)
        else:
            prompt = random.choice(camera_caption_prompts)

        if video_llm:
            video_path = os.path.join(self.video_path, line['video'])
            return [
                dict(type='video', value=video_path),
                dict(type='text', value=prompt)
            ]
        else:
            frames = self.save_video_frames(os.path.splitext(line['video'])[0])
            message = []
            for im in frames:
                message.append(dict(type='image', value=im))
            message.append(dict(type='text', value=prompt))
            return message

    def build_prompt(self, line, video_llm):
        if self.pack and not video_llm:
            return self.build_prompt_pack(line)
        else:
            return self.build_prompt_nopack(line, video_llm)

    @staticmethod
    def remove_side_quote(s, syms=[',', '"', "'"]):
        if np.all([x in syms for x in s]):
            return ''
        while s[0] in syms:
            s = s[1:]
        while s[-1] in syms:
            s = s[:-1]
        return s

    @staticmethod
    def robust_json_load(s):
        try:
            jsons = list(extract_json_objects(s))
            assert len(jsons) == 1
            return jsons[0]
        except Exception:
            if '{' in s and s.find('{') == s.rfind('{'):
                sub_str = s[s.find('{') + 1:].strip()
                lines = sub_str.split('\n')
                res = {}
                for l in lines:
                    l = l.strip()
                    if ': ' in l:
                        key = l.split(': ')[0].strip()
                        val = l.split(': ')[1].strip()
                        key = VDC.remove_side_quote(key)
                        val = VDC.remove_side_quote(val)
                        if len(key) and len(val):
                            res[key] = val
                return res
            return None

    def load_pack_answers(self, data_raw):
        vstats = defaultdict(lambda: 0)
        data = defaultdict(lambda: {})

        for k in data_raw:
            ans = data_raw[k].strip()
            if FAIL_MSG in ans:
                vstats['GEN_FAIL'] += 1
                continue
            res = self.robust_json_load(ans)
            if res is not None:
                data[k] = res
                vstats['PARSE_OK'] += 1
            else:
                vstats['PARSE_FAIL'] += 1

        # return data
        meta = cp.deepcopy(self.data)
        lt = len(meta)
        prediction = []
        for i in range(lt):
            line = meta.iloc[i]
            vid = line['video']
            idx = str(line['index'])
            prediction.append(data[vid][idx] if idx in data[vid] else None)
        meta['prediction'] = prediction
        vstats['VALIDQ'] = len([x for x in prediction if x is not None])
        vstats['INVALIDQ'] = len([x for x in prediction if x is None])
        return meta, vstats

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vdc import (SYSTEM_CAL_SCORE_PROMPT, SYSTEM_GENER_PRED_PROMPT,
                                get_dimension_rating, prepare_response_prompt,
                                prepare_score_prompt)

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'
        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 4)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)

        response_file = get_intermediate_file_path(eval_file, f'_{judge}_response', 'pkl')
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, f'_{judge}_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, f'_{judge}_score')

        model = build_judge(**judge_kwargs)

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)

            expanded_data = []
            for idx, row in data.iterrows():
                try:
                    questions = ast.literal_eval(row['question']) if isinstance(row['question'], str) else row['question']
                    for q_dict in questions:
                        new_row = row.copy()
                        new_row['question'] = q_dict['question']
                        new_row['answer'] = q_dict['answer']
                        expanded_data.append(new_row)
                except Exception as e:
                    print(f"Error parsing questions for row {idx}")
                    print(f"Error message: {str(e)}")
                    continue

            expanded_df = pd.DataFrame(expanded_data).reset_index(drop=True)

            data_un = expanded_df[~expanded_df['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]
            lt = len(data_un)

            response_prompts = [prepare_response_prompt(data_un.iloc[i]) for i in range(lt)]
            indices = [data_un.iloc[i]['index'] for i in range(lt)]

            model.system_prompt = SYSTEM_GENER_PRED_PROMPT
            if len(response_prompts):
                print(f"Processing {len(response_prompts)} valid prompts out of {lt} total items")
                _ = track_progress_rich(
                    model.generate,
                    response_prompts,
                    keys=indices,
                    save=response_file,
                    nproc=nproc,
                    chunksize=nproc
                )

            pred_map = load(response_file)
            data_un['pred_response'] = [pred_map[idx] for idx in data_un['index']]
            score_prompts = [prepare_score_prompt(data_un.iloc[i]) for i in range(lt)]
            model.system_prompt = SYSTEM_CAL_SCORE_PROMPT
            if len(score_prompts):
                _ = track_progress_rich(
                    model.generate,
                    score_prompts,
                    keys=indices,
                    save=tmp_file,
                    nproc=nproc,
                    chunksize=nproc
                )

            score_map = load(tmp_file)
            data['score'] = [score_map[idx] for idx in data['index']]

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
