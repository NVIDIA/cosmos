import copy as cp
import os
import os.path as osp
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import (dump, extract_json_objects, get_cache_path, get_file_extension,
                         get_intermediate_file_path, load, md5)
from ..utils import track_progress_rich
from .utils import build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


class MovieChat1k(VideoBaseDataset):

    MD5 = '7c0aa7e10de1cddb37af42b4abc9a2dd'

    TYPE = 'Video-VQA'

    def __init__(self, dataset='MovieChat1k', pack=False, nframe=0, fps=-1, subset='all', limit=1.0):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

        if subset == 'all':
            pass
        elif subset == 'global':
            self.data = self.data[self.data['mode'] == 'global']
        elif subset == 'breakpoint':
            self.data = self.data[self.data['mode'] == 'breakpoint']
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
        return ['MovieChat1k']

    def prepare_dataset(self, dataset_name='MovieChat1k', repo_id='Enxin/VLMEval-MovieChat1k'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video']:
                if not osp.exists(osp.join(pth, video_pth)):
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

                        if not osp.exists(osp.join(cache_path, 'videos')):
                            untar_video_data(output_tar, cache_path)
        dataset_path = cache_path
        self.video_path = osp.join(dataset_path, 'videos/')
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=osp.join(dataset_path, 'videos'))

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

        message.append(dict(type='text', value=line['question'], role='user'))
        return message

    def build_prompt_nopack(self, line, video_llm):
        """Build prompt for a single line without packing"""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        if video_llm:
            video_path = os.path.join(self.video_path, line['video'])
            return [
                dict(type='video', value=video_path),
                dict(type='text', value=line['question'])
            ]
        else:
            frames = self.save_video_frames(line['video'])
            message = []
            for im in frames:
                message.append(dict(type='image', value=im))
            message.append(dict(type='text', value=line['question']))
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
                for line in lines:
                    line = line.strip()
                    if ': ' in line:
                        key = line.split(': ')[0].strip()
                        val = line.split(': ')[1].strip()
                        key = MovieChat1k.remove_side_quote(key)
                        val = MovieChat1k.remove_side_quote(val)
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
        from .utils.moviechat1k import get_dimension_rating, prepare_score_prompt

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501
        judge = judge_kwargs.setdefault('model', 'chatgpt-0125')
        assert judge in ['chatgpt-0125'], f'Invalid judge model for MovieChat1k: {judge}'
        nproc = judge_kwargs.pop('nproc', 4)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)

        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, f'_{judge}_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, f'_{judge}_score')

        model = build_judge(**judge_kwargs)

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(eval_file)
            data_un = data[~data['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]
            lt = len(data_un)
            prompts = [prepare_score_prompt(data_un.iloc[i]) for i in range(lt)]
            indices = [data_un.iloc[i]['index'] for i in range(lt)]
            if len(prompts):
                _ = track_progress_rich(
                    model.generate,
                    prompts,
                    keys=indices,
                    save=tmp_file,
                    nproc=nproc,
                    chunksize=nproc
                )
            score_map = load(tmp_file)
            data['score'] = [score_map[idx] if idx in score_map else -1 for idx in data['index']]
            rejected = [x for x in score_map.values() if FAIL_MSG in x]
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(score_map)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as 0 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
