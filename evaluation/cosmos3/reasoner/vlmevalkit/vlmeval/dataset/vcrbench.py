import os
import os.path as osp

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import (dump, extract_json_objects, get_cache_path, get_file_extension,
                         get_intermediate_file_path, load, modelscope_flag_set)
from vlmeval.utils import track_progress_rich
from .utils import build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


# unzip video files
def unzip_video(pth):
    source_dir = os.path.join(pth, 'v1/videos/video.zip')
    target_dir = os.path.join(pth, 'v1/videos/')

    video_folder = os.path.join(pth, 'v1/videos/video')

    if not os.path.exists(video_folder):
        os.system(f'unzip -o {source_dir} -d {target_dir}')

        print('Video file decompression completed.')
    else:
        print('The video file has been decompressed.')


class VCRBench(VideoBaseDataset):

    SYS = 'You are an AI assistant responsible for answering questions about videos.'

    FRAMES_TMPL_NOPACK = """
You will be provided with {} separate frames uniformly sampled from a video, \
the frames are provided in chronological order of the video.
Please analyze these images and provide the answer to the question about the video content.
"""

    TYPE = 'Video-VQA'

    def __init__(self, dataset='VCR-Bench', pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['VCR-Bench']

    def prepare_dataset(self, dataset_name='VCR-Bench', repo_id='VLM-Reasoning/VCR-Bench'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_video(dataset_path)
        self.video_path = dataset_path
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def build_prompt_nopack(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        if video_llm:
            question = line['question']
            prefix, video_idx_path = os.path.split(line['video_path'])
            message = [dict(type='text', value=question)]
            message.append(
                dict(
                    type='video',
                    value=os.path.join(self.video_path, os.path.join(prefix, video_idx_path))
                )
            )
            return message
        else:
            frames = self.save_video_frames(line['video'])
            sys_prompt = self.FRAMES_TMPL_NOPACK.format(len(frames))
            message = [dict(type='text', value=sys_prompt)]
            for im in frames:
                message.append(dict(type='image', value=im))
            prompt = 'Question: {}\nAnswer: '.format(line['question'])
            message.append(dict(type='text', value=prompt))
        return message

    def build_prompt(self, line, video_llm):
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
                        key = VCRBench.remove_side_quote(key)
                        val = VCRBench.remove_side_quote(val)
                        if len(key) and len(val):
                            res[key] = val
                return res
            return None

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vcrbench.cau_acc import calu_acc_main, xlsx2json
        from .utils.vcrbench.cau_total import calu_pre_recall
        from .utils.vcrbench.eval import precision, recall
        from .utils.vcrbench.prompt import (Answer_Extraction_Prompt_part1,
                                            Answer_Scoring_Prompt_part1,
                                            Precision_Evaluation_Prompt, Recall_Evaluation_Prompt,
                                            build_Extraction_prompt, build_Precision_prompt,
                                            build_Recall_prompt, build_Scoring_prompt)

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501
        judge = judge_kwargs.pop('model', 'gpt-4o-0806')
        nproc = judge_kwargs.pop('nproc', 4)

        # step1: extract answer
        print("running step 1: extracting answer")
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_extracted_answer_tmp', 'pkl')
        extracted_answer_file = get_intermediate_file_path(eval_file, f'_{judge}_extracted_answer')
        model = build_judge(system_prompt=Answer_Extraction_Prompt_part1, model=judge, **judge_kwargs)

        if not osp.exists(extracted_answer_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(eval_file)
            data_un = data[~data['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]
            lt = len(data_un)
            prompts = [build_Extraction_prompt(data_un.iloc[i]) for i in range(lt)]
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
            extracted_answer_map = load(tmp_file)
            data['extracted_answer'] = [
                extracted_answer_map[idx] if idx in extracted_answer_map else -1 for idx in data['index']
            ]
            dump(data, extracted_answer_file)

        # step2: scoring
        print("running step 2: acc scoring")
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_answer_score_tmp', 'pkl')
        answer_score_file = get_intermediate_file_path(eval_file, f'_{judge}_answer_score')
        model = build_judge(system_prompt=Answer_Scoring_Prompt_part1, model=judge, **judge_kwargs)

        if not osp.exists(answer_score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(extracted_answer_file)
            data_un = data[~data['index'].isin(res)]
            lt = len(data_un)
            prompts = [build_Scoring_prompt(data_un.iloc[i]) for i in range(lt)]
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
            answer_score_map = load(tmp_file)
            data['answer_scoring'] = [answer_score_map[idx] if idx in answer_score_map else -1 for idx in data['index']]
            dump(data, answer_score_file)

        txt_file = get_intermediate_file_path(eval_file, f'_{judge}_answer_score', 'txt')
        answer_score_json = get_intermediate_file_path(eval_file, f'_{judge}_answer_score', 'json')
        xlsx2json(answer_score_file, answer_score_json)
        calu_acc_main(answer_score_json, txt_file)

        # step3: calulate precision_score
        print("running step 3: calulate precision_score")
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_pre_score_tmp', 'pkl')
        pre_score_file = get_intermediate_file_path(eval_file, f'_{judge}_pre_score')

        model = build_judge(system_prompt=Precision_Evaluation_Prompt, model=judge, **judge_kwargs)

        if not osp.exists(pre_score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(extracted_answer_file)
            data_un = data[~data['index'].isin(res)]
            lt = len(data_un)
            prompts = [build_Precision_prompt(data_un.iloc[i]) for i in range(lt)]
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
            pre_map = load(tmp_file)  # resume
            data['precision_eval'] = [
                pre_map[idx] if idx in pre_map else -1 for idx in data['index']
            ]
            valid_indices = []
            data['Video_precision'] = data['logic_precision'] = ""
            data['background_precision'] = data['overall_precision'] = data['efficiency'] = ""  # panda
            for index, row in data.iterrows():
                try:
                    data.loc[index] = precision(row)
                    valid_indices.append(index)
                except Exception:
                    pass
            data = data.loc[valid_indices]
            dump(data, pre_score_file)

        pre_score_json = get_intermediate_file_path(eval_file, f'_{judge}_pre_score', 'json')
        xlsx2json(pre_score_file, pre_score_json)

        # step4: calulate recall_score
        print("running step 4: calulate recall_score")
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_recall_score_tmp', 'pkl')
        recall_score_file = get_intermediate_file_path(eval_file, f'_{judge}_recall_score')

        model = build_judge(system_prompt=Recall_Evaluation_Prompt, model=judge, **judge_kwargs)

        if not osp.exists(recall_score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(extracted_answer_file)
            data_un = data[~data['index'].isin(res)]
            lt = len(data_un)
            prompts = [build_Recall_prompt(data_un.iloc[i]) for i in range(lt)]
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
            recall_map = load(tmp_file)
            data['recall_eval'] = [recall_map[idx] if idx in recall_map else -1 for idx in data['index']]
            valid_indices = []
            data['Video_recall'] = data['logic_recall'] = data['background_recall'] = data['overall_recall'] = ""
            for index, row in data.iterrows():
                try:
                    data.loc[index] = recall(row)
                    valid_indices.append(index)
                except Exception:
                    pass
            data = data.loc[valid_indices]
            dump(data, recall_score_file)

        txt_file = get_intermediate_file_path(eval_file, f'_{judge}_precision_recall_score', 'txt')
        recall_score_json = get_intermediate_file_path(eval_file, f'_{judge}_recall_score', 'json')
        xlsx2json(recall_score_file, recall_score_json)
        calu_pre_recall(pre_score_json, recall_score_json, txt_file)
