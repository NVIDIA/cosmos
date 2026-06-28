import ast
import json
import os

import decord
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from vlmeval.smp import LMUDataRoot, get_cache_path, load
from .video_base import VideoBaseDataset


class DSRBench(VideoBaseDataset):

    MD5 = ''
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""
    LMUData_root = LMUDataRoot()

    DATASET_URL = {
        'DSRBench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/DSRBench.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'DSRBench': '9273aeb004e8ca196df5a5e826bdf97d',
    }

    def __init__(self, dataset, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['DSRBench']

    def _task_category(self):
        return [
            'abs_dis',
            'abs_dir',
            'abs_ori',
            'abs_spd',
            'abs_spd_comp',
            'abs_dir_pred',
            'rel_dis',
            'rel_dir',
            'rel_ori',
            'rel_spd',
            'rel_spd_comp',
            'rel_dir_pred',
            'non_temp',
        ]

    def download_dsrbench(self, repo_id):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.dsrbench_extracted'
        raw_data_dir = os.path.join(cache_path, "raw_data")

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(raw_data_dir, SENTINEL_NAME))):
            pass
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth):
                import zipfile

                base_dir = pth
                zip_files = [
                    os.path.join(base_dir, f) for f in os.listdir(base_dir)
                    if f.endswith('.zip')
                ]
                zip_files.sort()

                for zip_file in tqdm(zip_files, desc='Unpacking Origin Data...'):
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            rel = os.path.normpath(info.filename).lstrip('/\\')
                            dst = os.path.join(pth, rel)

                            absp = os.path.abspath(pth)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text='done')
                print('DSR-Bench data extracted to current directory with original layout.')

            download_dir = snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',
                allow_patterns=["raw_data/dsr-bench.zip"],
            )

            raw_data_dir = os.path.join(download_dir, "raw_data")
            unzip_hf_zip(raw_data_dir)

        dataset_path = os.path.join(raw_data_dir, "dsr-bench")
        return dataset_path

    def prepare_dataset(self, dataset_name: str):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        # DSR-Bench did not provide the original data; EASI hosted a copy on its behalf.
        dataset_path = self.download_dsrbench(repo_id='lmms-lab-si/EASI-Leaderboard-Data')
        self.dataset_path = dataset_path

        variant_data_file = os.path.join(self.LMUData_root, f"{dataset_name}.tsv")

        return dict(data_file=variant_data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = os.path.join(self.data_root, video_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()

        indices = []

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            frame_paths = self.frame_paths(video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_path, len(indices))

        missing = [
            (idx, pth) for idx, pth in zip(indices, frame_paths)
            if not os.path.exists(pth)
        ]

        if missing and not video_llm:
            for frame_idx, pth in missing:
                try:
                    frame_data = vid[frame_idx].asnumpy()
                    Image.fromarray(frame_data).save(pth)
                except Exception as e:
                    error_msg = f"Error saving frame {frame_idx} from {video_path}: {str(e)}"
                    print(error_msg)

                    raise ValueError(error_msg) from e

        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        pre_prompt = self.FRAMES_TMPL_NOSUB
        question = line['question']
        options = line['options']

        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = ast.literal_eval(options)

        question += '\n' + '\n'.join(options)
        prompt = f'Question: {question}\nAnswer: '

        message = []
        message.append(dict(type='text', value=pre_prompt))
        if video_llm:
            message.append(dict(type='video', value=os.path.join(self.data_root, line['video'])))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='task_type',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'DSRBench')
        )
