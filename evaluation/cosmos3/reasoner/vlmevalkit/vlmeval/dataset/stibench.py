import ast
import os
import string

import decord
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from vlmeval.smp import LMUDataRoot, get_cache_path, load, modelscope_flag_set
from .video_base import VideoBaseDataset


class STIBench(VideoBaseDataset):
    """
    STI-Bench.

    Reference:
      STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?
      https://arxiv.org/abs/2503.23765
    """

    MD5 = ''
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    LMUData_root = LMUDataRoot()

    DATASET_URL = {
        'STI-Bench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/STIBench.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'STI-Bench': '9493f1a66374dbd7bf89992d6d1ff117',
    }

    def __init__(self, dataset, pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        subsets = ['STI-Bench']
        return subsets

    def _task_category(self):
        return [
            # Top level: Static Understanding
            'Dimensional Measurement',
            'Spatial Relation',
            '3D Video Grounding',

            # Top level: Dynamic Understanding
            'Displacement & Path Length',
            'Speed & Acceleration',
            'Ego-Centric Orientation',
            'Trajectory Description',
            'Pose Estimation',
        ]

    def download_stibench(self, repo_id='MINT-SJTU/STI-Bench'):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.stibench_extracted'

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
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
                print('STIBench data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        return dataset_path

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        dataset_path = self.download_stibench()
        self.dataset_path = dataset_path

        variant_data_file = os.path.join(self.LMUData_root, f'{dataset_name}.tsv')

        return dict(data_file=variant_data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = os.path.join(self.data_root, video_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()

        indices = []
        sample_fps = None

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            frame_paths = self.frame_paths(video_path)

            total_duration = video_nframes / video_fps if video_fps > 0 else 0.0
            sample_fps = (len(indices) / total_duration) if total_duration > 0 else None

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_path, len(indices))

            sample_fps = self.fps

        flag = np.all([os.path.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not os.path.exists(pth) and not video_llm:
                    im.save(pth)

        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
            'sample_fps': sample_fps
        }

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question']
        time_start = line['time_start']
        time_end = line['time_end']

        raw_options = line['options']

        upper_letters = list(string.ascii_uppercase)
        options_list = ast.literal_eval(raw_options)

        option_text = '\n'.join(
            f'{upper_letters[i]} {options_list[i]}'
            for i in range(len(options_list))
        )

        # following STI prompt format
        # https://github.com/MINT-SJTU/STI-Bench/blob/main/openai_test.py#L99

        question = (
            f"From {time_start} seconds to {time_end} seconds. "
            + question + "\n" + option_text
        )

        frames, _, video_info = self.save_video_frames(line['video'], video_llm)

        sample_fps = video_info.get('sample_fps')
        if sample_fps is not None:
            fps_text = f"which are sampled at about {sample_fps:.2f} FPS.\n"
        else:
            fps_text = "which are sampled at an unspecified frame rate.\n"

        prompt_text = (
            f"Answer the question below based on the frames provided, "
            f"{fps_text}"
            f"Question: {question}\n"
            f"Please output only the option you choose!"
        )

        message = []

        if video_llm:
            message.append(dict(type='video', value=os.path.join(self.data_root, line['video'])))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt_text))

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
            dataset_name=getattr(self, 'dataset_name', 'STIBench')
        )
