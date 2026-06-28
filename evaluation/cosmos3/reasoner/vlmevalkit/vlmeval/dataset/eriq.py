import ast
import os

from huggingface_hub import snapshot_download

from vlmeval.smp import get_cache_path, load, modelscope_flag_set, toliststr
from .image_mcq import ImageMCQDataset


class ERIQBench(ImageMCQDataset):
    """
    ERIQ.

    Reference:
      Unified Embodied VLM Reasoning with Robotic Action via Autoregressive Discretized Pre-training
      https://arxiv.org/abs/2512.24125
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        'ERIQ': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/ERIQ.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'ERIQ': 'aa21c81ee33e6016b4099d8d1b2e0d4f',
    }

    def _task_category(self):
        return [
            'QA_ACTION_UNDERSTANDING',
            'QA_DUALVIEW_MATCHING',
            'QA_FINE_GRAINED_PLAN',
            'QA_HUMAN_INTENTION',
            'QA_HUMAN_INTERACTION',
            'QA_MISTAKE_CLASSIFY',
            'QA_MISTAKE_EXISTENCE',
            'QA_MISTAKE_RECOVERY',
            'QA_RELATIVE_POS_GROUNDING',
            'QA_SCENE_UNDERSTANDING',
            'QA_SUBTASK_PLANNING',
            'QA_SUCCESS_DETECTION',
            'QA_TASK_GROUNDING',
            'QA_TASK_PROGRESS',
            'QA_TRAJ_UNDERSTANDING'
        ]

    def prepare_tsv(self, url, file_md5=None, repo_id='KineMind/ERIQ'):
        data = super().prepare_tsv(url, file_md5)

        SENTINEL_NAME = '.eriq_extracted'
        cache_path = get_cache_path(repo_id)

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            sentinel_path = os.path.join(dataset_path, SENTINEL_NAME)
            _write_sentinel(sentinel_path, text='done')
            print('ERIQ data downloaded successfully.')

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                image_root = os.path.join(dataset_path, 'images')

                return os.path.normpath(os.path.join(image_root, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['image_path'] = data['image_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        # ERIQ QA prompt format use in paper
        prompt = line['question_raw']

        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split('<image>')
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type='text', value=part))
            if (i != len(parts) - 1) and (i < len(images)):
                segs.append(dict(type='image', value=images[i]))

        return [s for s in segs if s['value']]

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
            dataset_name=getattr(self, 'dataset_name', 'ERIQ'),
        )
