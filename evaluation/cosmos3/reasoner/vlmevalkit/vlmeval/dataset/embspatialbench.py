import ast
import json
import string

from vlmeval.smp import load, toliststr
from .image_mcq import ImageMCQDataset


class EmbSpatialBench(ImageMCQDataset):
    """
    EmbSpatial-Bench.

    Reference:
      EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models
      https://arxiv.org/abs/2406.05756
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        'EmbSpatialBench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/EmbSpatialBench.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'EmbSpatialBench': 'a836cfd8fbe84bb42928ecef1e62ad32'
    }

    def _task_category(self):
        return [
            'mp3d',
            'ai2thor',
            'scannet'
        ]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = line['options']

        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = ast.literal_eval(options)

        upper_letters = list(string.ascii_uppercase)
        option_text = '\n'.join(
            f'{upper_letters[i]}: {options[i]}'
            for i in range(len(options))
        )

        # EmbSpatial has not yet released official inference code,
        # We use the SiteBench prompt format here.
        prompt = ''
        prompt += 'Question: ' + question + '\n'
        prompt += 'Options:\n' + option_text + '\n'
        post_prompt = 'Give me the answer letter directly. The best answer is:'
        prompt += post_prompt

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='data_source',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'EmbSpatialBench')
        )
