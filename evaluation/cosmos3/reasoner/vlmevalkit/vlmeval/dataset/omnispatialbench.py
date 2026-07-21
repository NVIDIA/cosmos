import ast
import os
import string
from collections import OrderedDict

import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

from vlmeval.smp import get_cache_path, load, toliststr
from .image_mcq import ImageMCQDataset

# Prompt template adapted from the official OmniSpatial codebase:
# https://github.com/qizekun/OmniSpatial/tree/main
RE_FORMAT = """
End your answer with a separate line formatted exactly as:

Answer: X
where X ∈ {A, B, C, D}.
"""


DEFAULT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""


ZERO_SHOT_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Think step by step and provide the answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""


MANUAL_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Guidelines
----------
Please follow these steps to analyze the image and answer the question:
1. First, carefully observe the image and identify all relevant objects and their spatial relationships.
2. Next, break down the question into key components that need to be addressed.
3. Think through the spatial reasoning step-by-step to arrive at your answer. It may be necessary to transfer perspective to better understand the scene.   # noqa: E501
4. Finally, select the most appropriate option (A, B, C, or D) based on your analysis.

Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""


class OmniSpatialBench(ImageMCQDataset):
    """
    OmniSpatial.

    Reference:
      OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
      https://arxiv.org/abs/2506.03135
    """

    TYPE = 'MCQ'

    OMNI_TSV_URL = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/OmniSpatialBench.tsv'
    OMNI_TSV_MD5 = '5d0896fc57c452055b020cc309ed799b'

    VARIANTS = [
        'OmniSpatialBench',
        'OmniSpatialBench_default',
        'OmniSpatialBench_zeroshot_cot',
        'OmniSpatialBench_manual_cot',
    ]

    DATASET_URL = {}
    DATASET_MD5 = {}

    for name in VARIANTS:
        DATASET_URL[name] = OMNI_TSV_URL
        DATASET_MD5[name] = OMNI_TSV_MD5

    SYS_PROMPTS = {
        'default': DEFAULT_SYSTEM_PROMPT,
        'zeroshot_cot': ZERO_SHOT_COT_SYSTEM_PROMPT,
        'manual_cot': MANUAL_COT_SYSTEM_PROMPT,
    }

    CATEGORY_TASK_ORDER = OrderedDict([
        ('Dynamic_Reasoning', ['Manipulation', 'Motion_Analysis']),
        ('Spatial_Interaction', ['Traffic_Analysis', 'Localization', 'Geospatial_Strategy']),
        ('Complex_Logic', ['Pattern_Recognition', 'Geometric_Reasoning']),
        ('Perspective_Taking', ['Egocentric', 'Allocentric', 'Hypothetical']),
    ])

    def __init__(self, dataset, skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.prompt_mode = self.parse_dataset_name(dataset)

    def parse_dataset_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ''

        lower = name.lower()

        for key in self.SYS_PROMPTS.keys():
            if lower.endswith(f'_{key}'.lower()):
                return key

        return ''

    def prepare_tsv(self, url, file_md5=None, repo_id='qizekun/OmniSpatial'):
        data = super().prepare_tsv(url, file_md5)

        SENTINEL_NAME = '.omnispatialbench_extracted'
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
                print('OmniSpatial data extracted to current directory with original layout.')

            dataset_path = snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',
                revision='main',
                allow_patterns=['OmniSpatial-test.zip'],
            )

            unzip_hf_zip(dataset_path)

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, 'OmniSpatial-test', s.lstrip(r'\/')))

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

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        option_text = ''
        for key, item in options.items():
            option_text += f'\n{key}. {item}'

        # prompt format from OmniSpatial codebase
        if self.prompt_mode in self.SYS_PROMPTS.keys():
            system_prompt = self.SYS_PROMPTS[self.prompt_mode]
            prompt = system_prompt + '\n' + RE_FORMAT + '\n\n' + question + option_text

        # EASI also provides direct QA format
        else:
            prompt = question + option_text + '\nAnswer directly with the option letter from the given choices. '

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

        raw = eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col=['task_type', 'sub_task_type'],
            order={
                'task_type': list(self.CATEGORY_TASK_ORDER.keys()),
                'sub_task_type': sum(self.CATEGORY_TASK_ORDER.values(), []),
            },
            dataset_name=getattr(self, 'dataset_name', 'OmniSpatialBench'),
        )

        pretty = OrderedDict()
        pretty['overall'] = raw['overall']

        for cat, tasks in self.CATEGORY_TASK_ORDER.items():
            for t in tasks:
                k = f'task.{t}_accuracy'
                if k in raw:
                    pretty[f'{t}_accuracy'] = raw[k]
            cat_key = f'{cat}_accuracy'
            if cat_key in raw:
                pretty[cat_key] = raw[cat_key]

        keys_str = ', '.join(pretty.keys())
        vals_str = ', '.join(f'{v:.3f}' for v in pretty.values())
        pretty['tabulated_keys'] = keys_str
        pretty['tabulated_results'] = vals_str

        return pretty
