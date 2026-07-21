import ast
import os

from huggingface_hub import snapshot_download
from tqdm import tqdm

from vlmeval.smp import get_cache_path, modelscope_flag_set, toliststr
from .image_base import ImageBaseDataset


class SPBench(ImageBaseDataset):
    """
    SPBench.

    Reference:
      SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models
      https://arxiv.org/abs/2510.08531
    """

    TYPE = 'MCQ'

    # Prompt template directly from SPBench codebase:
    # https://github.com/ZJU-REAL/SpatialLadder/blob/main/eval_spld/data_utils/vsi_utils.py
    THINKING_TEMPLATE = (
        "Question: {question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "  # noqa: E501
        "It's encouraged to include self-reflection or verification in the reasoning process. \n"
    )

    PROMPT_TEMPLATES = {
        'default': {
            'pre_prompt': 'Question: {question}\n',
            'mca_post_prompt': "Please answer with the option's letter from the given choices (e.g., A, B, etc.) directly.",  # noqa: E501
            'na_post_prompt': 'Please answer the question using a numerical value (e.g., 42 or 3.1) directly.',
        },
        'thinking': {
            'pre_prompt': THINKING_TEMPLATE,
            'mca_post_prompt': (
                'Please provide your detailed reasoning between the <think> </think> tags, '
                "and then answer the question with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags."  # noqa: E501
            ),
            'na_post_prompt': (
                'Please provide your detailed reasoning between the <think> </think> tags, '
                'and then answer the question with a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.'  # noqa: E501
            ),
        },
    }

    DATASET_URL = {
        'SPBench-MV': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SPBench-MV.tsv',
        'SPBench-MV_CoT': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SPBench-MV.tsv',  # noqa: E501
        'SPBench-SI': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SPBench-SI.tsv',
        'SPBench-SI_CoT': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SPBench-SI.tsv',  # noqa: E501
    }

    DATASET_MD5 = {
        'SPBench-MV': '33c5fc89d6c431164c538d811e23b06e',
        'SPBench-MV_CoT': '33c5fc89d6c431164c538d811e23b06e',
        'SPBench-SI': 'a9467c2932993fe4487af25472326bb2',
        'SPBench-SI_CoT': 'a9467c2932993fe4487af25472326bb2',
    }

    def __init__(self, dataset, skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

        self._CATEGORY_TASK_ORDER = None
        self.use_cot = self.parse_dataset_name(dataset)
        print(f'Evaluate {dataset} with CoT = {self.use_cot}')

    @staticmethod
    def parse_dataset_name(name: str) -> bool:
        if not isinstance(name, str):
            return False

        lower = name.lower()
        return lower.endswith('_cot')

    def get_task_type(self, question_type):
        mcq_items = [
            'object_rel_direction',
            'object_rel_distance'
        ]

        na_items = [
            'object_counting',
            'object_abs_distance',
            'object_size_estimation'
        ]

        if question_type in mcq_items:
            return 'MCQ'
        elif question_type in na_items:
            return 'NA'
        else:
            raise ValueError(f'Unknown question type: {question_type}')

    def download_spbench(self, repo_id='hongxingli/SPBench'):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.spbench_extracted'

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

                            rel = os.path.normpath(info.filename).lstrip('/\\\\')
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
                print('SPBench data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        return dataset_path

    def prepare_tsv(self, url, file_md5=None):
        data = super().prepare_tsv(url, file_md5)

        dataset_path = self.download_spbench()

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\\/')))

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
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = line['candidates']

        if options is None:
            options = []
        elif isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except Exception:
                options = [options] if options.strip() else []
        elif not isinstance(options, (list, tuple)):
            options = [options]

        if options:
            question += '\nOptions:\n' + '\n'.join(options)

        question_type = line['question_type']
        task_type = self.get_task_type(question_type)

        # Prompt format in SPBench codebase
        prompt_type = 'thinking' if self.use_cot else 'default'
        prompt_template = self.PROMPT_TEMPLATES.get(prompt_type)

        prompt_text = prompt_template['pre_prompt'].format(question=question)
        if task_type == 'MCQ':
            prompt_text += '\n' + prompt_template['mca_post_prompt']
        elif task_type == 'NA':
            prompt_text += '\n' + prompt_template['na_post_prompt']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .vsibench import VsiBench

        # Reuse VsiBench.evaluate because SPBench and VsiBench share the same metric computation logic.
        if not hasattr(type(self), '_aggregate'):
            type(self)._aggregate = VsiBench._aggregate
        return VsiBench.evaluate(self, eval_file, **judge_kwargs)
