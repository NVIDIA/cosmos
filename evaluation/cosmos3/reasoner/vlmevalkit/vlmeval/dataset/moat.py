import json
import os
import os.path as osp
import warnings
import zipfile
from random import seed, shuffle

import pandas as pd

from vlmeval.dataset.utils import build_judge
from vlmeval.smp import (LMUDataRoot, download_file, dump, get_intermediate_file_path, get_logger,
                         load, md5, toliststr)
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE

logger = get_logger(__name__)


RANDOM_SEED = 0

VQA_SYSTEM_PROMPT = json.dumps({
    'task': 'Answer the question presented to you truthfully.',
    'requirements': [
        'Analyze the image(s) first, then answer the question. If you are given a list of possible answers, you must choose from it.',  # noqa: E501
        'You must answer in the following json format: {"analysis": "(write your analysis here)", "answer": "(your answer)"}'  # noqa: E501
    ]
})

EVAL_SYSTEM_PROMPT = json.dumps({
    'task': 'Evaluate whether the answer to a question is correct.',
    'requirements': [
        'Compare an answer to a question with the ground truth answer. Determine whether it is correct.',
        'You must ignore any analysis of the problem if present. You must focus only on the final answer.',
        'You must answer in the following json format: {"verdict": "(1 for correct, 0 for incorrect)"}'
    ]
})


def str2json(s: str):
    try:
        return json.loads(s)
    except Exception:
        if s.startswith('```json') and s.endswith('```'):
            return json.loads(s[6:-3])
        return s


class MOAT(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MOAT': "https://huggingface.co/datasets/waltsun/MOAT/resolve/main/MOAT.tsv",
    }
    DATASET_MD5 = {
        'MOAT': '803b5a176a5b01aa1b8094fae73532a2',
    }

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        seed(RANDOM_SEED)  # seed the random number generator to ensure reproducibility

    def post_build(self, dataset):
        assert dataset == "MOAT", f"Wrong dataset name {dataset}"
        ROOT = LMUDataRoot()
        os.makedirs(ROOT, exist_ok=True)

        # Download images zip file
        zip_url = "https://huggingface.co/datasets/waltsun/MOAT/resolve/main/MOAT_images.zip"
        zip_md5 = 'c0818a3e0a3f0bc7ee2be89ff04d73a6'
        zip_path = osp.join(ROOT, "MOAT_images.zip")
        if osp.exists(zip_path) and md5(zip_path) == zip_md5:
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(zip_url, zip_path)

        # Extract images
        self.img_root = osp.join(ROOT, 'MOAT_images')
        if not osp.exists(self.img_root):
            with zipfile.ZipFile(osp.join(ROOT, 'MOAT_images.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.img_root)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question, choices, images, outside_knowledge_text, outside_knowledge_images = line['question'], line['choices'], line['images'], line['outside_knowledge_text'], line['outside_knowledge_images']  # noqa: E501
        choices, images, outside_knowledge_images = toliststr(choices), toliststr(images), toliststr(outside_knowledge_images)  # noqa: E501

        if len(choices):
            shuffle(choices)        # shuffle the choices to avoid bias
            question += f'\nThe choices are: {choices}'
        msgs = [
            {
                'type': 'text',
                'value': VQA_SYSTEM_PROMPT + '\n' + question,
            },
        ]
        for img in images:
            msgs.append({'type': 'image', 'value': osp.join(self.img_root, img)})
        if not pd.isna(outside_knowledge_text):
            msgs.append({'type': 'text', 'value': 'Hint:\n' + outside_knowledge_text})
        for img in outside_knowledge_images:
            msgs.append({'type': 'image', 'value': osp.join(self.img_root, img)})
        return msgs

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs['model']
        result_path = get_intermediate_file_path(eval_file, f"_{model}")
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(result_path):
            data = load(eval_file)
            model = build_judge(**judge_kwargs)
            assert model.working(), ('MOAT evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            def extract_prediction(s: str):
                try:
                    return json.loads(s)['answer']
                except Exception:
                    pass
                try:
                    if s.startswith('```json') and s.endswith('```'):
                        return json.loads(s[7:-3])['answer']
                except Exception:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", s)
                    return s

            def extract_verdict(s: str):
                try:
                    return json.loads(s)['verdict']
                except Exception:
                    if s.startswith('```json') and s.endswith('```'):
                        return json.loads(s[7:-3])['verdict']
                    raise

            def verdict_one(model, line):
                prediction = extract_prediction(line['prediction'])
                answer = line['answer']
                prompt = EVAL_SYSTEM_PROMPT + '\n' + f'The answer to evaluate is {prediction}\nThe ground truth answer is {answer}'  # noqa: E501
                res = model.generate(prompt)
                return extract_verdict(res)

            verdict_list = track_progress_rich(
                lambda line: verdict_one(model, line),
                [data.iloc[i] for i in range(len(data))],
                nproc=nproc,
                chunksize=nproc,
            )
            data['verdict'] = verdict_list
            dump(data, result_path)

        data = load(result_path)
        overall_acc = data['verdict'].mean()
        capability_set = set()
        for i in range(len(data)):
            capability_set.update(toliststr(data.iloc[i]['capability']))
        capability_score_map = {capability: (0, 0) for capability in capability_set}
        for i in range(len(data)):
            line = data.iloc[i]
            capabilities = toliststr(line['capability'])
            verdict = line['verdict']
            for capability in capabilities:
                capability_score_map[capability] = (
                    capability_score_map[capability][0] + verdict, capability_score_map[capability][1] + 1
                )
        capability_score_map = {capability: score[0] / score[1] for capability, score in capability_score_map.items()}
        metrics = {
            'overall_acc': overall_acc,
            'result_path': result_path,
            'capability_acc': capability_score_map,
        }
        score_pth = get_intermediate_file_path(eval_file, "_score", "json")
        dump(metrics, score_pth)

        return metrics
