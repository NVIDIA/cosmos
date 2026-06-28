import os.path as osp
import re
from collections import defaultdict

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load, toliststr
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

SYSTEM_PROMPT = """\
Please act as an impartial evaluator and assess the quality of the responses provided by two AI assistants to a given user prompt and accompanying image. You will be provided with Assistant A's and Assistant B's answers. Your task is to determine which assistant's response is superior.

Start your evaluation by generating your own answer to the prompt and image. Ensure that you complete your answer before reviewing any assistant responses.

When evaluating the assistants' responses, compare each one to your own answer.

First, assess whether the assistants' answers are helpful and relevant. A response is considered helpful if it appropriately addresses the prompt, follows the given instructions, and is well-organized. A relevant answer closely aligns with the context or requirements of the prompt.

When applicable, consider the creativity and novelty of each assistant's response and evaluate the writing quality of both responses.

Then, identify and correct any errors or inaccuracies in the assistants' answers. Lastly, identify any critical information missing from the assistants' responses that should have been included to improve the answer.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""  # noqa: E501

SYSTEM_PROMPT_GT = """\
Please act as an impartial evaluator and assess the quality of the responses provided by two AI assistants to a given user prompt and accompanying image. You will be provided with Assistant A's and Assistant B's answers. Your task is to determine which assistant's response is superior.

Start your evaluation by generating your own answer to the prompt and image. Ensure that you complete your answer before reviewing any assistant responses.

When evaluating the assistants' responses, compare each one to your own answer.

First, assess whether the assistants' answers are helpful and relevant. A response is considered helpful if it appropriately addresses the prompt, follows the given instructions, and is well-organized. A relevant answer closely aligns with the context or requirements of the prompt.

When applicable, consider the creativity and novelty of each assistant's response and evaluate the writing quality of both responses.

Then, identify and correct any errors or inaccuracies in the assistants' answers. Lastly, identify any critical information missing from the assistants' responses that should have been included to improve the answer. Please refer to the provided Ground Truth answer, which constitutes the key fact relevant to the question.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""  # noqa: E501

PROMPT_TEMPLATE = """**INPUT**:

<|User Prompt|>\n{question}

<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>
"""


PROMPT_TEMPLATE_GT = """**INPUT**:

<|User Prompt|>\n{question}

<|Ground Truth|>\n{gt}

<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>
"""


REGEX_PATTERN = re.compile("\[\[([AB<>=]+)\]\]")  # noqa: W605


def get_score(judgement, pattern=REGEX_PATTERN):
    matches = pattern.findall(judgement)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        return matches[0].strip("\n"), False
    else:
        return None, True


def MMAlignBench_auxeval(model, line):
    if 'gt' in line and str(line['gt']) != 'nan':
        config = dict(question=line['question'], gt=line['gt'], answer_1=line['A'], answer_2=line['B'])
        prompt = SYSTEM_PROMPT_GT + '\n' + PROMPT_TEMPLATE_GT.format(**config)
        # prompt = PROMPT_TEMPLATE.format(**config)
        print('gt_prompt' + prompt)
    else:
        config = dict(question=line['question'], answer_1=line['A'], answer_2=line['B'])
        prompt = SYSTEM_PROMPT + '\n' + PROMPT_TEMPLATE.format(**config)
        # prompt = PROMPT_TEMPLATE.format(**config)
        print('prompt' + prompt)

    prefix = 'data:image/jpeg;base64,'
    img = prefix + line['image']

    messages = [
        dict(type='text', value=prompt),
        dict(type='image', value=img)
    ]

    retry = 2
    while retry:
        resp = model.generate(messages)
        score, try_again = get_score(resp)
        if not try_again:
            break
        retry -= 1

    if score is None:
        return 'Unknown'
    return [score, resp]


class MMAlignBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'MMAlignBench': 'https://opencompass.openxlab.space/utils/VLMEval/MMAlignBench.tsv'}
    DATASET_MD5 = {'MMAlignBench': 'd00d8e61c99257cbaf76d8d5e926f01e'}

    score_map = {
        'A>>B': -2,
        'A>B': -1,
        'A=B': 0,
        'B>A': 1,
        'B>>A': 2
    }

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        # WildVision adopts text first
        msgs = [dict(type='text', value=question)] + msgs
        return msgs

    @classmethod
    def gen_eval_base(self, eval_file, b64_map):
        data = load(eval_file)
        data['B'] = data.pop('prediction')
        data['A'] = data.pop('claude3_sonnet')
        data['image'] = [b64_map[x] for x in data['index']]
        return data

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        # We adopt pairwise evaluation (twice for a pair) for this dataset
        model = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'csv')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            raw_data = MMAlignBench('MMAlignBench').data
            b64_map = {x: y for x, y in zip(raw_data['index'], raw_data['image'])}
            data = self.gen_eval_base(eval_file, b64_map)

            # judge_kwargs['system_prompt'] = SYSTEM_PROMPT
            judge_kwargs['temperature'] = 0
            judge_kwargs['img_detail'] = 'high'
            judge_kwargs['timeout'] = 300
            model = build_judge(max_tokens=4096, **judge_kwargs)

            assert model.working(), (
                'MMAlignBench evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            )

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMAlignBench_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    ans[k] = {'score': v[0], 'resp': v[1]}
            else:
                for k, v in ans.items():
                    ans[k] = {'score': v[0], 'resp': v[1]}
            # breakpoint()
            data['score'] = [ans[x]['score'] for x in data['index']]
            data['judge'] = [ans[x]['resp'] for x in data['index']]
            data.pop('image')
            dump(data, storage)

        data = load(storage)
        lt = len(data)

        scores = defaultdict(lambda: 0)
        type_scores = defaultdict(lambda: defaultdict(lambda: 0))

        for i in range(lt):
            item = data.iloc[i]
            if item['score'] not in self.score_map:
                score = 0
            else:
                score = self.score_map[item['score']]
                if '_rev' in item['index']:
                    score = -score
            scores[score] += 1
            type = item['type']
            type_scores[type][score] += 1

        name_map = {
            2: 'Much Better',
            1: 'Better',
            0: 'Tie',
            -1: 'Worse',
            -2: 'Much Worse'
        }
        scores = {name_map[k]: v for k, v in scores.items()}
        scores['Reward'] = (
            100 * scores.get('Much Better', 0)
            + 50 * scores.get('Better', 0)
            - 50 * scores.get('Worse', 0)
            - 100 * scores.get('Much Worse', 0)
        ) / lt
        scores['Win Rate'] = (scores.get('Better', 0) + scores.get('Much Better', 0)) / lt
        scores = {k: [v] for k, v in scores.items()}
        scores = pd.DataFrame(scores)

        for type_name, type_score_dict in type_scores.items():
            type_score_dict = {name_map[k]: v for k, v in type_score_dict.items()}
            type_lt = sum(type_score_dict.values())

            type_score_dict['Reward'] = (
                (
                    100 * type_score_dict.get('Much Better', 0)
                    + 50 * type_score_dict.get('Better', 0)
                    - 50 * type_score_dict.get('Worse', 0)
                    - 100 * type_score_dict.get('Much Worse', 0)
                )
                / type_lt
                if type_lt > 0
                else 0
            )

            type_score_dict['Win Rate'] = (
                (type_score_dict.get('Better', 0) + type_score_dict.get('Much Better', 0)) / type_lt
                if type_lt > 0
                else 0
            )

            # 将该类型的得分添加到结果中
            type_score_df = pd.DataFrame(
                {
                    f"{type_name}_Much Better": [type_score_dict.get('Much Better', 0)],
                    f"{type_name}_Better": [type_score_dict.get('Better', 0)],
                    f"{type_name}_Tie": [type_score_dict.get('Tie', 0)],
                    f"{type_name}_Worse": [type_score_dict.get('Worse', 0)],
                    f"{type_name}_Much Worse": [type_score_dict.get('Much Worse', 0)],
                    f"{type_name}_Reward": [type_score_dict['Reward']],
                    f"{type_name}_Win Rate": [type_score_dict['Win Rate']],
                }
            )
            scores = pd.concat([scores, type_score_df], axis=1)

        dump(scores, score_file)
        return scores
