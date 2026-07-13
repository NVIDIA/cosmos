import os
import os.path as osp
import string
from collections import defaultdict

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

PROMPT_MCQ_MULTIMODAL = {
    "instruct": [
        {
            "type":
            "instruction",
            "value":
            "The following is a question about chemistry. Please answer by responding with the letter(s) of the correct answer(s). Separate multiple answers with a comma.",  # noqa: E501
        },
        {
            "type": "instruction",
            "value": "Question:"
        },
        {
            "type": "question",
            "value": None
        },
        {
            "type": "answer",
            "value": None
        },
        {
            "type":
            "text",
            "value":
            "You MUST include the final answer within the following tags: [ANSWER] and [/ANSWER].For example, '[ANSWER]<answer>[/ANSWER].",  # noqa: E501
        },
        {
            "type":
            "text",
            "value":
            "For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is comma- or space-separated list of the correct letters. Always answer in exactly this format of comma-separated letters between the two tags, even if you are unsure. We require this because we use automatic parsing.",  # noqa: E501
        },
        {
            "type": "constraint",
            "value": "Answer: "
        },
    ],
}

PROMPT_FLOATQ_MULTIMODAL = {
    "instruct": [
        {
            "type":
            "instruction",
            "value":
            "The following is a question about chemistry or material science. Please return the answer as a number."
        },
        {
            "type": "instruction",
            "value": "Question:"
        },
        {
            "type": "question",
            "value": None
        },
        {
            "type":
            "constraint",
            "value":
            "You MUST include the final answer within the following tags: [ANSWER] and [/ANSWER]. For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is only one number. Always answer in exactly this format, with ONE NUMBER between the two tags, even if you are unsure. We require this because we use automatic parsing.",  # noqa: E501
        },
        {
            "type": "constraint",
            "value": "Answer: "
        },
    ],
}

EXTRACTOR_MCQ_PROMPT = """From the following answer, please extract the answer in the format of [ANSWER]<answer>[/ANSWER]. For example, if the answer is A, the completion should be [ANSWER]A[/ANSWER].
The answer is one or multiple comma-separated single letters of the alphabet, never full words. You must return the answer in the format of [ANSWER]<answer>[/ANSWER].
Only return the letters of the correct answer. If there is no correct answer, return nothing, i.e. [ANSWER][/ANSWER].{options}

You need to extract the answer from the next text:

{completion}"""  # noqa: E501

EXTRACTOR_EXACT_MATCH_PROMPT = """From the following answer, please extract the answer in the format of [ANSWER]<answer>[/ANSWER]. For example, if the answer is 2, the completion should be [ANSWER]2[/ANSWER].
The answer is one number. Do not use roman numbers or any special symbols. Return then as integer or float. You must return the answer in the format of [ANSWER]<answer>[/ANSWER].
Only return the number of the correct answer. If there is no correct answer, return nothing, i.e. [ANSWER][/ANSWER].

You need to extract the answer from the next text:

{completion}"""  # noqa: E501


def join_append(messages: list[dict], new: dict, join_text=''):
    if len(messages) == 0:
        messages.append(new)
    elif messages[-1]['type'] == 'text' and new['type'] == 'text':
        messages[-1]['value'] = ''.join(
            [messages[-1]['value'], join_text, new['value']])
    else:
        messages.append(new)
    return messages


def macbench_auxeval(model, line):
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }

    if options:
        # mcq
        option_text = '\nThe options are:\n' + ', '.join(
            f'{letter}: {option}' for letter, option in options.items())
        prompt = EXTRACTOR_MCQ_PROMPT.format(completion=line['prediction'],
                                             options=option_text)
    else:
        prompt = EXTRACTOR_EXACT_MATCH_PROMPT.format(
            completion=line['prediction'])
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)
        final_answer = res.partition('[ANSWER]')[-1].partition('[/ANSWER]')[0]

        if '[ANSWER]' in res and '[/ANSWER]' in res:
            log += 'Succeed'
            return dict(log=log, res=final_answer)
        else:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


class MaCBench(ImageBaseDataset):
    TYPE = 'MaCBench'
    DATASET_URL = {'MaCBench': ''}
    DATASET_MD5 = {'MaCBench': '0e163396dd28886fd828e101f24afdf6'}

    def __init__(self, dataset='MMBench', skip_noimg=True, interleave=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.interleave = interleave

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        prefix, _, suffix = question.partition('{image}')
        tgt_path = self.dump_image(line)

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        msgs = []
        if options:
            # multiple-choice questions
            for item in PROMPT_MCQ_MULTIMODAL['instruct']:
                if item['type'] == 'question':
                    join_append(msgs, dict(type='text', value=prefix), ' ')
                    for img_path in tgt_path:
                        join_append(msgs, dict(type='image', value=img_path))
                    join_append(msgs, dict(type='text', value=suffix))
                elif item['type'] == 'answer':
                    option_text = '\n'.join(
                        f'{cand}: {option}'
                        for cand, option in options.items())
                    join_append(msgs, dict(type='text', value=option_text),
                                '\n')
                else:
                    join_append(msgs, dict(type='text', value=item['value']),
                                '\n')
        else:
            # single target quesiton
            for item in PROMPT_FLOATQ_MULTIMODAL['instruct']:
                if item['type'] == 'question':
                    join_append(msgs, dict(type='text', value=prefix), ' ')
                    for img_path in tgt_path:
                        join_append(msgs, dict(type='image', value=img_path))
                    join_append(msgs, dict(type='text', value=suffix))
                else:
                    join_append(msgs, dict(type='text', value=item['value']),
                                '\n')

        if not self.interleave:
            new_msgs = []
            for img_path in tgt_path:
                join_append(new_msgs, dict(type='image', value=img_path))
            for msg in msgs:
                if msg['type'] == 'text':
                    join_append(new_msgs, dict(type='text',
                                               value=msg['value']), ' ')
                elif msg['type'] == 'image':
                    join_append(new_msgs, dict(type='text', value='<IMAGE>'),
                                ' ')
            msgs = new_msgs
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        if 'model' in judge_kwargs:
            model = judge_kwargs['model']
        else:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(
            ), 'MaCBench evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    macbench_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = self.cal_acc(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        dump(score, score_pth)
        return score

    def cal_acc(self, storage):
        data = load(storage)
        group_total = defaultdict(int)
        group_hit = defaultdict(int)

        for _, line in data.iterrows():
            group_total[line['category']] += 1
            pred = line['res'].strip() if isinstance(line['res'],
                                                     str) else line['res']
            if not pred:
                continue

            if 'A' in line and not pd.isna(line['A']):
                # mcq
                if pred == line['answer']:
                    group_hit[line['category']] += 1
            else:
                try:
                    pred = float(pred)
                    answer = float(line['answer'])
                except Exception:
                    continue

                tolarnce = line.get('relative_tolerance', None)
                if not tolarnce:
                    tolarnce = answer * 0.01
                else:
                    tolarnce = float(tolarnce)
                if abs(float(pred) - answer) < tolarnce:
                    group_hit[line['category']] += 1

        result = defaultdict(list)
        result['Overall'].append(
            sum(group_hit.values()) / sum(group_total.values()))
        for group in group_total.keys():
            result[group].append(
                float(group_hit.get(group, 0) / group_total[group]))
        return pd.DataFrame(result)
