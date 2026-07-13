import os.path as osp
import string
import warnings

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge
from .utils.multiple_choice import eval_circular_group, eval_vanilla, report_acc
from .utils.shortqa import ShortQA_prompt


def ShortQA_auxeval(model, line):
    def proc_str(s):
        chs = set(s)
        chs = [x for x in chs if x not in string.ascii_letters + ': ']
        for ch in chs:
            s = s.replace(ch, ' ')
        return s

    def extraction(resp):
        correct, reason = None, None
        correct_st, correct_ed = '[Begin Correctness]', '[End Correctness]'
        reason_st, reason_ed = '[Begin Reason]', '[End Reason]'
        if correct_st in resp and correct_ed in resp:
            correct = resp.split(correct_st)[1].split(correct_ed)[0].strip().lower()
            if ('yes' in correct) ^ ('no' in correct):
                correct = 1 if 'yes' in correct else 0
                if reason_st in resp and reason_ed in resp:
                    reason = resp.split(reason_st)[1].split(reason_ed)[0].strip()
                return correct, reason
            else:
                return None, None
        else:
            return None, None

    prompt = ShortQA_prompt(line)
    retry = 3
    for i in range(retry):
        output = model.generate(prompt, temperature=0.5 * i)
        ans = extraction(output)
        # print(output, ans)
        if ans[0] in [0, 1]:
            return dict(hit=ans[0], log=ans[1])

    return dict(hit=0, log='Fail to Judge')


def Comprehensive_auxeval(model, data):
    def valid(record, key_name):
        return key_name in record and (not pd.isna(record[key_name])) and record[key_name] != ''

    if isinstance(data, pd.DataFrame) and len(data) > 1:
        # Should Adopt CircularEval
        assert valid(data.iloc[0], 'A')
        data['GT'] = data['answer']
        return eval_circular_group(model, data)
    else:
        item = data.iloc[0] if isinstance(data, pd.DataFrame) else data
        if valid(item, 'A') and len(item['answer']) == 1:
            item['GT'] = item['answer']
            return eval_vanilla(model, item)
        else:
            return ShortQA_auxeval(model, item)


class ImageShortQADataset(ImageBaseDataset):
    TYPE = 'Short'

    DATASET_URL = {
        'LiveMMBench_Infographic': '',
        'LiveMMBench_Perception': '',
        'LiveMMBench_Reasoning': '',
        'LiveMMBench_Reasoning_circular': '',
        'LiveMMBench_Spatial': '',
        'hle': 'https://opencompass.openxlab.space/utils/VLMEval/hle.tsv',
    }

    DATASET_MD5 = {
        'hle': 'a83cbdbea89f27c2aa5b8f34a8894b72',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\nPlease directly provide a short answer to the question. '
        return msgs

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        _ = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        storage = get_intermediate_file_path(eval_file, '_judge')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model = judge_kwargs.pop('model', 'gpt-4o-mini')
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(model=model, **judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

            if model is not None:
                if 'g_index' not in data:
                    lines = [data.iloc[i] for i in range(len(data))]
                    indices = [x['index'] for x in lines if x['index'] not in ans_map]
                    lines = [x for x in lines if x['index'] not in ans_map]
                    tups = [(model, line) for line in lines]
                else:
                    main_data = data[[x == y for x, y in zip(data['index'], data['g_index'])]]
                    lines = [data[data['g_index'] == x] for x in main_data['index']]
                    indices = [x.iloc[0]['g_index'] for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    lines = [x for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    tups = [(model, x) for x in lines]
                    data = main_data

                if len(lines):
                    res = track_progress_rich(
                        Comprehensive_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['hit'] = [x['hit'] for x in judge_results]
            data['log'] = [x['log'] for x in judge_results]
            dump(data, storage)

        data = load(storage)
        acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)
        return acc


class PathVQA_VAL(ImageShortQADataset):
    DATASET_URL = {
        'PathVQA_VAL': 'https://huggingface.co/datasets/Pfei111/PathVQA/resolve/main/PathVQA_VAL.tsv',
    }

    DATASET_MD5 = {
        'PathVQA_VAL': None,
    }


class PathVQA_TEST(ImageShortQADataset):
    DATASET_URL = {
        'PathVQA_TEST': 'https://huggingface.co/datasets/Pfei111/PathVQA/resolve/main/PathVQA_TEST.tsv',
    }

    DATASET_MD5 = {
        'PathVQA_TEST': None,
    }
